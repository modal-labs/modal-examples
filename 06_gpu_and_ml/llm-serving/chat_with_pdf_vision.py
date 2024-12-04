# # Chat with PDF: RAG with ColQwen2

# In this example, we demonstrate how to use the the [ColQwen2](https://huggingface.co/vidore/colqwen2-v0.1) model to build a simple
# "Chat with PDF" retrieval-augmented generation (RAG) app.
# The ColQwen2 model is based on [ColPali](https://huggingface.co/blog/manu/colpali) but uses the
# [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) vision-language model.
# ColPali is in turn based on the late-interaction embedding approach pioneered in [ColBERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075).

# Vision-language models with high-quality embeddings obviate the need for complex pre-processing pipelines.
# See [this blog post from Jo Bergum of Vespa](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/) for more.

# ## Setup

# First, we’ll import the libraries we need locally and define some constants.

import os
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

import modal

MINUTES = 60  # seconds

# ## Setting up dependenices

# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# We install the packages required for our application in those images.

model_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        [
            "git+https://github.com/illuin-tech/colpali.git@782edcd50108d1842d154730ad3ce72476a2d17d",  # we pin the commit id
            "hf_transfer==0.1.8",
            "qwen-vl-utils==0.0.8",
            "torchvision==0.19.1",
        ]
    )
)

# These dependencies are only installed remotely, so we can't import them locally.
# Use the `.imports` context manager to import them only on Modal instead.

with model_image.imports():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ## Downloading ColQwen2

# Vision-language models (VLMs) for embedding and generation add another layer of simplification
# to RAG apps based on vector search: we only need one model.
# Here, we use the Qwen2-VL-2B-Instruct model from Alibaba.
# The function below downloads the model from the Hugging Face Hub.


def download_model(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        model_name,
        local_dir=model_dir,
        revision=model_revision,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    move_cache()


# We can also include other files that our application needs in the container image.
# Here, we add the model weights to the image by executing our `download_model` function.

model_image = model_image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_function(
    download_model,
    timeout=20 * MINUTES,
    kwargs={
        "model_dir": "/model-qwen2-VL-2B-Instruct",
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "model_revision": "aca78372505e6cb469c4fa6a35c60265b00ff5a4",
    },
)

# ## Managing state with Modal Volumes and Dicts

# Chat services are stateful:
# the response to an incoming user message depends on past user messages in a session.

# RAG apps add even more state:
# the documents being retrieved from and the index over those documents,
# e.g. the embeddings.

# Modal Functions are stateless in and of themselves.
# They don't retain information from input to input.
# That's what enables Modal Functions to automatically scale up and down
# [based on the number of incoming requests](https://modal.com/docs/guide/cold-start).

# ### Managing chat sessions with Modal Dicts

# In this example, we use a [`modal.Dict`](https://modal.com/docs/guide/dicts-and-queues)
# to store state information between Function calls.

# Modal Dicts behave similarly to Python dictionaries,
# but they are backed by remote storage and accessible to all of your Modal Functions.
# They can contain any Python object
# that can be serialized using [`cloudpickle`](https://github.com/cloudpipe/cloudpickle).

# A Dict can hold a few gigabytes across keys of size up to 100 MiB,
# so it works well for our chat session state, which is a few KiB per session,
# and for our embeddings, with are a few hundred KiB per PDF page,
# up to about 100,000 pages of PDFs.

# At a larger scale, we'd need to replace this with a database, like Postgres,
# or push more state to the client.

sessions = modal.Dict.from_name("colqwen-chat-sessions", create_if_missing=True)


class Session:
    def __init__(self):
        self.images = None
        self.messages = []
        self.pdf_embeddings = None


# ### Storing PDFs on a Modal Volume

# Images extracted from PDFs are larger than our session state or embeddings
# -- low tens of MiB per page.

# So we store them on a [Modal Volume](https://modal.com/docs/guide/volumes),
# which can store terabytes (or more!) of data across tens of thousands of files.

# Volumes behave like a remote file system:
# we read and write from them much like a local file system.

pdf_volume = modal.Volume.from_name("colqwen-chat-pdfs", create_if_missing=True)
PDF_ROOT = Path("/vol/pdfs/")


# ## Defining a Chat with PDF service

# To deploy an autoscaling "Chat with PDF" vision-language model service on Modal,
# we just need to wrap our Python logic in a [Modal App](https://modal.com/docs/guide/apps):

# It uses [Modal `@app.cls`](https://modal.com/docs/guide/lifecycle-functions) decorators
# to organize the "lifecycle" of the app:
# to ensure all model files are downloaded (`@modal.build`)
# to load the model on container start (`@modal.enter`)
# and to run inference on request (`@modal.method`).

# We include in the arguments to the `@app.cls` decorator
# all the information about this service's infrastructure:
# the container image, the remote storage, and the GPU requirements.

app = modal.App("chat-with-pdf")


@app.cls(
    image=model_image,
    gpu=modal.gpu.A100(size="80GB"),
    container_idle_timeout=10 * MINUTES,  # spin down when inactive
    volumes={"/vol/pdfs/": pdf_volume},
)
class Model:
    @modal.build()
    @modal.enter()
    def load_models(self):
        self.colqwen2_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.colqwen2_processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v0.1"
        )
        self.qwen2_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16
        )
        self.qwen2_vl_model.to("cuda:0")
        self.qwen2_vl_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
        )

    @modal.method()
    def index_pdf(self, session_id, target: bytes | list):
        # We store concurrent user chat sessions in a modal.Dict

        # For simplicity, we assume that each user only runs one session at a time

        session = sessions.get(session_id)
        if session is None:
            session = Session()

        if isinstance(target, bytes):
            images = convert_pdf_to_images.remote(target)
        else:
            images = target

        # Store images on a Volume for later retrieval
        session_dir = PDF_ROOT / f"{session_id}"
        session_dir.mkdir(exist_ok=True, parents=True)
        for ii, image in enumerate(images):
            filename = session_dir / f"{str(ii).zfill(3)}.jpg"
            image.save(filename)

        # Generated embeddings from the image(s)
        BATCH_SZ = 4
        pdf_embeddings = []
        batches = [
            images[i : i + BATCH_SZ] for i in range(0, len(images), BATCH_SZ)
        ]
        for batch in batches:
            batch_images = self.colqwen2_processor.process_images(batch).to(
                self.colqwen2_model.device
            )
            pdf_embeddings += list(
                self.colqwen2_model(**batch_images).to("cpu")
            )

        # Store the image embeddings in the session, for later retrieval
        session.pdf_embeddings = pdf_embeddings

        # Write embeddings back to the modal.Dict
        sessions[session_id] = session

    @modal.method()
    def respond_to_message(self, session_id, message):
        session = sessions.get(session_id)
        if session is None:
            session = Session()

        pdf_volume.reload()  # make sure we have the latest data

        images = (PDF_ROOT / str(session_id)).glob("*.jpg")
        images = list(sorted(images, key=lambda p: int(p.stem)))

        # Nothing to chat about without a PDF!
        if not images:
            return "Please upload a PDF first"
        elif session.pdf_embeddings is None:
            return "Indexing PDF..."

        # RAG, Retrieval-Augmented Generation, is two steps:

        # _Retrieval_ of the most relevant data to answer the user's query
        relevant_image = self.get_relevant_image(message, session, images)

        # _Generation_ based on the retrieved data
        output_text = self.generate_response(message, session, relevant_image)

        # Update session state for future chats
        append_to_messages(message, session, user_type="user")
        append_to_messages(output_text, session, user_type="assistant")
        sessions[session_id] = session

        return output_text

    # Retrieve the most relevant image from the PDF for the input query
    def get_relevant_image(self, message, session, images):
        import PIL

        batch_queries = self.colqwen2_processor.process_queries([message]).to(
            self.colqwen2_model.device
        )
        query_embeddings = self.colqwen2_model(**batch_queries)

        # This scores our query embedding against the image embeddings from index_pdf
        scores = self.colqwen2_processor.score_multi_vector(
            query_embeddings, session.pdf_embeddings
        )[0]

        # Select the best matching image
        max_index = max(range(len(scores)), key=lambda index: scores[index])
        return PIL.Image.open(images[max_index])

    # Pass the query and retrieved image along with conversation history into the VLM for a response
    def generate_response(self, message, session, image):
        chatbot_message = get_chatbot_message_with_image(message, image)
        query = self.qwen2_vl_processor.apply_chat_template(
            [*session.messages, chatbot_message],
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info([chatbot_message])
        inputs = self.qwen2_vl_processor(
            text=[query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        generated_ids = self.qwen2_vl_model.generate(
            **inputs, max_new_tokens=512
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen2_vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text


# ## Loading PDFs as images

# Vision-Language Models operate on images, not PDFs directly,
# so we need to convert out PDFs into images first.

# We separate this from our indexing and chatting logic --
# we run on a different container with different dependencies.

pdf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils")
    .pip_install("pdf2image==1.17.0")
)


@app.function(image=pdf_image)
def convert_pdf_to_images(pdf_bytes):
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes, fmt="jpeg")
    return images


# ## Chatting with a PDF from the terminal

# Before deploying in a UI, we can test our service from the terminal.

# Just run
# ```bash
# modal run chat_with_pdf_vision.py
# ```

# and optionally pass in a path to or URL of a PDF with the `--pdf-path` argument
# and specify a question with the `--question` argument.

# Continue a previous chat by passing the session ID printed to the terminal at start
# with the `--session-id` argument.


@app.local_entrypoint()
def main(question: str = None, pdf_path: str = None, session_id: str = None):
    model = Model()
    if session_id is None:
        session_id = str(uuid4())
        print("Starting a new session with id", session_id)

        if pdf_path is None:
            pdf_path = "https://arxiv.org/pdf/1706.03762"  # all you need

        if pdf_path.startswith("http"):
            pdf_bytes = urlopen(pdf_path).read()
        else:
            pdf_path = Path(pdf_path)
            pdf_bytes = pdf_path.read_bytes()

        print("Indexing PDF from", pdf_path)
        model.index_pdf.remote(session_id, pdf_bytes)
    else:
        if pdf_path is not None:
            raise ValueError("Start a new session to chat with a new PDF")
        print("Resuming session with id", session_id)

    if question is None:
        question = "What is this document about?"

    print("QUESTION:", question)
    print(model.respond_to_message.remote(session_id, question))


# ## A hosted Gradio interface

# With the [Gradio](https://gradio.app) library, we can create a simple web interface around our class in Python,
# then use Modal to host it for anyone to try out.

# To deploy your own, run

# ```bash
# modal deploy chat_with_pdf_vision.py
# ```

# and navigate to the URL that appears in your teriminal.
# If you’re editing the code, use `modal serve` instead to see changes hot-reload.


web_image = pdf_image.pip_install(
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
    "gradio==4.44.1",
    "pillow==10.4.0",
    "gradio-pdf==0.0.15",
    "pdf2image==1.17.0",
)


@app.function(
    image=web_image,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 1000 concurrent inputs
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def ui():
    import uuid

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from gradio_pdf import PDF
    from pdf2image import convert_from_path

    web_app = FastAPI()

    # Since this Gradio app is running from its own container,
    # allowing us to run the inference service via .remote() methods.
    model = Model()

    def upload_pdf(path, session_id):
        if session_id == "" or session_id is None:
            # Generate session id if new client
            session_id = str(uuid.uuid4())

        images = convert_from_path(path)
        # Call to our remote inference service to index the PDF
        model.index_pdf.remote(session_id, images)

        return session_id

    def respond_to_message(message, _, session_id):
        # Call to our remote inference service to run RAG
        return model.respond_to_message.remote(session_id, message)

    with gr.Blocks(theme="soft") as demo:
        session_id = gr.State("")

        gr.Markdown("# Chat with PDF")
        with gr.Row():
            with gr.Column(scale=1):
                gr.ChatInterface(
                    fn=respond_to_message,
                    additional_inputs=[session_id],
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn=None,
                )
            with gr.Column(scale=1):
                pdf = PDF(
                    label="Upload a PDF",
                )
                pdf.upload(upload_pdf, [pdf, session_id], session_id)

    return mount_gradio_app(app=web_app, blocks=demo, path="/")


# ## Addenda

# The remainder of this code consists of utility functions and boiler plate used in the
# main code above.


def get_chatbot_message_with_image(message, image):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": message},
        ],
    }


def append_to_messages(message, session, user_type="user"):
    session.messages.append(
        {
            "role": user_type,
            "content": {"type": "text", "text": message},
        }
    )
