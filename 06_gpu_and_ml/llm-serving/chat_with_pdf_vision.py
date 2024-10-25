# ---
# deploy: true
# cmd: ["modal", "serve", "06_gpu_and_ml/llm-serving/chat_with_pdf_vision.py"]
# ---

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

import modal

MINUTES = 60  # seconds

# ## Downloading the Model

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

# ## Building the image

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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# These dependencies are only installed remotely, so we can't import them locally.
# Use the `.imports` context manager to import them only on Modal instead.

with model_image.imports():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# We can also include other files that our application needs in the container image.
# Here, we we add the model weights to the image by executing our `download_model` function.

model_image = model_image.run_function(
    download_model,
    timeout=MINUTES * 20,
    kwargs={
        "model_dir": "/model-qwen2-VL-2B-Instruct",
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "model_revision": "aca78372505e6cb469c4fa6a35c60265b00ff5a4",
    },
)

## Defining a Chat with PDF application

# Running an inference service on Modal is as easy as writing an inference function in Python.
# We just need to wrap that in a Modal App:

app = modal.App("chat-with-pdf")

# To allow concurrent users, each user chat session state is stored in a modal.Dict
sessions = modal.Dict.from_name("colqwen-chat-sessions", create_if_missing=True)

class Session:
    def __init__(self):
        self.images = None
        self.messages = []
        self.pdf_embeddings = None


# Here we define our on-GPU ColQwen2 service, which runs document indexing and inference
# It uses the [Modal @app.cls](https://modal.com/docs/guide/lifecycle-functions) feature to load the model on container start, and inference on request.
@app.cls(
    image=model_image,
    gpu=modal.gpu.A100(size="80GB"),
    container_idle_timeout=10 * MINUTES,  # spin down when inactive
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
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
        )
        self.qwen2_vl_model.to("cuda:0")
        self.qwen2_vl_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
        )

    @modal.method()
    def index_pdf(self, session_id, images):

        # We store concurrent user chat sessions in a modal.Dict

        # For simplicity, we assume that each session only runs one request at a time
        # This assumption lets us fetch the session, and write to it later, without fear of race conditions

        session = sessions.get(session_id)
        if session is None:
            session = Session()

        session.images = images

        # Generated embeddings from the image(s)
        batch_images = self.colqwen2_processor.process_images(images).to(
            self.colqwen2_model.device
        )
        pdf_embeddings = self.colqwen2_model(**batch_images)

        # Store the image embeddings in the session, for later RAG use
        session.pdf_embeddings = pdf_embeddings

        # Write session state, including embeddings, back to the modal.Dict
        sessions[session_id] = session

    @modal.method()
    def respond_to_message(self, session_id, message):
        if session_id not in sessions:
            sessions[session_id] = Session()
        session = sessions[session_id]

        # Nothing to chat about without a PDF!
        if session.images is None:
            return "Please upload a PDF first"
        elif session.pdf_embeddings is None:
            return "Indexing PDF..."

        # Retrieve the most relevant image from the PDF for the input query
        def get_relevant_image(message):
            batch_queries = self.colqwen2_processor.process_queries(
                [message]
            ).to(self.colqwen2_model.device)
            query_embeddings = self.colqwen2_model(**batch_queries)

            # This scores our query embedding against the image embeddings from index_pdf
            scores = self.colqwen2_processor.score_multi_vector(
                query_embeddings, session.pdf_embeddings
            )[0]

            # Return the best matching image
            max_index = max(range(len(scores)), key=lambda index: scores[index])
            return session.images[max_index]

        # Helper functions for chatbot formatting
        def get_chatbot_message_with_image(message, image):
            return {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": message},
                ],
            }

        def append_to_messages(message, user_type="user"):
            session.messages.append(
                {
                    "role": user_type,
                    "content": {"type": "text", "text": message},
                }
            )

        # Pass the query and retrieved image along with conversation history into the VLM for a response
        def generate_response(message, image):
            chatbot_message = get_chatbot_message_with_image(message, image)
            query = self.qwen2_vl_processor.apply_chat_template(
                [chatbot_message, *session.messages],
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
                **inputs, max_new_tokens=128
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

        # RAG, Retrieval-Augmented Generation, is two steps:

        # Retrieval of the most relevant image to answer the user's query
        relevant_image = get_relevant_image(message)

        # Generation, passing in the query and the retrieved image
        output_text = generate_response(message, relevant_image)

        # Update session state for future chats
        append_to_messages(message, user_type="user")
        append_to_messages(output_text, user_type="assistant")
        sessions[session_id] = session

        return output_text


# ## A hosted Gradio interface

# With the Gradio library, we can create a simple web interface around our class in Python,
# then use Modal to host it for anyone to try out.

# To deploy your own, run

# ```bash
# modal deploy chat_with_pdf_vision.py
# ```

# and navigate to the URL that appears in your teriminal.
# If you’re editing the code, use `modal serve` instead to see changes hot-reload.


web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils")
    .pip_install(
        "gradio==4.44.1",
        "pillow==10.4.0",
        "gradio-pdf==0.0.15",
        "pdf2image==1.17.0",
    )
)
@app.function(
    image=web_image,
    keep_warm=1,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
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
