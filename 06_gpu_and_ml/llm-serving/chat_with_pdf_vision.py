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
from fastapi import FastAPI

MINUTES = 60  # seconds

# ## Downloading the Model

# VLMs for embedding and generation add another layer of simplification to RAG apps based on vector search:
# we only need one model. Here, we use the Qwen2-VL-2B-Instruct model from Alibaba.
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

# We can also include other files our application needs in the container image.
# Here, that's the model weights, which we download by executing our `download_model` function.

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

# Running an inference service on Modal is as easy as writing inference in Python.

# We just need to wrap that in

app = modal.App("chat-with-pdf")


@app.cls(
    image=model_image,
    gpu=modal.gpu.A100(size="80GB"),
    container_idle_timeout=10 * MINUTES,  # spin down when inactive
)
class Model:
    # we stack the build and enter decorators here to ensure that the weights
    # downloaded in from_pretrained are cached in the image
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

        self.pdf_embeddings = None
        self.images = None
        self.messages = []

    @modal.method()
    def index_pdf(self, images):
        self.images = images
        batch_images = self.colqwen2_processor.process_images(images).to(
            self.colqwen2_model.device
        )
        self.pdf_embeddings = self.colqwen2_model(**batch_images)

    @modal.method()
    def respond_to_message(self, message):
        # nothing to chat about without a PDF!
        if self.images is None:
            return "Please upload a PDF first"
        elif self.pdf_embeddings is None:
            return "Indexing PDF..."

        # retrieve the most relevant image from the PDF for the input query
        def get_relevant_image(message):
            batch_queries = self.colqwen2_processor.process_queries(
                [message]
            ).to(self.colqwen2_model.device)
            query_embeddings = self.colqwen2_model(**batch_queries)
            scores = self.colqwen2_processor.score_multi_vector(
                query_embeddings, self.pdf_embeddings
            )[0]
            max_index = max(range(len(scores)), key=lambda index: scores[index])
            return self.images[max_index]

        # helper function to put message in the format chatbot
        def get_chatbot_message_with_image(message, image):
            return {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": message},
                ],
            }

        # helper function add messages to the conversation history in chatbot format
        def append_to_messages(message, user_type="user"):
            self.messages.append(
                {
                    "role": user_type,
                    "content": {"type": "text", "text": message},
                }
            )

        # pass the query and retrieved image along with conversation history into the VLM for a response
        def generate_response(message, image):
            chatbot_message = get_chatbot_message_with_image(message, image)
            query = self.qwen2_vl_processor.apply_chat_template(
                [chatbot_message, *self.messages],
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

        relevant_image = get_relevant_image(message)
        output_text = generate_response(message, relevant_image)
        append_to_messages(message, user_type="user")
        append_to_messages(output_text, user_type="assistant")
        return output_text


# ## A hosted Gradio interface

# With the Gradio library, we can create a simple web interface around our class in Python,
# then use Modal to host it for anyone to try out.

# To deploy up your own, run

# ```bash
# modal deploy chat_with_pdf_vision.py
# ```

# and navigate to the URL that appears in your teriminal.
# If you’re editing the code, use `modal serve` instead to see changes live.

web_app = FastAPI()

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
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from gradio_pdf import PDF
    from pdf2image import convert_from_path

    model = Model()

    def respond_to_message(message, _):
        return model.respond_to_message.remote(message)

    def upload_pdf(path):
        images = convert_from_path(path)
        model.index_pdf.remote(images)

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# Chat with PDF")
        with gr.Row():
            with gr.Column(scale=1):
                gr.ChatInterface(
                    fn=respond_to_message,
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn=None,
                )
            with gr.Column(scale=1):
                pdf = PDF(label="Upload a PDF")
                pdf.upload(upload_pdf, pdf)

    return mount_gradio_app(app=web_app, blocks=demo, path="/")
