import modal
import os

from fastapi import FastAPI

MINUTES = 60  # seconds

def download_model(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        model_name,
        local_dir= model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    move_cache()

model_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install([
        "git+https://github.com/illuin-tech/colpali", 
        "hf_transfer", 
        "qwen-vl-utils",
        "torchvision"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
        timeout=MINUTES * 20,
        kwargs={"model_dir": "/model-qwen2-VL-2B-Instruct", "model_name": "Qwen/Qwen2-VL-2B-Instruct"},
    )
)


app = modal.App("chat-with-pdf")

with model_image.imports():
    import torch
    from PIL import Image
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info


@app.cls(
    image=model_image,
    gpu=modal.gpu.A100(size="80GB"),
    container_idle_timeout=10 * MINUTES,
    allow_concurrent_inputs=4,
)
class Model:
    @modal.build()
    @modal.enter()
    def load_model(self):
        self.colqwen2_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
        )
        self.colqwen2_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
        self.qwen2_vl_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        self.qwen2_vl_model.to("cuda:0")
        self.qwen2_vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        
        self.pdf_embeddings = None
        self.images = None
        self.messages = []


    @modal.method()
    def index_pdf(self, images):
        self.images = images
        batch_images = self.colqwen2_processor.process_images(images).to(self.colqwen2_model.device)
        self.pdf_embeddings = self.colqwen2_model(**batch_images)

    @modal.method()
    def respond_to_message(self, message):
        if self.images is None:
            return "Please upload a PDF first"
        elif self.pdf_embeddings is None:
            return "Indexing PDF..."


        def get_relevant_image(message):
            batch_queries = self.colqwen2_processor.process_queries([message]).to(self.colqwen2_model.device)
            query_embeddings = self.colqwen2_model(**batch_queries)
            scores = self.colqwen2_processor.score_multi_vector(query_embeddings, self.pdf_embeddings)[0]
            max_index = max(range(len(scores)), key=lambda index: scores[index])
            return self.images[max_index]
        
        def get_chatbot_message_with_image(message, image):
            return {
                "role": "user", 
                "content": [{
                    "type": "image",
                    "image": image
                }, {
                    "type": "text",
                    "text": message
                }]
            }
    
        def append_to_messages(message, user_type="user"):
            self.messages.append(
                {
                    "role": user_type, 
                    "content": {
                        "type": "text",
                        "text": message
                    }
                }
            )

        def generate_response(message, image):
            chatbot_message = get_chatbot_message_with_image(message, image)
            query = self.qwen2_vl_processor.apply_chat_template(
                [chatbot_message, *self.messages], tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info([chatbot_message])
            inputs = self.qwen2_vl_processor(
                text=[query],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda:0")

            generated_ids = self.qwen2_vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text
        
        relevant_image = get_relevant_image(message)
        output_text = generate_response(message, relevant_image)
        append_to_messages(message, user_type="user")
        append_to_messages(output_text, user_type="assistant")
        return output_text

web_app = FastAPI()

web_image = modal.Image.debian_slim().apt_install("poppler-utils").pip_install(
    "gradio", "pillow", "gradio-pdf", "pdf2image"
)

@app.function(image=web_image, keep_warm=1, concurrency_limit=1, allow_concurrent_inputs=1000)
@modal.asgi_app()
def ui():
    from gradio.routes import mount_gradio_app
    import gradio as gr
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