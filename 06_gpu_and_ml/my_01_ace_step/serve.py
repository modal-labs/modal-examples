from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.modal_app import stub, generate
import io

app = FastAPI()

@app.get("/generate")
async def generate_image(prompt: str):
    img_bytes = await stub.generate.remote(prompt)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")