# # GLiNER2 on Modal
#
# [GLiNER2](https://github.com/fastino-ai/GLiNER2) structured extraction (`extract_json`) and
# classification (`classify_text`) on one string. From the repo root:
# `modal run misc/gliner2_modal_demo.py`

import json
from pathlib import Path

import modal

APP_NAME = "example-gliner2-modal"
MODEL_ID = "fastino/gliner2-base-v1"
DEFAULT_TEXT = (
    "Tim Cook unveiled iPhone 15 Pro for $999 in Cupertino; "
    "reviewers praised the titanium design but criticized battery life."
)
SCHEMA = {
    "announcement": [
        "company::str",
        "person::str",
        "product::str",
        "price::str",
        "location::str",
    ]
}
CLS = {"sentiment": ["positive", "negative", "neutral", "mixed"]}

HERE = Path(__file__).parent.resolve()

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(
    str(HERE / "gliner2_modal_demo_requirements.txt")
)


@app.cls(image=image, cpu=2.0, memory=2048)
class GLiNERService:
    @modal.enter()
    def load(self):
        from gliner2 import GLiNER2

        self.model = GLiNER2.from_pretrained(MODEL_ID)

    @modal.method()
    def analyze(self, text: str) -> dict:
        m = self.model
        return {
            "structured": m.extract_json(text, SCHEMA),
            "classification": m.classify_text(text, CLS, include_confidence=True),
        }


@app.local_entrypoint()
def main(text: str = DEFAULT_TEXT):
    print("Input:", text)
    print(json.dumps(GLiNERService().analyze.remote(text), indent=2))
