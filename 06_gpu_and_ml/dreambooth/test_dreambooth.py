import modal
from dreambooth_app import AppConfig

app = modal.App("test-local-entrypoint")


@app.local_entrypoint()
def run_dreambooth_inference():
    model = modal.Cls.lookup("example-dreambooth-flux", "Model")
    text = "a heroicon of a panther"
    config = AppConfig()
    results = model().inference.remote(text, config)
    print(results)
