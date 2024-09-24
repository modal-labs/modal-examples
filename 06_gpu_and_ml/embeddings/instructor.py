# # Create Instructor Embeddings on Modal
#
# This example runs the [Instructor](https://github.com/xlang-ai/instructor-embedding) embedding model and runs a simple sentence similarity computation.

import modal

MODEL_DIR = "/model"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/HKUNLP/instructor-embedding",
        # Package doesn't define it's requirements properly?
        "cd instructor-embedding && pip install -r requirements.txt",
    )
    .pip_install("InstructorEmbedding")
)

app = modal.App("instructor", image=image)

with image.imports():
    from InstructorEmbedding import INSTRUCTOR


@app.cls(gpu="any")
class InstructorModel:
    @modal.build()
    def download_model(self):
        model = INSTRUCTOR("hkunlp/instructor-large")
        model.save(MODEL_DIR)

    @modal.enter()
    def enter(self):
        self.model = INSTRUCTOR(MODEL_DIR, device="cuda")

    @modal.method()
    def compare(self, sentences_a, sentences_b):
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings_a = self.model.encode(sentences_a)
        embeddings_b = self.model.encode(sentences_b)
        similarities = cosine_similarity(embeddings_a, embeddings_b)
        return similarities.tolist()


@app.local_entrypoint()
def run():
    sentences_a = [
        [
            "Represent the Science sentence: ",
            "Parton energy loss in QCD matter",
        ],
        [
            "Represent the Financial statement: ",
            "The Federal Reserve on Wednesday raised its benchmark interest rate.",
        ],
    ]
    sentences_b = [
        [
            "Represent the Science sentence: ",
            "The Chiral Phase Transition in Dissipative Dynamics",
        ],
        [
            "Represent the Financial statement: ",
            "The funds rose less than 0.5 per cent on Friday",
        ],
    ]

    model = InstructorModel()
    similarities = model.compare.remote(sentences_a, sentences_b)
    print(similarities)
