# ---
# pytest: false
# ---


class Tokenizer:
    def __init__(self, text):
        self.unique_chars = sorted(set(text))  # sorted to ensure consistent
        self.stoi = {c: i for i, c in enumerate(self.unique_chars)}
        self.itos = {i: c for i, c in enumerate(self.unique_chars)}
        self.vocab_size = len(self.unique_chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return [self.itos[int(t)] for t in tokens]
