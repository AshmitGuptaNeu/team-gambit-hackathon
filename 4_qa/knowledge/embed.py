"""Sentence embeddings with a pinned BAAI/bge-small-en-v1.5 (plain transformers, no extra package).

bge models: CLS-token pooling, then L2-normalize, so a dot product is the cosine similarity.
Queries get the bge retrieval prefix; passages (chunks, cards, items) do not.
"""
import numpy as np


class Embedder:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.cfg, self.torch = cfg, torch
        self.dim = cfg["dim"]
        self.version = f"{cfg['model']}@{cfg['revision'][:12]}"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
        self.model = AutoModel.from_pretrained(cfg["model"], revision=cfg["revision"]).to(cfg["device"]).eval()

    def _encode(self, texts):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.cfg["batch_size"]):
                batch = self.tokenizer(texts[i:i + self.cfg["batch_size"]], padding=True, truncation=True,
                                       max_length=self.cfg["max_length"], return_tensors="pt").to(self.cfg["device"])
                cls = self.model(**batch).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def passages(self, texts):
        """(n, dim) float32, unit length."""
        return self._encode(list(texts))

    def query(self, text):
        """(dim,) float32, unit length."""
        return self._encode([self.cfg["query_prefix"] + text])[0]
