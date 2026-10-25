"""Shared fixtures: an offline stand-in for Ollama and a temporary index.

The tests never need Ollama or the real models. Embeddings are replaced by a
deterministic hashed bag-of-words vector, which is enough to test routing,
metadata filtering, fact extraction and evidence selection.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class FakeOllama:
    DIM = 512

    def __init__(self):
        self.embed_calls = 0
        self.chat_calls = []

    @classmethod
    def vector(cls, text):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
        v = [0.0] * cls.DIM
        for feature in features:
            h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
            v[h % cls.DIM] += 1.0 if (h >> 20) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, model="", input="", **kwargs):
        self.embed_calls += 1
        items = [input] if isinstance(input, str) else list(input)
        return {"embeddings": [self.vector(t) for t in items]}

    def chat(self, model="", messages=None, **kwargs):
        self.chat_calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "<think></think>LLM "}}, {"message": {"content": "answer"}}])
        return {"message": {"content": "<think>hidden</think>LLM answer"}}


@pytest.fixture(scope="session")
def fake_ollama():
    import ollama

    fake = FakeOllama()
    original = (ollama.embed, ollama.chat)
    ollama.embed, ollama.chat = fake.embed, fake.chat
    yield fake
    ollama.embed, ollama.chat = original


@pytest.fixture(scope="session")
def rag_module(fake_ollama, tmp_path_factory):
    """Build a fresh index in a temp folder and point rag.py at it."""
    import config

    db_dir = tmp_path_factory.mktemp("chroma_db")
    config.CHROMA_DIR = db_dir

    ingest = importlib.import_module("ingest_clean_v2")
    ingest.CHROMA_DIR = db_dir
    ingest.main()

    rag = importlib.import_module("rag")
    rag.CHROMA_DIR = db_dir
    rag.reset_cache()
    return rag


@pytest.fixture()
def calls(fake_ollama):
    fake_ollama.embed_calls = 0
    fake_ollama.chat_calls.clear()
    return fake_ollama
