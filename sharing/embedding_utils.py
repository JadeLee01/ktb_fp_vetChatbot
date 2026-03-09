import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(override=False)

DEFAULT_EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"
DEFAULT_CHROMA_DB_DIR = "./models/chroma_db_e5_base"
E5_PREFIX_MODELS = ("intfloat/multilingual-e5",)


def get_embedding_model_id() -> str:
    return os.getenv("EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID)


def get_chroma_db_dir() -> str:
    return os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DB_DIR)


def should_use_e5_prefix(model_id: str) -> bool:
    configured = os.getenv("EMBEDDING_USE_E5_PREFIX")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes", "on"}
    return model_id.startswith(E5_PREFIX_MODELS)


def resolve_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class PrefixedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, *args, query_prefix: str = "", passage_prefix: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _prefix_texts(self, texts: Iterable[str], prefix: str) -> list[str]:
        return [f"{prefix}{text}" if text else prefix.rstrip() for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(self._prefix_texts(texts, self.passage_prefix))

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self._prefix_texts([text], self.query_prefix)[0])


def build_embeddings(device: str | None = None) -> PrefixedHuggingFaceEmbeddings:
    model_id = get_embedding_model_id()
    use_e5_prefix = should_use_e5_prefix(model_id)
    query_prefix = "query: " if use_e5_prefix else ""
    passage_prefix = "passage: " if use_e5_prefix else ""
    target_device = device or resolve_device()

    return PrefixedHuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": target_device},
        encode_kwargs={"normalize_embeddings": use_e5_prefix},
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
