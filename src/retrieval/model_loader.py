"""
模型加载器：模块级单例缓存 + CrossEncoder 延迟加载
确保 text2vec-base-chinese 全局只加载一次，CrossEncoder 仅在首次 rerank 时初始化
"""
import logging
from pathlib import Path
from typing import Optional, Sequence

import chromadb
import yaml
from chromadb.api.types import EmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_config = None
EMBED_DIM = 768  # text2vec-base-chinese 输出维度


def _load_config() -> dict:
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


# ── 模块级单例：HuggingFaceEmbeddings ────────────────────────────────

_base_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_base_embeddings() -> HuggingFaceEmbeddings:
    """返回模块级单例，text2vec 模型全局只存在一份实例"""
    global _base_embeddings
    if _base_embeddings is None:
        config = _load_config()
        cache_path = config.get("models", {}).get("cache_path") or None
        _base_embeddings = HuggingFaceEmbeddings(
            model_name=config["chroma"]["embeddings_model"],
            cache_folder=cache_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("模型单例加载: %s", config["chroma"]["embeddings_model"])
    return _base_embeddings


# ── 模块级单例：ChromaDB PersistentClient ──────────────────────────────

_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client(persist_directory: str = None) -> chromadb.PersistentClient:
    """返回模块级单例，ChromaDB client 全局只存在一份实例"""
    global _client
    if _client is None:
        config = _load_config()
        path = persist_directory or config["chroma"]["persist_directory"]
        from chromadb.config import Settings
        _client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB Client 单例初始化，持久化目录: %s", path)
    return _client


# ── ChromaDB EmbeddingFunction（供 get_or_create_collection 使用）───────

class ChromaDBEmbedFn(EmbeddingFunction):
    """ChromaDB 1.x 兼容：包装单例 HuggingFaceEmbeddings，截断至 EMBED_DIM 维"""

    def __call__(self, input: Sequence[str]):
        vecs = get_base_embeddings().embed_documents(list(input))
        return [v[:EMBED_DIM] for v in vecs]

    @staticmethod
    def name() -> str:
        return "shibing624/text2vec-base-chinese"


# ── 延迟加载：CrossEncoder ─────────────────────────────────────────────

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """返回 CrossEncoder 单例，首次调用时初始化（延迟加载）"""
    global _reranker
    if _reranker is None:
        config = _load_config()
        cache_path = config.get("models", {}).get("cache_path") or None
        _reranker = CrossEncoder(
            config["retrieval"]["rerank_model"],
            cache_folder=cache_path,
        )
        logger.info("CrossEncoder 延迟加载完成: %s", config["retrieval"]["rerank_model"])
    return _reranker


def reset_caches() -> None:
    """仅供测试用：重置所有模块级缓存"""
    global _base_embeddings, _client, _reranker
    _base_embeddings = None
    _client = None
    _reranker = None
    logger.info("模型缓存已重置")
