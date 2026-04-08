import jieba
import logging
import time
from functools import lru_cache
from typing import Optional

from rank_bm25 import BM25Okapi

from src.retrieval.model_loader import (
    ChromaDBEmbedFn,
    _load_config,
    get_base_embeddings,
    get_chroma_client,
    get_reranker,
)

logger = logging.getLogger(__name__)


def rrf_fusion(vector_results: list, bm25_results: list, k: int = 60) -> list[dict]:
    """
    RRF (Reciprocal Rank Fusion)：对 vector 和 BM25 结果按排名融合。
    """
    fused = {}

    for rank, item in enumerate(vector_results, start=1):
        doc_id = item["id"]
        fused.setdefault(doc_id, {
            "content": item["content"],
            "source": item["source"],
            "page": item["page"],
            "timestamp": item.get("timestamp", ""),
        })
        fused[doc_id]["vector_score"] = 1 / (k + rank)

    for rank, item in enumerate(bm25_results, start=1):
        doc_id = item["id"]
        if doc_id in fused:
            fused[doc_id]["bm25_score"] = 1 / (k + rank)
        else:
            fused.setdefault(doc_id, {
                "content": item["content"],
                "source": item["source"],
                "page": item["page"],
                "timestamp": item.get("timestamp", ""),
            })
            fused[doc_id]["bm25_score"] = 1 / (k + rank)

    for doc_id in fused:
        fused[doc_id]["rrf_score"] = (
            fused[doc_id].get("vector_score", 0) + fused[doc_id].get("bm25_score", 0)
        )

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


class RetrievalEngine:
    def __init__(self, persist_directory: str = None):
        config = _load_config()
        self.persist_directory = persist_directory or config["chroma"]["persist_directory"]

        # ChromaDB Client 单例
        self._client = get_chroma_client(self.persist_directory)
        # Embedding 单例
        self._embeddings = get_base_embeddings()
        # ChromaDB 1.x 兼容的 embedding function
        self._embed_fn = ChromaDBEmbedFn()
        # BM25 索引缓存（按 collection name）
        self._bm25_cache: dict[str, BM25Okapi] = {}
        # CrossEncoder 延迟加载
        self._reranker = None

        # ── BM25 预热：初始化时构建所有 collection 的索引 ──────────────
        self._bm25_warm()

        logger.info(
            "RetrievalEngine 初始化完成，持久化目录: %s",
            self.persist_directory,
        )

    def _reranker_instance(self):
        """延迟加载 CrossEncoder，首次 rerank 时初始化"""
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def _bm25_warm(self) -> None:
        """初始化时预构建所有 collection 的 BM25 索引，避免首次查询延迟"""
        collections_to_warm = ["public_collection", "internal_collection"]
        for col_name in collections_to_warm:
            col = self._client.get_or_create_collection(
                name=col_name,
                embedding_function=self._embed_fn,
            )
            if col.count() > 0:
                self._bm25_cache[col_name] = self._build_bm25_index(col)
                logger.info("BM25 索引预热完成: %s", col_name)

    def _build_bm25_index(self, collection) -> BM25Okapi:
        all_data = collection.get(include=["documents", "metadatas"])
        docs = all_data.get("documents", [])
        if not docs:
            raise ValueError("Collection 为空，无法构建 BM25 索引")
        tokenized = [jieba.lcut(doc) for doc in docs]
        return BM25Okapi(tokenized)

    def _bm25_search(
        self, collection, query: str, top_k: int
    ) -> list[dict]:
        if collection.name not in self._bm25_cache:
            self._bm25_cache[collection.name] = self._build_bm25_index(collection)

        bm25: BM25Okapi = self._bm25_cache[collection.name]
        all_data = collection.get(include=["documents", "metadatas"])
        doc_ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        doc_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}

        scores = bm25.get_scores(list(jieba.cut(query)))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                meta = metadatas[idx] if idx < len(metadatas) else {}
                results.append({
                    "id": doc_map[idx],
                    "content": all_data["documents"][idx],
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "timestamp": meta.get("timestamp", ""),
                    "bm25_score": float(scores[idx]),
                })
        return results

    def _vector_search(
        self, collection, query_embedding: list, top_k: int
    ) -> list[dict]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        doc_ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        raw_distances = results.get("distances")
        distances = raw_distances[0] if raw_distances is not None else []

        results_list = []
        for i, doc_id in enumerate(doc_ids):
            results_list.append({
                "id": doc_id,
                "content": documents[i],
                "source": metadatas[i].get("source", "unknown"),
                "page": metadatas[i].get("page", 0),
                "timestamp": metadatas[i].get("timestamp", ""),
                "vector_distance": distances[i] if i < len(distances) else None,
            })
        return results_list

    def _rerank(
        self, query: str, candidates: list[dict], top_n: int
    ) -> list[dict]:
        """
        批量 Rerank：
        1. 一次性将所有 candidate pairs 送入 CrossEncoder.predict()
        2. 当分数接近（相差 < 0.05）时，timestamp 更新的优先
        """
        if not candidates:
            return []

        reranker = self._reranker_instance()
        # 批量预测：所有 pairs 一次传入
        pairs = [[query, item["content"]] for item in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)
        if isinstance(scores, (int, float)):
            scores = [scores]

        for i, (item, score) in enumerate(zip(candidates, scores)):
            item["rerank_score"] = float(score)

        def ts_rank(item: dict) -> float:
            ts = item.get("timestamp", "")
            if not ts:
                return 0.0
            try:
                from datetime import datetime
                return datetime.fromisoformat(ts).timestamp()
            except Exception:
                return 0.0

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: (x[1], ts_rank(x[0])),
            reverse=True,
        )

        ordered = [item for item, _ in scored]
        ordered_scores = [score for _, score in scored]

        for i in range(1, len(ordered)):
            if abs(ordered_scores[i] - ordered_scores[i - 1]) < 0.05:
                if ts_rank(ordered[i]) > ts_rank(ordered[i - 1]):
                    ordered[i - 1], ordered[i] = ordered[i], ordered[i - 1]
                    ordered_scores[i - 1], ordered_scores[i] = ordered_scores[i], ordered_scores[i - 1]

        return ordered[:top_n]

    def close(self) -> None:
        logger.info("RetrievalEngine 关闭中...")
        self._bm25_cache.clear()
        self._client = None
        logger.info("RetrievalEngine 已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @lru_cache(maxsize=100)
    def search(
        self,
        query: str,
        role: str = "public",
        top_k: int = None,
        collection_override: Optional[str] = None,
    ) -> dict:
        """
        检索入口（缓存单位：(query, role)）：
        1. 统一计算 query_embedding（只算一次）
        2. 遍历多个 collection 的 vector + BM25 检索
        3. RRF 融合
        4. 批量 rerank（候选上限 rerank_top_k × 2）
        """
        config = _load_config()
        top_k = top_k or config["retrieval"]["default_top_k"]
        fusion_k = config["retrieval"]["fusion_k"]
        rerank_top_k = config["retrieval"]["rerank_top_k"]
        # rerank 候选严格限制：不超过 rerank_top_k × 2，且不超过 fusion_k
        rerank_candidate_limit = min(rerank_top_k * 2, fusion_k)

        if collection_override:
            collections_to_search = [collection_override]
        elif role == "internal":
            collections_to_search = ["public_collection", "internal_collection"]
        else:
            collections_to_search = ["public_collection"]

        logger.info(
            "检索: query='%s', role=%s, collections=%s, top_k=%d",
            query, role, collections_to_search, top_k
        )

        t0 = time.perf_counter()
        t_emb = time.perf_counter()
        query_embedding = self._embeddings.embed_query(query)
        emb_time = time.perf_counter() - t_emb

        all_vector = []
        all_bm25 = []

        for col_name in collections_to_search:
            col = self._client.get_or_create_collection(
                name=col_name,
                embedding_function=self._embed_fn,
            )
            all_vector.extend(self._vector_search(col, query_embedding, top_k))
            all_bm25.extend(self._bm25_search(col, query, top_k))

        vector_time = time.perf_counter() - t0
        logger.info(
            "Vector 共召回 %d 条 (%.3fs, emb=%.3fs), BM25 共召回 %d 条",
            len(all_vector), vector_time, emb_time, len(all_bm25)
        )

        t_rrf = time.perf_counter()
        fused = rrf_fusion(all_vector, all_bm25, k=fusion_k)
        rrf_time = time.perf_counter() - t_rrf

        # 候选集严格限制在 rerank_top_k × 2（最多 10 条），减少 CrossEncoder 开销
        rerank_candidates = fused[:rerank_candidate_limit]

        t_rerank = time.perf_counter()
        # 批量 rerank（候选上限 rerank_candidate_limit = min(rerank_top_k*2, fusion_k)）
        reranked = self._rerank(query, rerank_candidates, top_n=rerank_top_k)
        rerank_time = time.perf_counter() - t_rerank

        content_parts = [item["content"] for item in reranked]
        sources = [
            {"source": item["source"], "page": item["page"]}
            for item in reranked
        ]

        total_retrieval_time = vector_time + rrf_time + rerank_time
        logger.info(
            "Rerank 后命中 %d 条片段 (RRF=%.3fs, rerank=%.3fs, 总检索=%.3fs)",
            len(sources), rrf_time, rerank_time, total_retrieval_time
        )
        return {
            "content": "\n\n".join(content_parts),
            "sources": sources,
            "timing": {
                "vector": vector_time,
                "rrf": rrf_time,
                "rerank": rerank_time,
                "total": total_retrieval_time,
            },
        }
