"""
FastAPI 服务：企业 RAG 系统 API 层
/ query  — POST — 检索 + 生成
/ health — GET  — 存活检查
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# ── CORS ───────────────────────────────────────────────────────────────────
# 仅允许指定的域名，生产环境建议限制为实际前端地址
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ── Audit Logger ──────────────────────────────────────────────────────────────

_AUDIT_LOG_FILE = Path(__file__).parent.parent.parent / "logs" / "audit.log"
_AUDIT_LOG_FILE.parent.mkdir(exist_ok=True)

_audit_logger = logging.getLogger("audit")
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False
_handler = logging.FileHandler(_AUDIT_LOG_FILE, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_audit_logger.addHandler(_handler)


def _write_audit(
    request_id: str,
    api_key: str,
    role: str,
    query: str,
    sources: list,
    timing: dict,
    answer_preview: str,
) -> None:
    """将审计信息异步写入 audit.log"""
    source_names = ", ".join(s.get("source", "") for s in sources)
    rerank_score = timing.get("rerank", 0)
    total_time = timing.get("total", 0)
    msg = (
        f"request_id={request_id} | "
        f"api_key={api_key} | "
        f"role={role} | "
        f"query={query!r} | "
        f"sources=[{source_names}] | "
        f"rerank={rerank_score:.3f}s | "
        f"total={total_time:.3f}s | "
        f"answer_preview={answer_preview[:80]!r}"
    )
    _audit_logger.info(msg)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    role: Optional[str] = Field("public", description="权限角色：public 或 internal（可由 query param api_key 覆盖）")
    top_k: Optional[int] = Field(None, description="检索条数，默认从 config.yaml 读取")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "X1型号无人机的续航时间是多少？", "role": "public", "top_k": 10}
            ]
        }
    }


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[dict]
    timing: dict


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    embedding_model: str
    rerank_model: str
    timestamp: str


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description="企业知识库检索 API，支持 Hybrid Search + RRF + Rerank",
    version="1.0.0",
)

# CORS：允许跨域访问，生产环境建议限制为实际前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load components on first request
_engine = None
_llm = None


def _get_engine():
    global _engine
    if _engine is None:
        from src.retrieval.engine import RetrievalEngine
        _engine = RetrievalEngine()
    return _engine


def _get_llm():
    global _llm
    if _llm is None:
        from src.generation.llm_client import MiniMaxManager
        _llm = MiniMaxManager()
    return _llm


# ── API Key Auth ─────────────────────────────────────────────────────────────

_API_KEY_MAP = {
    os.getenv("INTERNAL_API_KEY", "internal_key"): "internal",
    os.getenv("PUBLIC_API_KEY", "public_key"): "public",
}


def _resolve_role(api_key: str) -> str:
    role = _API_KEY_MAP.get(api_key)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key",
        )
    return role


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="检索并生成回答",
    responses={
        200: {"description": "成功返回回答"},
        401: {"description": "API Key 无效"},
        500: {"description": "服务端错误"},
    },
)
def query(req: QueryRequest, request: Request, api_key: str = Query(..., description="API Key：public_key=public权限，internal_key=internal权限")):
    request_id = request.headers.get("X-Request-ID", "")

    # Auth
    role = _resolve_role(api_key)

    # Retrieval
    engine = _get_engine()
    t_start = time.perf_counter()
    result = engine.search(query=req.query, role=role, top_k=req.top_k)
    t_retrieval = time.perf_counter() - t_start

    # Generation (streaming → collect full answer for audit)
    llm = _get_llm()
    answer_parts = []
    t_llm_start = time.perf_counter()
    for chunk in llm.generate_answer_stream(req.query, result["content"]):
        answer_parts.append(chunk)
    t_llm = time.perf_counter() - t_llm_start
    answer = "".join(answer_parts)

    timing = {
        **result.get("timing", {}),
        "llm": t_llm,
        "total": t_retrieval + t_llm,
    }

    # Async audit write (non-blocking)
    Thread(
        target=_write_audit,
        args=(request_id, api_key, role, req.query, result["sources"], timing, answer),
        daemon=True,
    ).start()

    return QueryResponse(
        request_id=request_id,
        answer=answer,
        sources=result["sources"],
        timing=timing,
    )


@app.get("/health", response_model=HealthResponse, summary="健康检查")
def health():
    try:
        engine = _get_engine()
        col = engine._client.get_or_create_collection("public_collection")
        count = col.count()
        chromadb_status = "healthy" if count >= 0 else "unhealthy"
    except Exception as e:
        chromadb_status = f"unhealthy: {e}"

    return HealthResponse(
        status="ok",
        chromadb=chromadb_status,
        embedding_model="shibing624/text2vec-base-chinese",
        rerank_model="BAAI/bge-reranker-base",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/", summary="根路径")
def root():
    return {"service": "Enterprise RAG API", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=port, reload=False)
