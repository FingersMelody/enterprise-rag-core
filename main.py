"""
生产环境 RAG 入口
用法：
  python main.py --role public --query "X1无人机保修期多久？"
  python main.py --role internal --query "今年的奖金系数是多少？"
  python main.py --reingest          # 清库并重新入库所有文档
"""
import argparse
import logging
import os
import sys
import time as time_module
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.llm_client import MiniMaxManager
from src.ingestion.vector_manager import VectorManager
from src.retrieval.engine import RetrievalEngine

load_dotenv()

_log_file = Path(__file__).parent / "logs" / "app.log"
_log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def reingest() -> None:
    """清空并重建向量数据库"""
    import shutil

    chroma_path = Path("./data/chroma_db")
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        logger.info("已清空向量数据库: %s", chroma_path)

    with VectorManager() as vm:
        vm.ingest_directory("./data/public", "public_collection")
        vm.ingest_directory("./data/internal", "internal_collection")
    logger.info("重新入库完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="企业 RAG 系统 — 生产入口")
    parser.add_argument("--role", choices=["public", "internal"], default="public",
                        help="用户身份，决定可检索的文档范围")
    parser.add_argument("--query", type=str, required=True,
                        help="用户问题")
    parser.add_argument("--reingest", action="store_true",
                        help="清空并重新入库所有 PDF 文档")
    args = parser.parse_args()

    if args.reingest:
        reingest()
        return

    with VectorManager() as vm, RetrievalEngine() as engine:
        llm = MiniMaxManager()

        logger.info("开始检索: role=%s, query='%s'", args.role, args.query)
        t_start = time_module.perf_counter()
        result = engine.search(query=args.query, role=args.role, top_k=10)
        t_retrieval = time_module.perf_counter() - t_start

        print("\n" + "=" * 60)
        print(f"【系统审计】检索范围: [{args.role}_collection]，"
              f"命中 {len(result['sources'])} 条片段")
        print("=" * 60)

        print("\n【参考来源】")
        if result["sources"]:
            for src in result["sources"]:
                print(f"  - {src['source']} (页码: {src['page']})")
        else:
            print("  （无相关文档）")

        print("\n【AI 回答】")
        t_llm_start = time_module.perf_counter()
        for chunk in llm.generate_answer_stream(args.query, result["content"]):
            print(chunk, end="", flush=True)
        t_llm = time_module.perf_counter() - t_llm_start
        print()  # 换行

        total = t_retrieval + t_llm
        timing = result.get("timing", {})

        print(f"\n{'=' * 60}")
        print("  【耗时统计】")
        print(f"  {'阶段':<20} {'耗时':>10}")
        print(f"  {'-' * 34}")
        if timing:
            print(f"  {'向量检索':<20} {timing.get('vector', 0):>10.3f}s")
            print(f"  {'RRF 融合':<20} {timing.get('rrf', 0):>10.3f}s")
            print(f"  {'Rerank':<20} {timing.get('rerank', 0):>10.3f}s")
        print(f"  {'LLM 生成':<20} {t_llm:>10.3f}s")
        print(f"  {'-' * 34}")
        print(f"  {'总耗时':<20} {total:>10.3f}s")
        print(f"{'=' * 60}")

        logger.info("全部完成，总耗时 %.3fs", total)


if __name__ == "__main__":
    main()
