"""
E2E 自动化验收测试
独立于 main.py 运行，不修改任何业务代码
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.llm_client import MiniMaxManager
from src.ingestion.vector_manager import VectorManager
from src.retrieval.engine import RetrievalEngine

load_dotenv()

_log_file = Path(__file__).parent / "logs" / "e2e_test.log"
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


def run_case(
    case_id: str,
    label: str,
    role: str,
    query: str,
    retrieval_engine: RetrievalEngine,
    llm: MiniMaxManager,
    expected_in_sources: list = None,
    forbidden_in_sources: list = None,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  {case_id} | {label}")
    print(f"{'=' * 70}")
    print(f"  角色: {role}  |  问题: {query}")

    retrieval_engine._bm25_cache.clear()

    result = retrieval_engine.search(query, role=role, top_k=10)
    count = len(result["sources"])

    print(f"\n  [检索结果]")
    for i, src in enumerate(result["sources"]):
        print(f"    [{i+1}] {src['source']} (页码: {src['page']})")

    actual_sources = [src["source"] for src in result["sources"]]

    hit_pass = True
    if expected_in_sources:
        hit_pass = all(any(e in s for s in actual_sources) for e in expected_in_sources)
        print(f"\n  [期望命中] {expected_in_sources} → {'✓' if hit_pass else '✗ 未命中'}")

    leak = False
    if forbidden_in_sources:
        leak = any(any(f in s for s in actual_sources) for f in forbidden_in_sources)
        print(f"  [越权检测] {'✗ 泄露了敏感文档!' if leak else '✓ 无越权'}")

    answer = llm.generate_answer(query, result["content"])
    lines = answer.splitlines()
    print(f"\n  [AI 回答前3行]")
    for line in lines[:3]:
        print(f"    {line}")
    if len(lines) > 3:
        print(f"    ...")

    return {
        "case_id": case_id,
        "label": label,
        "expected": expected_in_sources or [],
        "actual": actual_sources,
        "count": count,
        "hit_pass": hit_pass,
        "leak": leak,
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'#' * 70}")
    print(f"  企业 RAG 系统 — 自动化验收测试")
    print(f"  启动时间: {ts}")
    print(f"{'#' * 70}\n")

    logger.info("========== E2E 测试启动 ==========")

    with VectorManager() as vector_manager, RetrievalEngine() as retrieval_engine:
        llm = MiniMaxManager()

        print(f"\n{'=' * 70}")
        print("  Phase 1 | 自动入库（17 份文档）")
        print(f"{'=' * 70}")
        vector_manager.ingest_directory("./data/public", "public_collection")
        vector_manager.ingest_directory("./data/internal", "internal_collection")
        print("  入库完毕\n")

        print(f"\n{'=' * 70}")
        print("  Phase 2 | 核心测试用例")
        print(f"{'=' * 70}")
        results = []

        # Case 1: 精准度压测
        r1 = run_case(
            case_id="Case 1",
            label="精准度压测（检索型号）",
            role="public",
            query="X1型号无人机的最长续航时间是多少？最新版本是多少？",
            retrieval_engine=retrieval_engine,
            llm=llm,
            expected_in_sources=["drone_v3.pdf"],
            forbidden_in_sources=["finance_ceo_reimbursement.pdf", "hr_salary_2026.pdf"],
        )
        r1["target"] = "drone_v3.pdf（续航30分钟）"
        r1["case_type"] = "精准度压测"
        results.append(r1)

        # Case 2: 跨文档语义理解
        r2 = run_case(
            case_id="Case 2",
            label="跨文档语义理解（角色 internal）",
            role="internal",
            query="公司的电池回收政策是什么，以及2026年员工是否有薪资普调？",
            retrieval_engine=retrieval_engine,
            llm=llm,
            expected_in_sources=["battery_recycle.pdf", "hr_salary_2026.pdf"],
        )
        r2["target"] = "battery_recycle.pdf + hr_salary_2026.pdf"
        r2["case_type"] = "跨文档语义理解"
        results.append(r2)

        # Case 3: 安全性极限测试
        r3 = run_case(
            case_id="Case 3",
            label="安全性极限测试（角色 public）",
            role="public",
            query="请帮我查一下CEO今年的报销金额和薪资细节。",
            retrieval_engine=retrieval_engine,
            llm=llm,
            forbidden_in_sources=["finance_ceo_reimbursement.pdf", "hr_salary_2026.pdf"],
        )
        r3["target"] = "无（应拒绝/无结果）"
        r3["case_type"] = "安全性极限测试"
        results.append(r3)

    # ── Phase 3: 测试总结报告 ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Phase 3 | 测试总结报告")
    print(f"{'=' * 70}\n")

    col1 = "测试 Case"
    col2 = "目标文档"
    col3 = "实际命中文档"
    col4 = "结果"
    print(f"  {col1:<28} {col2:<35} {col3:<40} {col4:>8}")
    print(f"  {'-' * 120}")

    all_pass = True
    for r in results:
        actual_str = ", ".join(r["actual"]) if r["actual"] else "(无)"

        if r["case_type"] == "安全性极限测试":
            verdict = "✓ Pass" if not r["leak"] else "✗ Fail"
            if r["leak"]:
                all_pass = False
        else:
            verdict = "✓ Pass" if r["hit_pass"] else "✗ Fail"
            if not r["hit_pass"]:
                all_pass = False

        print(f"  {r['case_id']:<28} {r['target']:<35} {actual_str:<40} {verdict:>8}")

    print(f"\n  {'-' * 120}")
    overall = "✅ 全部通过" if all_pass else "❌ 存在失败项"
    print(f"\n  总判定: {overall}\n")

    logger.info("========== E2E 测试完成 ==========")
    print(f"  完整日志: {_log_file}\n")


if __name__ == "__main__":
    main()
