# Enterprise RAG Core

> 企业级 RAG 系统，支持中文文档的混合检索、智能重排序与流式回答。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 特性

- **混合检索** — Vector Search + BM25 双路召回，中文 jieba 分词
- **RRF 融合** — Reciprocal Rank Fusion 合并多路结果
- **CrossEncoder 重排序** — BAAI/bge-reranker-base 语义精排
- **权限隔离** — public / internal 双角色，internal 可访问全量敏感文档
- **流式输出** — LLM 生成内容实时推送，首字秒出
- **API 服务** — FastAPI + Swagger UI，API Key 鉴权
- **审计日志** — 每次查询异步写入 `logs/audit.log`
- **模型缓存** — Embedding 单例 + CrossEncoder 延迟加载

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 MINIMAX_API_KEY

# 3. 首次入库（仅执行一次）
python main.py --reingest

# 4. CLI 查询
python main.py --role public --query "采购多少钱需要部门负责人审批？"

# 或启动 API 服务
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/query?api_key=public_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "采购审批"}'
```

## 文档说明

`data/public/` 和 `data/internal/` 中的 PDF 为内置测试文档。

**升级为真实文档：** 只需将 PDF 文件放入对应目录，执行 `--reingest` 重新入库即可。无需修改任何代码。

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 检索 + 生成，`api_key` 参数切换角色 |
| `/health` | GET | 健康检查 |

**切换角色：**

```
public_key  → public 角色，仅公开文档
internal_key → internal 角色，可访问薪资/财务等敏感文档
```

浏览器访问 `http://localhost:8000/docs` 使用 Swagger UI 在线调试。

## 配置

所有参数集中在 `config.yaml`：

```yaml
llm:
  model: "MiniMax-M2.7"
  temperature: 0.01

chroma:
  persist_directory: "./data/chroma_db"
  embeddings_model: "shibing624/text2vec-base-chinese"

retrieval:
  default_top_k: 10
  fusion_k: 60
  rerank_top_k: 5
  rerank_model: "BAAI/bge-reranker-base"
```

## 目录结构

```
.
├── config.yaml              # 统一配置
├── main.py                  # CLI 入口
├── test_e2e.py             # 自动化测试（3 cases）
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
├── data/
│   ├── public/             # 公开文档（放入 PDF 后执行 --reingest）
│   ├── internal/           # 内部文档（放入 PDF 后执行 --reingest）
│   └── chroma_db/          # 向量数据库（自动生成）
├── logs/
│   └── audit.log           # 审计日志
└── src/
    ├── api/server.py        # FastAPI 服务
    ├── generation/llm_client.py  # LLM 客户端
    ├── retrieval/
    │   ├── engine.py        # 检索引擎
    │   └── model_loader.py  # 模型单例 + 延迟加载
    └── ingestion/vector_manager.py  # PDF 入库
```

## 测试

```bash
python test_e2e.py
```

## 依赖

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量存储 | ChromaDB 1.1 | 文档向量持久化 |
| Embedding | shibing624/text2vec-base-chinese | 768 维中文语义向量 |
| 关键词检索 | rank-bm25 + jieba | BM25 + 中文分词 |
| 重排序 | BAAI/bge-reranker-base | CrossEncoder 语义精排 |
| LLM | MiniMax-M2.7 | OpenAI 兼容接口 |
| API 框架 | FastAPI + Uvicorn | REST API |
| 重试策略 | tenacity | LLM 请求指数退避 |
| PDF 解析 | PyPDFLoader + RecursiveCharacterTextSplitter | 文档切分入库 |

> **升级说明：** Embedding 和 Rerank 模型更换后需执行 `--reingest` 重新索引全部文档；LLM 只需修改 `.env` 中的 `base_url` 和 `api_key`。

## License

MIT
