# Quick Setup

After cloning, add your PDF documents:

```
data/public/     → 公开文档（public + internal 角色均可访问）
data/internal/    → 内部文档（仅 internal 角色可访问）
```

Then run ingestion:

```bash
python main.py --reingest
```

This builds the ChromaDB vector index and is required before first use.
