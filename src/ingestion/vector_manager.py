import logging
import os
from datetime import datetime
from glob import glob
from typing import Sequence

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retrieval.model_loader import ChromaDBEmbedFn, get_base_embeddings, get_chroma_client

logger = logging.getLogger(__name__)

EMBED_DIM = 768  # text2vec-base-chinese 输出维度


class VectorManager:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory
        # ChromaDB Client 单例
        self._client = get_chroma_client(persist_directory)
        # HuggingFaceEmbeddings 单例（首次调用时加载）
        self._base_embeddings = get_base_embeddings()
        # ChromaDB 1.x 兼容的 embedding function
        self._embed_fn = ChromaDBEmbedFn()
        logger.info("VectorManager 初始化完成，持久化目录: %s", self.persist_directory)

    def close(self) -> None:
        logger.info("VectorManager 关闭中...")
        self._client = None
        logger.info("VectorManager 已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def ingest_directory(self, directory_path: str, collection_name: str) -> None:
        logger.info("开始入库: directory=%s, collection=%s", directory_path, collection_name)
        pdf_files = glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            logger.warning("未找到 PDF 文件: %s", directory_path)
            return

        collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )

        for pdf_path in pdf_files:
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load()

            raw_preview = documents[0].page_content[:50] if documents else "(空文档)"
            logger.debug("自检 %s 解析预览: %s", os.path.basename(pdf_path), repr(raw_preview))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=60,
            )
            chunks = text_splitter.split_documents(documents)

            source = os.path.basename(pdf_path)
            now = datetime.now().isoformat()
            ids = []
            texts = []
            metadatas = []
            for i, chunk in enumerate(chunks):
                ids.append(f"{source}_p{chunk.metadata.get('page', 0)}_c{i}")
                texts.append(chunk.page_content)
                metadatas.append({
                    "source": source,
                    "page": chunk.metadata.get("page", 0),
                    "category": collection_name,
                    "timestamp": now,
                })

            collection.add(ids=ids, documents=texts, metadatas=metadatas)
            logger.info("入库完成: %s, chunk 数: %d", source, len(chunks))
