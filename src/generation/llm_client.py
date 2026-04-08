import logging
import os
from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_config = None


def _load_config() -> dict:
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


class MiniMaxManager:
    def __init__(self):
        config = _load_config()
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.base_url = os.getenv("MINIMAX_BASE_URL")

        if not self.api_key:
            raise ValueError("未找到 MINIMAX_API_KEY 环境变量")

        self.client = ChatOpenAI(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info("MiniMaxManager 初始化完成，模型: %s", config["llm"]["model"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        reraise=True,
    )
    def generate_answer(self, query: str, context: str) -> str:
        """
        统一的生成入口
        query: 用户的问题
        context: 从向量数据库检索出来的相关文本块
        """
        config = _load_config()
        system_prompt = config["prompts"]["rag_system"]
        user_content = (
            f"背景资料（Context）：\n{context}\n\n待回答的问题（Query）：\n{query}"
        )

        try:
            response = self.client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
            logger.info("LLM 生成成功")
            return response.content
        except Exception as e:
            logger.error("LLM 请求失败: %s", e)
            return f"LLM 暂时无法处理请求，请稍后再试。(Error: {str(e)[:50]})"

    def generate_answer_stream(self, query: str, context: str):
        """
        流式生成入口 YEild chunks as they arrive.
        """
        config = _load_config()
        system_prompt = config["prompts"]["rag_system"]
        user_content = (
            f"背景资料（Context）：\n{context}\n\n待回答的问题（Query）：\n{query}"
        )

        try:
            for chunk in self.client.stream([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]):
                if chunk.content:
                    yield chunk.content
            logger.info("LLM 流式生成完成")
        except Exception as e:
            logger.error("LLM 流式请求失败: %s", e)
            yield f"\n[LLM 暂时无法处理请求，请稍后再试。(Error: {str(e)[:50]})"
