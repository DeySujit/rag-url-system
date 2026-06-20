"""Thin async wrapper around the OpenAI chat completion API."""
from __future__ import annotations

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import settings


class LLM:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)

    @retry(reraise=True, stop=stop_after_attempt(4),
           wait=wait_random_exponential(multiplier=1, max=20))
    async def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        logger.debug("LLM produced {} chars", len(content))
        return content.strip()
