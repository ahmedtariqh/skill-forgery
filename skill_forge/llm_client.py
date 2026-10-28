import re
import json
import time
import random
import asyncio
from typing import List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
import litellm
from litellm import acompletion
from rich.console import Console
from .models import VideoSnapshot

litellm.suppress_debug_info = True
litellm.enable_json_schema_validation = True

console = Console()
T = TypeVar("T", bound=BaseModel)


class AsyncRateLimiter:
    """
    Combines an asyncio.Semaphore (in-flight concurrency cap) with an optional
    Requests-Per-Minute (RPM) token-bucket pacer.
    """

    def __init__(self, max_concurrency: int = 8, rpm_limit: Optional[int] = None):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rpm_limit = rpm_limit
        self.min_interval = (60.0 / rpm_limit) if (rpm_limit and rpm_limit > 0) else 0.0
        self._pacer_lock = asyncio.Lock()
        self._last_request_ts = 0.0

    async def __aenter__(self):
        await self.semaphore.acquire()
        if self.min_interval > 0:
            async with self._pacer_lock:
                now = time.monotonic()
                elapsed = now - self._last_request_ts
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
                self._last_request_ts = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class UniversalLLMClient:
    """
    Universal Async LiteLLM Client supporting any provider (OpenAI, Anthropic, Gemini,
    Ollama, vLLM, Groq, DeepSeek) with multimodal image routing and self-healing JSON validation.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        vision_model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 4,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ):
        self.model = model
        self.vision_model = vision_model
        self.api_base = api_base
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter or AsyncRateLimiter(max_concurrency=8)

    @staticmethod
    def _extract_json_block(raw_text: str) -> str:
        raw_text = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        start, end = raw_text.find("{"), raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw_text[start : end + 1]
        return raw_text

    @staticmethod
    def _model_supports_vision(target_model: str) -> bool:
        try:
            if litellm.supports_vision(model=target_model):
                return True
        except Exception:
            pass
        vision_keywords = ("vl", "vision", "llava", "pixtral", "gemini", "gpt-4o", "claude-3", "claude-4", "gemma3")
        return any(k in target_model.lower() for k in vision_keywords)

    async def _adescribe_snapshots_via_vision_model(self, snapshots: List[VideoSnapshot], section_title: str) -> str:
        v_model = self.vision_model or "gemini/gemini-2.5-flash"
        content_blocks = [
            {
                "type": "text",
                "text": (
                    f"Extract all text, tables, charts, flowcharts, formulas, and figure labels "
                    f"from these {len(snapshots)} visual snapshots for '{section_title}'. "
                    "Reconstruct any tables in Markdown and flowcharts in Mermaid syntax."
                ),
            }
        ]
        for snap in snapshots:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{snap.base64_jpeg}"},
            })

        async with self.rate_limiter:
            resp = await acompletion(
                model=v_model,
                messages=[{"role": "user", "content": content_blocks}],
                temperature=0.1,
            )
        return resp.choices[0].message.content

    async def agenerate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        snapshots: Optional[List[VideoSnapshot]] = None,
    ) -> T:
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            "CRITICAL OUTPUT FORMAT REQUIREMENT:\n"
            "You MUST respond ONLY with a valid JSON object that strictly conforms to the following JSON Schema. "
            "Do not include any extra commentary outside the JSON.\n"
            f"--- JSON SCHEMA ---\n{schema_json}"
        )

        user_content = user_prompt
        if snapshots:
            if self._model_supports_vision(self.model) and not self.vision_model:
                user_content = [{"type": "text", "text": user_prompt}]
                for snap in snapshots:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{snap.base64_jpeg}"},
                    })
            else:
                try:
                    visual_ledger = await self._adescribe_snapshots_via_vision_model(snapshots, "Visual Segment")
                    user_content = f"{user_prompt}\n\n--- EXTRACTED VISUAL FIGURES & TABLES ---\n{visual_ledger}"
                except Exception as vis_err:
                    console.print(f"[yellow]Vision bridge skipped ({vis_err}); continuing with text.[/yellow]")

        messages = [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": user_content},
        ]

        supports_native_schema = False
        try:
            supports_native_schema = litellm.supports_response_schema(model=self.model)
        except Exception:
            supports_native_schema = False

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                if supports_native_schema and attempt == 1:
                    kwargs["response_format"] = response_model

                async with self.rate_limiter:
                    response = await acompletion(**kwargs)

                raw_content = response.choices[0].message.content
                cleaned_json = self._extract_json_block(raw_content)
                return response_model.model_validate_json(cleaned_json)

            except ValidationError as ve:
                last_error = ve
                messages.append({"role": "assistant", "content": raw_content if "raw_content" in locals() else "{}"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous JSON output failed schema validation:\n{ve}\n"
                        "Fix the JSON so it strictly matches the required schema and return ONLY valid JSON."
                    ),
                })
            except Exception as e:
                last_error = e
                supports_native_schema = False
                backoff = min(2 ** attempt, 20) + random.uniform(0.2, 2.0)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Failed to obtain valid {response_model.__name__} after {self.max_retries} attempts: {last_error}"
        )
