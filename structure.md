# gemini response

This modular package (`skill_forge/`) upgrades the monolithic script into a clean-architecture Python project with two major architectural capabilities:

1. **Multimodal Ebook Visual Extraction (`ingestors/ebook.py`)**:
   * **PDFs (`PyMuPDF`)**: Extracts both **embedded raster images** (`page.get_images()`) and **vector-drawn figures/charts/tables** (by clustering vector drawing bounding boxes via `page.get_drawings()` and rendering high-DPI crops with `page.get_pixmap()`). Each visual is scored by Canny edge/contrast density and stamped with `[Page N]`.
   * **EPUBs (`ebooklib` + `BeautifulSoup`)**: Parses `<img>` and `<image>` tags inside each chapter XHTML document, resolves relative paths against the archive manifest, filters out tiny decorative icons, and stamps `[Ch.X Fig.Y]` badges.
2. **Pass 3 — Semantic Quality Auditing, Gap Recovery & Micro-Consolidation (`engine/auditor.py`)**:
   * **Deterministic Ground-Truth Ledger**: Extracts high-signal tokens (equations, numeric dosages/thresholds, percentages, ratios, named protocols, and visual titles) from the source `KnowledgeExtractionChunk` list and cross-checks them against the synthesized skill.
   * **Micro-Consolidation**: Merges fragmented or near-duplicate concepts, procedures, and visual artifacts inside the skill into unified, non-redundant entries with conditional branches.
   * **Zero-Loss Patching**: Recovers any omitted formulas, edge cases, contraindications, or tables and injects them directly into the final `SynthesizedAgentSkill`, producing an `audit_report.json` alongside `SKILL.md`.

---

## 1. Project Directory Structure

```text
skill_forge/
├── requirements.txt
├── main.py
└── skill_forge/
    ├── __init__.py
    ├── models.py
    ├── cache.py
    ├── llm_client.py
    ├── exporter.py
    ├── ingestors/
    │   ├── __init__.py
    │   ├── visual_utils.py
    │   ├── ebook.py
    │   └── youtube.py
    └── engine/
        ├── __init__.py
        ├── pipeline.py
        └── auditor.py
```

---

## 2. `requirements.txt`

```text
litellm>=1.60.0
pydantic>=2.9.0
pymupdf>=1.24.10
ebooklib>=0.18
beautifulsoup4>=4.12.3
youtube-transcript-api>=1.0.0
yt-dlp>=2025.1.0
faster-whisper>=1.1.0
ctranslate2>=4.5.0
opencv-python-headless>=4.10.0.84
pillow>=10.4.0
numpy>=1.26.4
rich>=13.9.0
```

---

## 3. Complete Modular Python Files

### `skill_forge/__init__.py`

```python
"""SkillForge: Multimodal Ebook & YouTube Playlist to Agent Skill Compiler."""

__version__ = "4.0.0"
```

---

### `skill_forge/models.py`

```python
from typing import List
from pydantic import BaseModel, Field


class TheoreticalConcept(BaseModel):
    name: str = Field(description="Name of the theory, principle, mental model, or definition.")
    explanation: str = Field(description="Precise, comprehensive explanation of the concept.")
    governing_rules_or_formulas: List[str] = Field(
        default_factory=list,
        description="Underlying equations, axioms, mechanisms, or cause-effect rules."
    )
    source_context: str = Field(description="Video or chapter title where this was captured.")


class AppliedProcedure(BaseModel):
    title: str = Field(description="Name of the actionable workflow, protocol, or SOP.")
    trigger_condition: str = Field(description="When the agent should execute this procedure.")
    prerequisites: List[str] = Field(default_factory=list, description="Required inputs or context before starting.")
    steps: List[str] = Field(description="Exact, sequential step-by-step instructions.")
    decision_branches: List[str] = Field(
        default_factory=list,
        description="Conditional logic (If X then Y; else Z) for edge cases."
    )
    anti_patterns: List[str] = Field(
        default_factory=list,
        description="Common mistakes, contraindications, or pitfalls to avoid."
    )
    verification_checklist: List[str] = Field(
        default_factory=list,
        description="Criteria to verify the procedure was executed accurately."
    )


class VisualArtifact(BaseModel):
    title: str = Field(description="Title of the visual table, chart, diagram, slide, or figure.")
    timestamp_or_page: str = Field(description="Timestamp (e.g., '04:20') or page badge (e.g., 'Page 42').")
    artifact_type: str = Field(
        description="Type of visual: 'table', 'flowchart', 'formula_sheet', 'system_diagram', or 'annotated_figure'."
    )
    structured_representation: str = Field(
        description=(
            "Complete reconstruction of the visual in agent-readable Markdown: "
            "use a Markdown Table for spreadsheets/matrices, a Mermaid.js code block (```mermaid) for flowcharts/pathways, "
            "or LaTeX/structured bullet points for equations and schematic diagrams."
        )
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Critical numbers, thresholds, relationships, or steps visible in this snapshot."
    )


class VideoSnapshot(BaseModel):
    timestamp_sec: float = Field(description="Position in seconds or page index.")
    timestamp_str: str = Field(description="Human-readable badge such as '04:20' or 'Page 14'.")
    base64_jpeg: str = Field(description="Base64-encoded JPEG image string.")
    info_score: float = Field(description="Visual edge/contrast density score.")


class ContentSegment(BaseModel):
    segment_id: str
    title: str
    text: str
    snapshots: List[VideoSnapshot] = Field(default_factory=list)


class KnowledgeExtractionChunk(BaseModel):
    segment_id: str = Field(description="Unique identifier of the video or chapter.")
    section_title: str = Field(description="Title of the video or chapter.")
    detected_domain: str = Field(
        description="Normalized macro-field (e.g., Medicine, Finance, Law, Nutrition, Business, Philosophy)."
    )
    detected_subtopic: str = Field(
        description="Specific sub-discipline (e.g., Equity Valuation, Clinical Endocrinology, Contract Negotiation)."
    )
    executive_summary: str = Field(description="2-sentence summary of the core knowledge inside this segment.")
    theoretical_concepts: List[TheoreticalConcept]
    applied_procedures: List[AppliedProcedure]
    visual_artifacts: List[VisualArtifact] = Field(
        default_factory=list,
        description="Tables, charts, diagrams, formulas, and figures captured from visual snapshots."
    )
    domain_heuristics: List[str] = Field(
        description="High-leverage rules of thumb, numeric thresholds, or diagnostic checks."
    )


class SkillClusterAssignment(BaseModel):
    cluster_slug: str = Field(description="kebab-case slug for this skill (e.g., corporate-valuation-modeling).")
    cluster_title: str = Field(description="Human-readable title for the skill.")
    macro_domain: str = Field(description="Standardized field name (e.g., Finance, Medicine, Nutrition).")
    segment_ids: List[str] = Field(description="List of segment_ids that belong to this skill cluster.")


class PlaylistTaxonomyPlan(BaseModel):
    clusters: List[SkillClusterAssignment]


class SynthesizedAgentSkill(BaseModel):
    skill_slug: str = Field(description="kebab-case identifier for the skill.")
    skill_title: str
    domain: str = Field(description="Standardized macro-domain (e.g., Medicine, Finance, Law, Nutrition).")
    subtopics_covered: List[str] = Field(default_factory=list)
    trigger_description: str = Field(description="Clear description of WHEN an AI agent must activate this skill.")
    core_persona_and_stance: str = Field(
        description="How the agent should reason, prioritize accuracy, and structure outputs in this domain."
    )
    primary_workflows: List[AppliedProcedure]
    theoretical_foundations: List[TheoreticalConcept]
    visual_frameworks_and_tables: List[VisualArtifact] = Field(default_factory=list)
    master_heuristics: List[str]
    global_anti_patterns: List[str]


class AuditPatchReport(BaseModel):
    coverage_score_before: float = Field(description="Estimated completeness percentage (0-100) before auditing.")
    coverage_score_after: float = Field(description="Estimated completeness percentage (0-100) after patching.")
    missing_gaps_recovered: List[str] = Field(
        default_factory=list,
        description="Specific formulas, thresholds, steps, or contraindications recovered from source chunks."
    )
    consolidated_merges_made: List[str] = Field(
        default_factory=list,
        description="Closely related concepts, workflows, or visual tables that were unified."
    )
    refined_skill: SynthesizedAgentSkill = Field(
        description="The final audited, gap-patched, and micro-consolidated Agent Skill."
    )
```

---

### `skill_forge/cache.py`

```python
import json
import sqlite3
import hashlib
import threading
from typing import List, Optional
from .models import VideoSnapshot, KnowledgeExtractionChunk


class PipelineCache:
    """
    Thread-safe and coroutine-safe SQLite checkpoint cache using WAL mode.
    Stores transcripts, video/ebook visual snapshots, and LLM knowledge extractions.
    """

    def __init__(self, db_path: str = "./skill_forge_cache.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self):
        with self._lock, self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    transcript_text TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS visual_snapshots (
                    source_id TEXT PRIMARY KEY,
                    snapshots_json TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    cache_key TEXT PRIMARY KEY,
                    segment_id TEXT,
                    model_name TEXT,
                    detected_domain TEXT,
                    chunk_json TEXT
                )
            """)

    def get_transcript(self, video_id: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.execute("SELECT transcript_text FROM transcripts WHERE video_id = ?", (video_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def save_transcript(self, video_id: str, title: str, text: str):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO transcripts (video_id, title, transcript_text) VALUES (?, ?, ?)",
                (video_id, title, text),
            )

    def get_snapshots(self, source_id: str) -> Optional[List[VideoSnapshot]]:
        with self._lock:
            cur = self.conn.execute("SELECT snapshots_json FROM visual_snapshots WHERE source_id = ?", (source_id,))
            row = cur.fetchone()
            if row:
                return [VideoSnapshot.model_validate(item) for item in json.loads(row[0])]
            return None

    def save_snapshots(self, source_id: str, snapshots: List[VideoSnapshot]):
        payload = json.dumps([s.model_dump() for s in snapshots])
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO visual_snapshots (source_id, snapshots_json) VALUES (?, ?)",
                (source_id, payload),
            )

    @staticmethod
    def _make_key(segment_id: str, text: str, model_tag: str, domain_hint: str) -> str:
        return hashlib.sha256(f"{segment_id}:{model_tag}:{domain_hint}:{text}".encode("utf-8")).hexdigest()

    def get_extraction(
        self, segment_id: str, text: str, model_tag: str, domain_hint: str
    ) -> Optional[KnowledgeExtractionChunk]:
        key = self._make_key(segment_id, text, model_tag, domain_hint)
        with self._lock:
            cur = self.conn.execute("SELECT chunk_json FROM extractions WHERE cache_key = ?", (key,))
            row = cur.fetchone()
            return KnowledgeExtractionChunk.model_validate_json(row[0]) if row else None

    def save_extraction(
        self, segment_id: str, text: str, model_tag: str, domain_hint: str, chunk: KnowledgeExtractionChunk
    ):
        key = self._make_key(segment_id, text, model_tag, domain_hint)
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO extractions 
                   (cache_key, segment_id, model_name, detected_domain, chunk_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, segment_id, model_tag, chunk.detected_domain, chunk.model_dump_json()),
            )
```

---

### `skill_forge/llm_client.py`

```python
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
```

---

### `skill_forge/ingestors/__init__.py`

```python
from .ebook import EbookIngestor
from .youtube import YouTubeIngestor, WhisperAudioTranscriber
from .visual_utils import VideoSnapshotExtractor

__all__ = ["EbookIngestor", "YouTubeIngestor", "WhisperAudioTranscriber", "VideoSnapshotExtractor"]
```

---

### `skill_forge/ingestors/visual_utils.py`

```python
import io
import base64
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yt_dlp
from ..models import VideoSnapshot


def compute_visual_info_score(gray_img: np.ndarray) -> float:
    """
    Measures Canny edge density + contrast variance.
    Charts, tables, diagrams, and equations score higher than blank backgrounds or solid logos.
    """
    if gray_img.size == 0:
        return 0.0
    edges = cv2.Canny(gray_img, 80, 180)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size) * 100.0
    contrast_std = float(np.std(gray_img)) / 255.0 * 10.0
    return round(edge_ratio + contrast_std, 3)


def encode_pil_image_with_badge(
    pil_img: Image.Image,
    badge_text: str,
    max_dimension: int = 1024,
    jpeg_quality: int = 80,
) -> Tuple[str, float]:
    """
    Converts a PIL image to RGB, computes its visual information density score,
    resizes to `max_dimension`, stamps a top-left `[badge_text]` label, and returns (base64_jpeg, info_score).
    """
    rgb_img = pil_img.convert("RGB")
    np_rgb = np.array(rgb_img)
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    info_score = compute_visual_info_score(gray)

    w, h = rgb_img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        rgb_img = rgb_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(rgb_img)
    label = f" [{badge_text}] "
    box_width = max(110, len(label) * 8)
    draw.rectangle([(8, 8), (box_width, 34)], fill=(0, 0, 0))
    draw.text((14, 13), label, fill=(255, 255, 0))

    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64_str, info_score


def select_top_snapshots(candidates: List[VideoSnapshot], max_snapshots: int) -> List[VideoSnapshot]:
    """Selects top K snapshots by info_score and restores chronological/page order."""
    if len(candidates) <= max_snapshots:
        return sorted(candidates, key=lambda x: x.timestamp_sec)
    top_k = sorted(candidates, key=lambda x: x.info_score, reverse=True)[:max_snapshots]
    return sorted(top_k, key=lambda x: x.timestamp_sec)


class VideoSnapshotExtractor:
    """Extracts deduplicated, high-information keyframes from YouTube videos."""

    def __init__(
        self,
        max_snapshots: int = 12,
        sample_interval_sec: float = 8.0,
        min_scene_diff_threshold: float = 12.0,
        max_image_dimension: int = 1024,
        jpeg_quality: int = 80,
    ):
        self.max_snapshots = max_snapshots
        self.sample_interval_sec = sample_interval_sec
        self.min_scene_diff_threshold = min_scene_diff_threshold
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        mins, secs = int(seconds // 60), int(seconds % 60)
        hours, mins = int(mins // 60), mins % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _download_compact_video(video_id: str, temp_dir: str) -> Path:
        video_url = f"[https://www.youtube.com/watch?v=](https://www.youtube.com/watch?v=){video_id}"
        out_template = str(Path(temp_dir) / f"{video_id}_vid.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/bestvideo[height<=480]/best[height<=480]",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        files = list(Path(temp_dir).glob(f"{video_id}_vid.*"))
        if not files:
            raise FileNotFoundError(f"Video stream download failed for {video_id}")
        return files[0]

    def extract_snapshots(self, video_id: str) -> List[VideoSnapshot]:
        with tempfile.TemporaryDirectory(prefix="skillforge_vid_") as tmp_dir:
            video_path = self._download_compact_video(video_id, tmp_dir)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_sec = total_frames / fps if total_frames > 0 else 0
            interval = (
                max(self.sample_interval_sec, duration_sec / 120.0)
                if duration_sec > 0
                else self.sample_interval_sec
            )

            candidates: List[VideoSnapshot] = []
            prev_small_gray: Optional[np.ndarray] = None
            current_sec = 2.0

            while True:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000.0)
                success, frame = cap.read()
                if not success or frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (64, 36))
                mean_lum = float(np.mean(small_gray))

                if 15.0 < mean_lum < 245.0:
                    is_distinct = True
                    if prev_small_gray is not None:
                        diff = float(np.mean(cv2.absdiff(small_gray, prev_small_gray)))
                        is_distinct = diff >= self.min_scene_diff_threshold

                    if is_distinct:
                        ts_str = self._format_timestamp(current_sec)
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        b64_jpg, info_score = encode_pil_image_with_badge(
                            pil_img,
                            badge_text=ts_str,
                            max_dimension=self.max_image_dimension,
                            jpeg_quality=self.jpeg_quality,
                        )
                        candidates.append(
                            VideoSnapshot(
                                timestamp_sec=round(current_sec, 1),
                                timestamp_str=ts_str,
                                base64_jpeg=b64_jpg,
                                info_score=info_score,
                            )
                        )
                        prev_small_gray = small_gray

                current_sec += interval
                if duration_sec > 0 and current_sec >= duration_sec:
                    break

            cap.release()
            return select_top_snapshots(candidates, self.max_snapshots)
```

---

### `skill_forge/ingestors/ebook.py`

```python
import io
import posixpath
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from PIL import Image

from ..models import ContentSegment, VideoSnapshot
from ..cache import PipelineCache
from .visual_utils import encode_pil_image_with_badge, select_top_snapshots


class EbookIngestor:
    """
    Extracts structured chapters/sections AND embedded figures, charts, diagrams,
    and vector-drawn tables from PDF and EPUB ebooks.
    """

    MIN_IMAGE_WIDTH = 160
    MIN_IMAGE_HEIGHT = 160
    MIN_INFO_SCORE = 1.8

    @staticmethod
    def load(
        file_path: str,
        cache: Optional[PipelineCache] = None,
        max_chars_per_chunk: int = 35000,
        enable_visuals: bool = True,
        max_snapshots_per_chapter: int = 10,
    ) -> List[ContentSegment]:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            raw_segments = EbookIngestor._read_pdf(
                path,
                cache=cache,
                enable_visuals=enable_visuals,
                max_snapshots_per_chapter=max_snapshots_per_chapter,
            )
        elif ext == ".epub":
            raw_segments = EbookIngestor._read_epub(
                path,
                cache=cache,
                enable_visuals=enable_visuals,
                max_snapshots_per_chapter=max_snapshots_per_chapter,
            )
        else:
            raise ValueError(f"Unsupported ebook format: {ext}. Use .pdf or .epub.")

        return EbookIngestor.normalize_segments(raw_segments, max_chars_per_chunk)

    @staticmethod
    def _extract_pdf_page_visuals(doc: fitz.Document, page_index: int) -> List[VideoSnapshot]:
        """
        Extracts both:
          1. Embedded raster images (`page.get_images()`)
          2. Vector-drawn charts, flowcharts, and bordered tables (`page.get_drawings()`)
        from a single PDF page.
        """
        page = doc[page_index]
        page_num = page_index + 1
        snapshots: List[VideoSnapshot] = []
        seen_xrefs = set()

        # 1. Extract embedded raster figures/charts
        for img_idx, img_info in enumerate(page.get_images(full=True), 1):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base_img = doc.extract_image(xref)
                if not base_img or "image" not in base_img:
                    continue
                pil_img = Image.open(io.BytesIO(base_img["image"]))
                w, h = pil_img.size
                if w < EbookIngestor.MIN_IMAGE_WIDTH or h < EbookIngestor.MIN_IMAGE_HEIGHT:
                    continue
                aspect = max(w, h) / float(max(1, min(w, h)))
                if aspect > 7.0:  # Skip thin decorative page rules/banners
                    continue

                badge = f"Page {page_num} Fig.{img_idx}"
                b64_jpg, score = encode_pil_image_with_badge(pil_img, badge_text=badge)
                if score >= EbookIngestor.MIN_INFO_SCORE:
                    snapshots.append(
                        VideoSnapshot(
                            timestamp_sec=float(page_num) + (img_idx * 0.01),
                            timestamp_str=badge,
                            base64_jpeg=b64_jpg,
                            info_score=score,
                        )
                    )
            except Exception:
                continue

        # 2. Detect vector-drawn charts, anatomical/financial diagrams, or complex tables
        # If no raster figure was found on this page, check if the page has rich vector drawings
        if not snapshots:
            try:
                drawings = page.get_drawings()
                rects = [
                    fitz.Rect(d["rect"])
                    for d in drawings
                    if d.get("rect") and d["rect"].width > 4 and d["rect"].height > 4
                ]
                if len(rects) >= 18:
                    union_rect = rects[0]
                    for r in rects[1:]:
                        union_rect |= r
                    # Expand slightly to include axis labels and legends
                    union_rect = union_rect + (-15, -15, 15, 15)
                    union_rect &= page.rect

                    if union_rect.width >= 150 and union_rect.height >= 120:
                        pix = page.get_pixmap(clip=union_rect, dpi=160)
                        pil_vec = Image.open(io.BytesIO(pix.tobytes("png")))
                        badge = f"Page {page_num} Chart"
                        b64_jpg, score = encode_pil_image_with_badge(pil_vec, badge_text=badge)
                        if score >= EbookIngestor.MIN_INFO_SCORE + 0.5:
                            snapshots.append(
                                VideoSnapshot(
                                    timestamp_sec=float(page_num),
                                    timestamp_str=badge,
                                    base64_jpeg=b64_jpg,
                                    info_score=score,
                                )
                            )
            except Exception:
                pass

        return snapshots

    @staticmethod
    def _read_pdf(
        path: Path,
        cache: Optional[PipelineCache],
        enable_visuals: bool,
        max_snapshots_per_chapter: int,
    ) -> List[ContentSegment]:
        doc = fitz.open(path)
        toc = doc.get_toc()
        segments: List[ContentSegment] = []

        # Build chapter page ranges: (chapter_id, title, start_page_0_idx, end_page_0_idx)
        ranges = []
        if toc:
            for i, entry in enumerate(toc):
                _, title, start_page = entry
                end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)
                s_idx = max(0, start_page - 1)
                e_idx = min(len(doc), max(s_idx + 1, end_page))
                ranges.append((f"{path.stem}_ch_{i+1}", title, s_idx, e_idx))
        else:
            pages_per_group = 10
            for start in range(0, len(doc), pages_per_group):
                end = min(start + pages_per_group, len(doc))
                ranges.append((f"{path.stem}_p_{start+1}_{end}", f"Pages {start+1}-{end}", start, end))

        for seg_id, title, s_idx, e_idx in ranges:
            text_pages = [doc[p].get_text("text") for p in range(s_idx, e_idx)]
            combined_text = "\n".join(text_pages).strip()
            if len(combined_text) <= 250:
                continue

            chapter_snapshots: List[VideoSnapshot] = []
            if enable_visuals:
                cached_snaps = cache.get_snapshots(seg_id) if cache else None
                if cached_snaps is not None:
                    chapter_snapshots = cached_snaps
                else:
                    raw_snaps: List[VideoSnapshot] = []
                    for p in range(s_idx, e_idx):
                        raw_snaps.extend(EbookIngestor._extract_pdf_page_visuals(doc, p))
                    chapter_snapshots = select_top_snapshots(raw_snaps, max_snapshots_per_chapter)
                    if cache:
                        cache.save_snapshots(seg_id, chapter_snapshots)

            segments.append(
                ContentSegment(
                    segment_id=seg_id,
                    title=title,
                    text=combined_text,
                    snapshots=chapter_snapshots,
                )
            )

        doc.close()
        return segments

    @staticmethod
    def _read_epub(
        path: Path,
        cache: Optional[PipelineCache],
        enable_visuals: bool,
        max_snapshots_per_chapter: int,
    ) -> List[ContentSegment]:
        book = epub.read_epub(str(path))
        segments: List[ContentSegment] = []

        # Map image filenames/paths inside the EPUB archive to their raw binary content
        image_lookup: Dict[str, bytes] = {}
        if enable_visuals:
            for item in book.get_items():
                media_type = (item.media_type or "").lower()
                if item.get_type() == ebooklib.ITEM_IMAGE or media_type.startswith("image/"):
                    norm_name = posixpath.normpath(item.get_name()).lstrip("/")
                    image_lookup[norm_name] = item.get_content()
                    image_lookup[posixpath.basename(norm_name)] = item.get_content()

        for idx, doc_item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), 1):
            soup = BeautifulSoup(doc_item.get_content(), "html.parser")
            heading = soup.find(["h1", "h2", "h3", "title"])
            title = heading.get_text(strip=True) if heading else doc_item.get_name()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) <= 350:
                continue

            seg_id = f"{path.stem}_sec_{idx}"
            chapter_snapshots: List[VideoSnapshot] = []

            if enable_visuals:
                cached_snaps = cache.get_snapshots(seg_id) if cache else None
                if cached_snaps is not None:
                    chapter_snapshots = cached_snaps
                else:
                    doc_dir = posixpath.dirname(doc_item.get_name())
                    img_tags = soup.find_all(["img", "image"])
                    raw_snaps: List[VideoSnapshot] = []

                    for fig_idx, tag in enumerate(img_tags, 1):
                        src = tag.get("src") or tag.get("xlink:href") or tag.get("href")
                        if not src:
                            continue
                        src_clean = src.split("#")[0].split("?")[0]
                        rel_resolved = posixpath.normpath(posixpath.join(doc_dir, src_clean)).lstrip("/")
                        img_bytes = (
                            image_lookup.get(rel_resolved)
                            or image_lookup.get(src_clean.lstrip("/"))
                            or image_lookup.get(posixpath.basename(src_clean))
                        )
                        if not img_bytes:
                            continue

                        try:
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            w, h = pil_img.size
                            if w < EbookIngestor.MIN_IMAGE_WIDTH or h < EbookIngestor.MIN_IMAGE_HEIGHT:
                                continue
                            badge = f"Ch.{idx} Fig.{fig_idx}"
                            b64_jpg, score = encode_pil_image_with_badge(pil_img, badge_text=badge)
                            if score >= EbookIngestor.MIN_INFO_SCORE:
                                raw_snaps.append(
                                    VideoSnapshot(
                                        timestamp_sec=float(idx * 100 + fig_idx),
                                        timestamp_str=badge,
                                        base64_jpeg=b64_jpg,
                                        info_score=score,
                                    )
                                )
                        except Exception:
                            continue

                    chapter_snapshots = select_top_snapshots(raw_snaps, max_snapshots_per_chapter)
                    if cache:
                        cache.save_snapshots(seg_id, chapter_snapshots)

            segments.append(
                ContentSegment(
                    segment_id=seg_id,
                    title=title,
                    text=text,
                    snapshots=chapter_snapshots,
                )
            )

        return segments

    @staticmethod
    def normalize_segments(segments: List[ContentSegment], max_chars: int) -> List[ContentSegment]:
        normalized: List[ContentSegment] = []
        for seg in segments:
            if len(seg.text) <= max_chars:
                normalized.append(seg)
            else:
                paragraphs = seg.text.split("\n\n")
                buffer = ""
                chunks_text = []
                for para in paragraphs:
                    if len(buffer) + len(para) > max_chars and buffer:
                        chunks_text.append(buffer)
                        buffer = para
                    else:
                        buffer = f"{buffer}\n\n{para}" if buffer else para
                if buffer:
                    chunks_text.append(buffer)

                total_parts = len(chunks_text)
                snaps = seg.snapshots
                for p_idx, c_text in enumerate(chunks_text):
                    start_s = int(p_idx * len(snaps) / total_parts)
                    end_s = int((p_idx + 1) * len(snaps) / total_parts)
                    normalized.append(
                        ContentSegment(
                            segment_id=f"{seg.segment_id}_pt{p_idx+1}",
                            title=f"{seg.title} (Part {p_idx+1})",
                            text=c_text,
                            snapshots=snaps[start_s:end_s],
                        )
                    )
        return normalized
```

---

### `skill_forge/ingestors/youtube.py`

```python
import time
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Optional
import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from litellm import transcription as litellm_transcription
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from ..models import ContentSegment, VideoSnapshot
from ..cache import PipelineCache
from .visual_utils import VideoSnapshotExtractor
from .ebook import EbookIngestor

console = Console()


class WhisperAudioTranscriber:
    """Thread-safe local faster-whisper / cloud API audio transcriber."""

    def __init__(
        self,
        mode: str = "local",
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        api_model: str = "groq/whisper-large-v3-turbo",
        batch_size: int = 8,
    ):
        self.mode = mode.lower()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.api_model = api_model
        self.batch_size = batch_size

        self._local_model = None
        self._batched_pipeline = None
        self._model_lock = threading.Lock()

    def _ensure_local_model_loaded(self):
        if self._local_model is not None:
            return
        import ctranslate2
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        resolved_device = self.device
        if resolved_device == "auto":
            try:
                resolved_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                resolved_device = "cpu"

        resolved_compute = self.compute_type
        if resolved_compute == "auto":
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"

        console.print(
            f"[cyan]Loading local faster-whisper '{self.model_size}' "
            f"(device={resolved_device}, compute={resolved_compute})...[/cyan]"
        )
        self._local_model = WhisperModel(self.model_size, device=resolved_device, compute_type=resolved_compute)
        try:
            self._batched_pipeline = BatchedInferencePipeline(model=self._local_model)
        except Exception:
            self._batched_pipeline = None

    @staticmethod
    def _download_audio_stream(video_id: str, temp_dir: str) -> Path:
        video_url = f"[https://www.youtube.com/watch?v=](https://www.youtube.com/watch?v=){video_id}"
        out_template = str(Path(temp_dir) / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio[ext=m4a][abr<=64]/bestaudio[abr<=64]/bestaudio[ext=m4a]/bestaudio",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        files = list(Path(temp_dir).glob(f"{video_id}.*"))
        if not files:
            raise FileNotFoundError(f"Audio stream download failed for {video_id}")
        return files[0]

    def transcribe_video(self, video_id: str, preferred_language: Optional[str] = None) -> str:
        with tempfile.TemporaryDirectory(prefix="skillforge_audio_") as tmp_dir:
            audio_path = self._download_audio_stream(video_id, tmp_dir)

            if self.mode == "api":
                with open(audio_path, "rb") as audio_file:
                    kwargs = {"model": self.api_model, "file": audio_file}
                    if preferred_language:
                        kwargs["language"] = preferred_language
                    response = litellm_transcription(**kwargs)
                    text = response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")
                    return text.strip()

            with self._model_lock:
                self._ensure_local_model_loaded()
                transcribe_kwargs = {
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 500},
                    "beam_size": 5,
                }
                if preferred_language:
                    transcribe_kwargs["language"] = preferred_language

                if self._batched_pipeline is not None and self.batch_size > 1:
                    segments, _ = self._batched_pipeline.transcribe(
                        str(audio_path), batch_size=self.batch_size, **transcribe_kwargs
                    )
                else:
                    segments, _ = self._local_model.transcribe(str(audio_path), **transcribe_kwargs)

                return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()


class YouTubeIngestor:
    """Concurrently ingests YouTube videos/playlists with captions, Whisper fallback, and visual snapshots."""

    @staticmethod
    def _process_single_video_sync(
        vid: Dict[str, str],
        cache: PipelineCache,
        languages: List[str],
        request_delay: float,
        enable_whisper_fallback: bool,
        whisper_transcriber: Optional[WhisperAudioTranscriber],
        enable_visuals: bool,
        snapshot_extractor: Optional[VideoSnapshotExtractor],
    ) -> Optional[ContentSegment]:
        vid_id = vid["id"]
        vid_title = vid["title"]

        full_text = cache.get_transcript(vid_id)
        if not full_text:
            ytt_api = YouTubeTranscriptApi()
            try:
                try:
                    transcript = ytt_api.fetch(vid_id, languages=languages)
                except NoTranscriptFound:
                    transcript_list = ytt_api.list(vid_id)
                    first_available = next(iter(transcript_list))
                    transcript = first_available.fetch()

                full_text = " ".join(snippet.text for snippet in transcript).strip()
                if request_delay > 0:
                    time.sleep(request_delay)
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, StopIteration, Exception) as cap_err:
                if not enable_whisper_fallback or whisper_transcriber is None:
                    console.print(f"[yellow]Skipping '{vid_title}' ({vid_id}) - no captions ({cap_err}).[/yellow]")
                    return None
                try:
                    full_text = whisper_transcriber.transcribe_video(vid_id)
                except Exception as w_err:
                    console.print(f"[red]Whisper fallback failed for '{vid_title}' ({vid_id}): {w_err}[/red]")
                    return None

            if full_text and len(full_text) > 200:
                cache.save_transcript(vid_id, vid_title, full_text)
            else:
                return None

        snapshots: List[VideoSnapshot] = []
        if enable_visuals and snapshot_extractor is not None:
            cached_snaps = cache.get_snapshots(f"yt_{vid_id}")
            if cached_snaps is not None:
                snapshots = cached_snaps
            else:
                try:
                    snapshots = snapshot_extractor.extract_snapshots(vid_id)
                    cache.save_snapshots(f"yt_{vid_id}", snapshots)
                except Exception as snap_err:
                    console.print(f"[yellow]Snapshot extraction skipped for '{vid_title}': {snap_err}[/yellow]")

        return ContentSegment(
            segment_id=f"yt_{vid_id}",
            title=vid_title,
            text=full_text,
            snapshots=snapshots,
        )

    @staticmethod
    async def aload(
        url: str,
        cache: PipelineCache,
        ingest_concurrency: int = 4,
        languages: Optional[List[str]] = None,
        request_delay: float = 0.5,
        max_chars_per_segment: int = 40000,
        enable_whisper_fallback: bool = True,
        whisper_transcriber: Optional[WhisperAudioTranscriber] = None,
        enable_visuals: bool = True,
        snapshot_extractor: Optional[VideoSnapshotExtractor] = None,
    ) -> List[ContentSegment]:
        if languages is None:
            languages = ["en", "ar", "es", "fr", "de", "pt", "hi", "zh-Hans"]
        if enable_whisper_fallback and whisper_transcriber is None:
            whisper_transcriber = WhisperAudioTranscriber()
        if enable_visuals and snapshot_extractor is None:
            snapshot_extractor = VideoSnapshotExtractor()

        ydl_opts = {
            "extract_flat": "in_playlist",
            "quiet": True,
            "skip_download": True,
            "ignoreerrors": True,
        }
        info = await asyncio.to_thread(lambda: yt_dlp.YoutubeDL(ydl_opts).extract_info(url, download=False))
        if not info:
            raise ValueError(f"Could not retrieve metadata for URL: {url}")

        videos = []
        if "entries" in info:
            for entry in info["entries"]:
                if entry and entry.get("id"):
                    videos.append({"id": entry["id"], "title": entry.get("title") or entry["id"]})
        else:
            videos.append({"id": info["id"], "title": info.get("title") or info["id"]})

        console.print(f"Discovered [bold cyan]{len(videos)}[/bold cyan] video(s). Ingesting with concurrency={ingest_concurrency}...")
        ingest_sem = asyncio.Semaphore(ingest_concurrency)
        ordered_results: List[Optional[ContentSegment]] = [None] * len(videos)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Ingesting transcripts & visual snapshots...", total=len(videos))

            async def _worker(idx: int, vid_meta: Dict[str, str]):
                async with ingest_sem:
                    seg = await asyncio.to_thread(
                        YouTubeIngestor._process_single_video_sync,
                        vid_meta,
                        cache,
                        languages,
                        request_delay,
                        enable_whisper_fallback,
                        whisper_transcriber,
                        enable_visuals,
                        snapshot_extractor,
                    )
                    ordered_results[idx] = seg
                    progress.advance(task_id)

            await asyncio.gather(*[_worker(i, v) for i, v in enumerate(videos)])

        valid_segments = [s for s in ordered_results if s is not None]
        return EbookIngestor.normalize_segments(valid_segments, max_chars_per_segment)
```

---

### `skill_forge/engine/__init__.py`

```python
from .pipeline import SkillForgeEngine
from .auditor import SemanticSkillAuditor

__all__ = ["SkillForgeEngine", "SemanticSkillAuditor"]
```

---

### `skill_forge/engine/auditor.py`

```python
import re
import json
from typing import List, Dict, Any
from rich.console import Console

from ..models import (
    KnowledgeExtractionChunk,
    SynthesizedAgentSkill,
    AuditPatchReport,
)
from ..llm_client import UniversalLLMClient

console = Console()


class SemanticSkillAuditor:
    """
    Pass 3 Quality Auditor & Micro-Consolidator:
      1. Builds a deterministic Ground-Truth Ledger of all formulas, quantitative thresholds,
         procedures, concepts, and visual artifacts from the extracted source chunks.
      2. Audits the synthesized Agent Skill against the ledger to detect dropped formulas,
         missing contraindications/edge cases, or incomplete procedures.
      3. Micro-consolidates closely related concepts, workflows, and visual tables inside
         the skill so there is zero bloat or redundancy.
    """

    def __init__(self, llm: UniversalLLMClient):
        self.llm = llm

    @staticmethod
    def _build_ground_truth_ledger(chunks: List[KnowledgeExtractionChunk]) -> Dict[str, Any]:
        """
        Extracts a compact, deterministic ground-truth inventory of all concepts, formulas,
        quantitative expressions, procedure titles, and visual artifacts across source chunks.
        """
        concepts_inventory = []
        formulas_and_rules = []
        procedures_inventory = []
        visuals_inventory = []
        quantitative_tokens = set()

        # Regex for equations, ratios, dosages, percentages, and numeric bounds
        quant_pattern = re.compile(
            r"(?:[A-Za-z0-9_]+\s*=\s*[^,\n;]+|\b\d+(?:\.\d+)?\s*(?:%|mg|g|kg|mcg|IU|mL|mmol|bps|x|times|days|hours|years)\b)",
            re.IGNORECASE,
        )

        for ch in chunks:
            for c in ch.theoretical_concepts:
                concepts_inventory.append(f"{c.name} ({ch.section_title})")
                for rule in c.governing_rules_or_formulas:
                    formulas_and_rules.append(f"[{c.name}] {rule}")
                    for m in quant_pattern.findall(rule):
                        quantitative_tokens.add(m.strip())

            for p in ch.applied_procedures:
                procedures_inventory.append({
                    "title": p.title,
                    "trigger": p.trigger_condition,
                    "step_count": len(p.steps),
                    "branches": p.decision_branches,
                    "anti_patterns": p.anti_patterns,
                })
                for step in p.steps:
                    for m in quant_pattern.findall(step):
                        quantitative_tokens.add(m.strip())

            for v in ch.visual_artifacts:
                visuals_inventory.append({
                    "title": v.title,
                    "type": v.artifact_type,
                    "source": v.timestamp_or_page,
                    "takeaways": v.key_takeaways,
                })

        return {
            "total_source_chunks": len(chunks),
            "concepts_inventory": concepts_inventory[:120],
            "formulas_and_governing_rules": formulas_and_rules[:120],
            "procedures_inventory": procedures_inventory[:80],
            "visual_artifacts_inventory": visuals_inventory[:60],
            "high_signal_quantitative_expressions": sorted(list(quantitative_tokens))[:120],
        }

    async def aaudit_and_consolidate(
        self,
        skill: SynthesizedAgentSkill,
        source_chunks: List[KnowledgeExtractionChunk],
    ) -> AuditPatchReport:
        """
        Executes Pass 3: Semantic Gap-Checking, Missing-Knowledge Recovery, and
        Micro-Consolidation of closely related items.
        """
        ledger = self._build_ground_truth_ledger(source_chunks)

        system_prompt = (
            "You are a Principal AI Knowledge Auditor and Ontology Architect. "
            "Your job is to perform a strict Semantic Quality Audit, Gap Recovery, and Micro-Consolidation pass "
            "on a synthesized Agent Skill."
        )

        user_prompt = f"""
Perform a comprehensive **Pass 3 Audit & Micro-Consolidation** on the candidate Agent Skill "{skill.skill_title}" (Domain: {skill.domain}).

### TASK 1: GAP-CHECKING & ZERO-LOSS RECOVERY
Compare the Candidate Skill against the Ground-Truth Source Ledger below:
- Identify any omitted formulas, equations, quantitative dosages/thresholds/ratios, decision branches, contraindications, or visual tables/diagrams.
- Re-inject every missing formula, step, edge case, or visual table into `refined_skill` and list what you recovered in `missing_gaps_recovered`.

### TASK 2: MICRO-CONSOLIDATION OF CLOSELY RELATED PIECES
Inspect `theoretical_foundations`, `primary_workflows`, `visual_frameworks_and_tables`, and `master_heuristics` inside the Candidate Skill:
- **Combine Closely Related Theoretical Concepts**: If two or more concepts cover the same underlying mechanism or model from slightly different chapters/videos, merge them into ONE comprehensive `TheoreticalConcept` that preserves all governing equations and nuances.
- **Unify Overlapping Workflows**: If two `AppliedProcedure` items address phases of the same workflow or closely related triggers, combine them into a single cohesive multi-stage workflow with explicit `decision_branches` ("If scenario A -> do X; if scenario B -> do Y") and unified `verification_checklist`.
- **Merge Fragmented Visual Tables/Diagrams**: If multiple `VisualArtifact` entries represent parts of the same table or sequential flowcharts of the same process, combine their Markdown tables or Mermaid diagrams into one complete visual framework.
- Log every merge you performed inside `consolidated_merges_made`.

--- CANDIDATE SYNTHESIZED SKILL ---
{skill.model_dump_json(indent=2)}

--- GROUND-TRUTH SOURCE LEDGER ---
{json.dumps(ledger, indent=2)}
"""
        report = await self.llm.agenerate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AuditPatchReport,
        )
        # Preserve original slug so directory naming remains deterministic
        report.refined_skill.skill_slug = skill.skill_slug
        return report

    async def aconsolidate_combined_skill(
        self,
        merged_skill: SynthesizedAgentSkill,
        input_skills: List[SynthesizedAgentSkill],
    ) -> AuditPatchReport:
        """
        Audits a Master Skill produced by combining multiple existing skills, ensuring
        no specialized workflows or formulas from the constituent skills were lost and
        merging closely related procedures across skills.
        """
        source_inventory = [
            {
                "skill_title": s.skill_title,
                "workflows": [w.title for w in s.primary_workflows],
                "concepts": [
                    {"name": c.name, "formulas": c.governing_rules_or_formulas}
                    for c in s.theoretical_foundations
                ],
                "visuals": [v.title for v in s.visual_frameworks_and_tables],
                "heuristics_count": len(s.master_heuristics),
            }
            for s in input_skills
        ]

        system_prompt = (
            "You are a Principal AI Knowledge Auditor. Verify that a combined Master Skill preserves "
            "all specialized workflows, equations, and visual frameworks from its constituent skills while "
            "merging closely related concepts and procedures."
        )
        user_prompt = f"""
Audit and micro-consolidate the Combined Master Skill "{merged_skill.skill_title}" against its {len(input_skills)} constituent skills:
1. Recover any dropped formulas, conditional branches, or visual tables from the constituent skills.
2. Combine closely related concepts, workflows, and heuristics so the Master Skill is cohesive and non-redundant.

--- COMBINED MASTER SKILL ---
{merged_skill.model_dump_json(indent=2)}

--- CONSTITUENT SKILLS INVENTORY ---
{json.dumps(source_inventory, indent=2)}
"""
        report = await self.llm.agenerate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AuditPatchReport,
        )
        report.refined_skill.skill_slug = merged_skill.skill_slug
        return report
```

---

### `skill_forge/engine/pipeline.py`

```python
import re
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from ..models import (
    ContentSegment,
    KnowledgeExtractionChunk,
    SkillClusterAssignment,
    PlaylistTaxonomyPlan,
    SynthesizedAgentSkill,
    AuditPatchReport,
)
from ..cache import PipelineCache
from ..llm_client import UniversalLLMClient
from .auditor import SemanticSkillAuditor

console = Console()


class SkillForgeEngine:
    """
    Orchestrates:
      - Pass 1 (Concurrent Map): Multimodal chunk knowledge extraction
      - Pass 1.5 (Taxonomy Router): Domain & subtopic clustering
      - Pass 2 (Level-Parallel Tree-Reduce): Hierarchical skill synthesis
      - Pass 3 (Semantic Quality Auditor): Gap recovery & micro-consolidation of closely related pieces
    """

    def __init__(self, llm: UniversalLLMClient, cache: PipelineCache):
        self.llm = llm
        self.cache = cache
        self.auditor = SemanticSkillAuditor(llm=llm)

    async def aextract_chunk_knowledge(
        self,
        segment: ContentSegment,
        domain_hint: str = "auto",
    ) -> KnowledgeExtractionChunk:
        visual_tag = f"{self.llm.model}:vis_{len(segment.snapshots)}"
        cached = self.cache.get_extraction(segment.segment_id, segment.text, visual_tag, domain_hint)
        if cached:
            return cached

        domain_instruction = (
            "Automatically identify the primary macro-domain (e.g., Medicine, Finance, Law, Nutrition, Business, Philosophy) "
            "and specific subtopic of this content."
            if domain_hint.lower() == "auto"
            else f"Focus on the domain of '{domain_hint}', and identify the specific subtopic covered."
        )

        visual_note = (
            f"IMPORTANT: {len(segment.snapshots)} visual snapshots (figures, charts, tables, or slides) are attached. "
            "Inspect every chart, table, diagram, formula, and figure. "
            "Reconstruct all visual tables, flowcharts, and equations inside `visual_artifacts` using Markdown tables or Mermaid diagrams."
            if segment.snapshots
            else "Extract any tables or structured frameworks described in the text."
        )

        system_prompt = (
            "You are a Multimodal Principal Knowledge Engineer. Extract exhaustive theoretical, applied, "
            "and visual knowledge without losing formulas, concrete numbers, edge cases, or sequential steps."
        )
        user_prompt = f"""
Analyze segment "{segment.title}" (ID: `{segment.segment_id}`).
{domain_instruction}
{visual_note}

Capture EVERY piece of:
1. Theoretical Knowledge: Core concepts, definitions, mechanisms, equations, and mental models.
2. Applied Knowledge: Step-by-step protocols, workflows, decision trees (if X then Y), prerequisites, anti-patterns, and checklists.
3. Visual Artifacts: Reconstruct all figures, tables, spreadsheets, flowcharts, and equations into `visual_artifacts`.
4. Domain Heuristics: Practical rules of thumb and numeric thresholds.

--- SOURCE CONTENT ---
{segment.text}
"""
        chunk = await self.llm.agenerate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=KnowledgeExtractionChunk,
            snapshots=segment.snapshots,
        )
        chunk.segment_id = segment.segment_id
        chunk.section_title = segment.title
        self.cache.save_extraction(segment.segment_id, segment.text, visual_tag, domain_hint, chunk)
        return chunk

    async def aextract_all_chunks(
        self,
        segments: List[ContentSegment],
        domain_hint: str = "auto",
    ) -> List[KnowledgeExtractionChunk]:
        results: List[Optional[KnowledgeExtractionChunk]] = [None] * len(segments)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Pass 1 (Map): Concurrent multimodal knowledge extraction...", total=len(segments))

            async def _extract_worker(idx: int, seg: ContentSegment):
                try:
                    results[idx] = await self.aextract_chunk_knowledge(seg, domain_hint=domain_hint)
                except Exception as exc:
                    console.print(f"[red]Error extracting '{seg.title}': {exc}[/red]")
                finally:
                    progress.advance(task_id)

            await asyncio.gather(*[_extract_worker(i, s) for i, s in enumerate(segments)])

        return [r for r in results if r is not None]

    async def aplan_playlist_taxonomy(
        self,
        chunks: List[KnowledgeExtractionChunk],
        group_by_mode: str = "subtopic",
    ) -> PlaylistTaxonomyPlan:
        catalog = [
            {
                "segment_id": c.segment_id,
                "title": c.section_title,
                "detected_domain": c.detected_domain,
                "detected_subtopic": c.detected_subtopic,
                "summary": c.executive_summary,
            }
            for c in chunks
        ]

        grouping_rule = (
            "Group segments strictly by their macro-field so each distinct field (e.g., Finance, Medicine, Nutrition, Philosophy) "
            "gets ONE comprehensive master skill."
            if group_by_mode == "domain"
            else "Group closely related segments into specialized skill clusters by domain and subtopic."
        )

        plan = await self.llm.agenerate_structured(
            system_prompt="You are an AI Taxonomy Architect organizing extracted knowledge units into clean Agent Skill clusters.",
            user_prompt=(
                f"Below is a catalog of {len(catalog)} knowledge segments.\n{grouping_rule}\n"
                "Every `segment_id` from the input catalog MUST be assigned to at least one cluster.\n\n"
                f"--- CATALOG ---\n{json.dumps(catalog, indent=2)}"
            ),
            response_model=PlaylistTaxonomyPlan,
        )

        assigned_ids = {sid for cluster in plan.clusters for sid in cluster.segment_ids}
        missing = [c for c in chunks if c.segment_id not in assigned_ids]
        if missing:
            by_dom: Dict[str, List[str]] = {}
            for mc in missing:
                by_dom.setdefault(mc.detected_domain or "General", []).append(mc.segment_id)
            for dom, sids in by_dom.items():
                slug = re.sub(r"[^a-z0-9]+", "-", dom.lower()).strip("-") + "-skill"
                plan.clusters.append(
                    SkillClusterAssignment(
                        cluster_slug=slug,
                        cluster_title=f"{dom} Core Skill",
                        macro_domain=dom,
                        segment_ids=sids,
                    )
                )
        return plan

    async def _asynthesize_batch(
        self,
        chunks: List[KnowledgeExtractionChunk],
        domain: str,
        skill_name: str,
    ) -> SynthesizedAgentSkill:
        serialized = json.dumps([c.model_dump() for c in chunks], indent=2)
        system_prompt = (
            "You are a Principal AI Agent Architect designing a production-grade Agent Skill. "
            "Preserve all technical depth, concrete metrics, formulas, visual tables/diagrams, conditional logic, and edge cases. "
            "Combine closely related concepts or procedures within this batch into cohesive, non-redundant entries."
        )
        user_prompt = f"""
Synthesize the following {len(chunks)} extracted knowledge segments into a unified Agent Skill:
- Target Skill Name: "{skill_name}"
- Domain: "{domain}"

Requirements:
1. Consolidate closely related theoretical concepts while keeping all governing equations, rules, and mechanisms intact.
2. Unify applied procedures into deterministic step-by-step workflows with explicit triggers, decision branches, anti-patterns, and checklists.
3. Preserve all `visual_artifacts` inside `visual_frameworks_and_tables` as clean Markdown tables or Mermaid diagrams.

--- EXTRACTED KNOWLEDGE CHUNKS ---
{serialized}
"""
        return await self.llm.agenerate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SynthesizedAgentSkill,
        )

    async def _amerge_skill_group(
        self,
        skills: List[SynthesizedAgentSkill],
        merged_name: str,
        domain: str,
    ) -> SynthesizedAgentSkill:
        serialized_skills = json.dumps([s.model_dump() for s in skills], indent=2)
        system_prompt = (
            "You are a Principal AI Agent Architect merging multiple specialized Agent Skills in the same field "
            "into one comprehensive Master Skill without losing granular procedures, formulas, visual tables, or edge cases."
        )
        user_prompt = f"""
Combine the following {len(skills)} Agent Skills in the field of "{domain}" into one comprehensive Master Skill named "{merged_name}".

Rules:
1. Unify overlapping or closely related workflows into comprehensive multi-stage procedures while keeping all domain-specific edge cases and verification checks.
2. When two skills present different frameworks for the same problem, preserve both as conditional branches ("Approach A when X; Approach B when Y").
3. Consolidate all theoretical foundations, visual tables/diagrams, formulas, and heuristics without dropping granular details.

--- INPUT SKILLS TO MERGE ---
{serialized_skills}
"""
        return await self.llm.agenerate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SynthesizedAgentSkill,
        )

    async def acombine_skills(
        self,
        skills: List[SynthesizedAgentSkill],
        merged_name: str,
        domain: str,
        batch_size: int = 4,
        enable_audit: bool = True,
    ) -> Tuple[SynthesizedAgentSkill, Optional[AuditPatchReport]]:
        """
        Level-Parallel Tree-Reduce for combining skills, followed by optional Pass 3 Semantic Audit.
        """
        if len(skills) == 1:
            return skills[0], None

        current_layer = skills
        layer_num = 1
        while len(current_layer) > batch_size:
            batches = [current_layer[i : i + batch_size] for i in range(0, len(current_layer), batch_size)]
            console.print(
                f"  [dim]Tree-Combine Layer {layer_num}: merging {len(current_layer)} skills "
                f"across {len(batches)} parallel batches for '{merged_name}'...[/dim]"
            )
            current_layer = await asyncio.gather(*[
                self._amerge_skill_group(
                    batch,
                    merged_name=f"{merged_name} (L{layer_num}-B{b_idx+1})",
                    domain=domain,
                )
                for b_idx, batch in enumerate(batches)
            ])
            layer_num += 1

        merged = (
            current_layer[0]
            if len(current_layer) == 1
            else await self._amerge_skill_group(current_layer, merged_name=merged_name, domain=domain)
        )

        if enable_audit:
            console.print(f"  [cyan]Pass 3 (Audit & Micro-Consolidation): Auditing combined skill '{merged_name}'...[/cyan]")
            audit_report = await self.auditor.aconsolidate_combined_skill(merged, skills)
            console.print(
                f"  [green]Audit Complete:[/green] Coverage {audit_report.coverage_score_before:.1f}% -> "
                f"{audit_report.coverage_score_after:.1f}% "
                f"(+{len(audit_report.missing_gaps_recovered)} gaps recovered, "
                f"{len(audit_report.consolidated_merges_made)} closely related items unified)."
            )
            return audit_report.refined_skill, audit_report

        return merged, None

    async def ahierarchical_synthesize(
        self,
        chunks: List[KnowledgeExtractionChunk],
        domain: str,
        skill_name: str,
        batch_size: int = 6,
        enable_audit: bool = True,
    ) -> Tuple[SynthesizedAgentSkill, Optional[AuditPatchReport]]:
        """
        Pass 2 (Parallel Tree-Reduce) + Pass 3 (Semantic Quality Audit & Micro-Consolidation).
        """
        if len(chunks) <= batch_size:
            synthesized = await self._asynthesize_batch(chunks, domain=domain, skill_name=skill_name)
        else:
            batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
            console.print(
                f"  [dim]Synthesizing {len(chunks)} segments across {len(batches)} concurrent batches for '{skill_name}'...[/dim]"
            )
            partial_skills: List[SynthesizedAgentSkill] = await asyncio.gather(*[
                self._asynthesize_batch(
                    batch,
                    domain=domain,
                    skill_name=f"{skill_name} - Part {b_idx+1}",
                )
                for b_idx, batch in enumerate(batches)
            ])
            synthesized, _ = await self.acombine_skills(
                partial_skills,
                merged_name=skill_name,
                domain=domain,
                batch_size=4,
                enable_audit=False,  # Run full ground-truth chunk audit once at the root
            )

        if enable_audit:
            console.print(f"  [cyan]Pass 3 (Audit & Gap-Check): Auditing '{skill_name}' against {len(chunks)} source chunks...[/cyan]")
            audit_report = await self.auditor.aaudit_and_consolidate(synthesized, chunks)
            console.print(
                f"  [green]Audit Complete:[/green] Coverage {audit_report.coverage_score_before:.1f}% -> "
                f"{audit_report.coverage_score_after:.1f}% "
                f"(+{len(audit_report.missing_gaps_recovered)} gaps recovered, "
                f"{len(audit_report.consolidated_merges_made)} closely related items unified)."
            )
            return audit_report.refined_skill, audit_report

        return synthesized, None
```

---

### `skill_forge/exporter.py`

```python
import re
import json
from pathlib import Path
from typing import Optional
from .models import SynthesizedAgentSkill, AuditPatchReport


class SkillExporter:
    """Writes synthesized and audited skills into domain-organized Agent Skill folders."""

    @staticmethod
    def export(
        skill: SynthesizedAgentSkill,
        output_root: str,
        audit_report: Optional[AuditPatchReport] = None,
        organize_by_domain: bool = True,
    ) -> Path:
        domain_folder = re.sub(r"[^a-z0-9]+", "-", skill.domain.lower()).strip("-") or "general"
        slug = re.sub(r"[^a-z0-9]+", "-", skill.skill_slug.lower()).strip("-") or "agent-skill"

        base_dir = Path(output_root) / domain_folder if organize_by_domain else Path(output_root)
        skill_dir = base_dir / slug
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save machine-readable skill state
        (skill_dir / "skill_data.json").write_text(skill.model_dump_json(indent=2), encoding="utf-8")

        # 2. Save Pass 3 Audit Report if present
        if audit_report is not None:
            audit_summary = {
                "coverage_score_before": audit_report.coverage_score_before,
                "coverage_score_after": audit_report.coverage_score_after,
                "missing_gaps_recovered": audit_report.missing_gaps_recovered,
                "consolidated_merges_made": audit_report.consolidated_merges_made,
            }
            (skill_dir / "audit_report.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")

        # 3. Build SKILL.md
        skill_md_lines = [
            "---",
            f"name: {slug}",
            f"description: {json.dumps(skill.trigger_description)}",
            f"domain: {skill.domain}",
            f"subtopics: {json.dumps(skill.subtopics_covered)}",
            "---",
            "",
            f"# {skill.skill_title}",
            "",
            "## Core Operating Stance",
            skill.core_persona_and_stance,
            "",
            "## Progressive Disclosure & Reference Files",
            "- For deep theoretical concepts, mental models, and governing equations, read `references/theoretical_foundations.md`.",
            "- For complete step-by-step SOPs, decision trees, and checklists, read `references/applied_playbooks.md`.",
        ]
        if skill.visual_frameworks_and_tables:
            skill_md_lines.append(
                "- For reconstructed visual tables, charts, figures, and Mermaid diagrams, read `references/visual_models_and_tables.md`."
            )

        skill_md_lines.extend(["", "## Core Workflows Summary"])
        for idx, wf in enumerate(skill.primary_workflows, 1):
            skill_md_lines.extend([
                f"### {idx}. {wf.title}",
                f"- **When to Trigger:** {wf.trigger_condition}",
                "- **Key Steps:**",
                *[f"  {s_idx}. {step}" for s_idx, step in enumerate(wf.steps, 1)],
                "",
            ])

        skill_md_lines.extend([
            "## Master Domain Heuristics",
            *[f"- {h}" for h in skill.master_heuristics],
            "",
            "## Global Anti-Patterns (Never Do This)",
            *[f"- **AVOID:** {ap}" for ap in skill.global_anti_patterns],
        ])
        (skill_dir / "SKILL.md").write_text("\n".join(skill_md_lines), encoding="utf-8")

        # 4. Build references/theoretical_foundations.md
        theory_lines = [f"# Theoretical Foundations: {skill.skill_title}\n"]
        for concept in skill.theoretical_foundations:
            theory_lines.extend([
                f"## {concept.name}",
                f"*Source Context: {concept.source_context}*\n",
                concept.explanation,
                "",
            ])
            if concept.governing_rules_or_formulas:
                theory_lines.append("**Governing Rules / Formulas:**")
                theory_lines.extend([f"- {r}" for r in concept.governing_rules_or_formulas])
                theory_lines.append("")
        (refs_dir / "theoretical_foundations.md").write_text("\n".join(theory_lines), encoding="utf-8")

        # 5. Build references/applied_playbooks.md
        playbook_lines = [f"# Applied Playbooks & Checklists: {skill.skill_title}\n"]
        for wf in skill.primary_workflows:
            playbook_lines.extend([
                f"## {wf.title}",
                f"**Trigger:** {wf.trigger_condition}\n",
                "### Prerequisites",
                *([f"- {p}" for p in wf.prerequisites] or ["- None"]),
                "",
                "### Step-by-Step Execution",
                *[f"{i}. {s}" for i, s in enumerate(wf.steps, 1)],
                "",
                "### Decision Branches & Edge Cases",
                *([f"- {b}" for b in wf.decision_branches] or ["- Standard execution path"]),
                "",
                "### Pitfalls & Anti-Patterns",
                *([f"- {a}" for a in wf.anti_patterns] or ["- None noted"]),
                "",
                "### Verification Checklist",
                *([f"- [ ] {c}" for c in wf.verification_checklist] or ["- [ ] Output verified"]),
                "\n---\n",
            ])
        (refs_dir / "applied_playbooks.md").write_text("\n".join(playbook_lines), encoding="utf-8")

        # 6. Build references/visual_models_and_tables.md
        if skill.visual_frameworks_and_tables:
            vis_lines = [f"# Visual Models, Tables & Diagrams: {skill.skill_title}\n"]
            for v in skill.visual_frameworks_and_tables:
                vis_lines.extend([
                    f"## {v.title} ({v.artifact_type})",
                    f"*Timestamp / Page Source: {v.timestamp_or_page}*\n",
                    v.structured_representation,
                    "",
                ])
                if v.key_takeaways:
                    vis_lines.append("**Key Visual Takeaways:**")
                    vis_lines.extend([f"- {k}" for k in v.key_takeaways])
                    vis_lines.append("\n---\n")
            (refs_dir / "visual_models_and_tables.md").write_text("\n".join(vis_lines), encoding="utf-8")

        return skill_dir
```

---

### `main.py`

```python
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table

from skill_forge.models import SkillClusterAssignment, SynthesizedAgentSkill
from skill_forge.cache import PipelineCache
from skill_forge.llm_client import AsyncRateLimiter, UniversalLLMClient
from skill_forge.ingestors import (
    EbookIngestor,
    YouTubeIngestor,
    WhisperAudioTranscriber,
    VideoSnapshotExtractor,
)
from skill_forge.engine import SkillForgeEngine
from skill_forge.exporter import SkillExporter

console = Console()


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash",
        help="LiteLLM model string (e.g., 'openai/gpt-4o', 'anthropic/claude-3-7-sonnet-latest', 'ollama/qwen2.5:32b').",
    )
    parser.add_argument("--vision-model", default=None, help="Dedicated Vision LLM when --model is text-only.")
    parser.add_argument("--api-base", default=None, help="Optional custom API base URL.")
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=8,
        help="Maximum simultaneous in-flight LLM API calls (asyncio Semaphore size, default: 8).",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=None,
        help="Optional strict Requests-Per-Minute cap (e.g., 50 for Tier-1 API keys).",
    )
    parser.add_argument("--batch-size", type=int, default=6, help="Segments per batch during parallel tree reduction.")
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable Pass 3 Semantic Quality Auditing & Micro-Consolidation (enabled by default).",
    )
    parser.add_argument("--cache-db", default="./skill_forge_cache.db", help="Path to SQLite WAL cache.")
    parser.add_argument("--output", default="./skills_output", help="Root output directory.")


async def async_main():
    parser = argparse.ArgumentParser(
        description="SkillForge v4: Modular Multimodal Ebook & YouTube Playlist to Audited Agent Skill Compiler."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command 1: Ebook (PDF / EPUB with embedded figures & vector charts)
    ebook_parser = subparsers.add_parser("ebook", help="Extract agent skill(s) from a PDF or EPUB with embedded visuals.")
    ebook_parser.add_argument("file", help="Path to the .pdf or .epub file.")
    ebook_parser.add_argument("--domain", default="auto", help="Field name, or 'auto' to auto-detect.")
    ebook_parser.add_argument("--name", required=True, help="Target name for the skill.")
    ebook_parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Disable extracting embedded figures, tables, and vector charts from the ebook.",
    )
    ebook_parser.add_argument(
        "--max-snapshots",
        type=int,
        default=10,
        help="Maximum visual figures/charts extracted per chapter (default: 10).",
    )
    add_common_args(ebook_parser)

    # Command 2: YouTube (Video / Multi-Domain Playlist)
    yt_parser = subparsers.add_parser("youtube", help="Extract skills from a YouTube video or diverse playlist.")
    yt_parser.add_argument("url", help="YouTube video or playlist URL.")
    yt_parser.add_argument("--domain", default="auto", help="Set 'auto' for diverse playlists, or specify a domain.")
    yt_parser.add_argument("--group-by", choices=["subtopic", "domain", "single"], default="subtopic")
    yt_parser.add_argument("--auto-combine-domains", action="store_true")
    yt_parser.add_argument("--name", default="Playlist Skill")
    yt_parser.add_argument("--ingest-concurrency", type=int, default=4)
    yt_parser.add_argument("--yt-delay", type=float, default=0.5)
    yt_parser.add_argument("--no-whisper-fallback", action="store_true")
    yt_parser.add_argument("--whisper-mode", choices=["local", "api"], default="local")
    yt_parser.add_argument("--whisper-model", default="small")
    yt_parser.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"], default="auto")
    yt_parser.add_argument("--whisper-api-model", default="groq/whisper-large-v3-turbo")
    yt_parser.add_argument("--no-visuals", action="store_true", help="Disable video snapshot extraction.")
    yt_parser.add_argument("--max-snapshots", type=int, default=12)
    yt_parser.add_argument("--snapshot-interval", type=float, default=8.0)
    add_common_args(yt_parser)

    # Command 3: Combine Existing Skills
    combine_parser = subparsers.add_parser("combine", help="Combine multiple existing skills concurrently.")
    combine_parser.add_argument("paths", nargs="+")
    combine_parser.add_argument("--by-domain", action="store_true")
    combine_parser.add_argument("--domain", default="General")
    combine_parser.add_argument("--name", default="Comprehensive Master Skill")
    add_common_args(combine_parser)

    args = parser.parse_args()

    rate_limiter = AsyncRateLimiter(max_concurrency=args.llm_concurrency, rpm_limit=args.rpm_limit)
    cache = PipelineCache(db_path=args.cache_db)
    llm = UniversalLLMClient(
        model=args.model,
        vision_model=args.vision_model,
        api_base=args.api_base,
        rate_limiter=rate_limiter,
    )
    engine = SkillForgeEngine(llm=llm, cache=cache)
    enable_audit = not args.no_audit

    # -----------------------------------------------------------------
    # MODE A: EBOOK (Text + Embedded Figures/Vector Charts + Audit)
    # -----------------------------------------------------------------
    if args.command == "ebook":
        console.print(
            f"[bold cyan]Ingesting ebook '{args.file}' (visuals={not args.no_visuals}, "
            f"model={args.model}, concurrency={args.llm_concurrency})...[/bold cyan]"
        )
        segments = await asyncio.to_thread(
            EbookIngestor.load,
            args.file,
            cache=cache,
            enable_visuals=not args.no_visuals,
            max_snapshots_per_chapter=args.max_snapshots,
        )
        total_figs = sum(len(s.snapshots) for s in segments)
        console.print(
            f"Extracted [bold green]{len(segments)}[/bold green] chapter segments and "
            f"[bold green]{total_figs}[/bold green] embedded figures/charts."
        )

        chunks = await engine.aextract_all_chunks(segments, domain_hint=args.domain)
        resolved_domain = chunks[0].detected_domain if (args.domain == "auto" and chunks) else args.domain

        console.print(f"[bold cyan]Pass 2 & 3: Synthesizing and auditing '{args.name}' ({resolved_domain})...[/bold cyan]")
        skill, audit_report = await engine.ahierarchical_synthesize(
            chunks,
            domain=resolved_domain,
            skill_name=args.name,
            batch_size=args.batch_size,
            enable_audit=enable_audit,
        )
        out_path = SkillExporter.export(skill, args.output, audit_report=audit_report)
        console.print(f"[bold green]Exported Audited Ebook Skill to:[/bold green] {out_path}")

    # -----------------------------------------------------------------
    # MODE B: YOUTUBE (Transcripts + Whisper + Visual Keyframes + Audit)
    # -----------------------------------------------------------------
    elif args.command == "youtube":
        whisper_transcriber = WhisperAudioTranscriber(
            mode=args.whisper_mode,
            model_size=args.whisper_model,
            device=args.whisper_device,
            api_model=args.whisper_api_model,
        )
        snapshot_extractor = VideoSnapshotExtractor(
            max_snapshots=args.max_snapshots,
            sample_interval_sec=args.snapshot_interval,
        )

        segments = await YouTubeIngestor.aload(
            args.url,
            cache=cache,
            ingest_concurrency=args.ingest_concurrency,
            request_delay=args.yt_delay,
            enable_whisper_fallback=not args.no_whisper_fallback,
            whisper_transcriber=whisper_transcriber,
            enable_visuals=not args.no_visuals,
            snapshot_extractor=snapshot_extractor,
        )
        if not segments:
            console.print("[red]No segments retrieved.[/red]")
            return

        chunks = await engine.aextract_all_chunks(segments, domain_hint=args.domain)
        if not chunks:
            console.print("[red]No knowledge chunks extracted.[/red]")
            return

        if args.group_by == "single" or len(chunks) == 1:
            dom = chunks[0].detected_domain if args.domain == "auto" else args.domain
            title = args.name if args.name != "Playlist Skill" else chunks[0].section_title
            skill, audit_report = await engine.ahierarchical_synthesize(
                chunks, domain=dom, skill_name=title, batch_size=args.batch_size, enable_audit=enable_audit
            )
            out_path = SkillExporter.export(skill, args.output, audit_report=audit_report)
            console.print(f"[bold green]Exported Audited Skill to:[/bold green] {out_path}")
            return

        console.print(f"[bold cyan]Pass 1.5: Clustering {len(chunks)} segments across fields (mode: {args.group_by})...[/bold cyan]")
        taxonomy = await engine.aplan_playlist_taxonomy(chunks, group_by_mode=args.group_by)

        table = Table(title="Discovered Playlist Skill Clusters")
        table.add_column("Macro Domain", style="cyan")
        table.add_column("Skill Cluster Title", style="green")
        table.add_column("Segments", justify="right")
        for cl in taxonomy.clusters:
            table.add_row(cl.macro_domain, cl.cluster_title, str(len(cl.segment_ids)))
        console.print(table)

        chunk_map = {c.segment_id: c for c in chunks}

        async def _synthesize_cluster(cl: SkillClusterAssignment) -> Optional[SynthesizedAgentSkill]:
            c_chunks = [chunk_map[sid] for sid in cl.segment_ids if sid in chunk_map]
            if not c_chunks:
                return None
            skill, audit_report = await engine.ahierarchical_synthesize(
                c_chunks,
                domain=cl.macro_domain,
                skill_name=cl.cluster_title,
                batch_size=args.batch_size,
                enable_audit=enable_audit,
            )
            skill.skill_slug = cl.cluster_slug
            out_path = SkillExporter.export(skill, args.output, audit_report=audit_report)
            console.print(f"  [green]-> Exported '{cl.cluster_title}' [{cl.macro_domain}]:[/green] {out_path}")
            return skill

        console.print(f"[bold cyan]Pass 2 & 3: Synthesizing and auditing {len(taxonomy.clusters)} clusters concurrently...[/bold cyan]")
        synthesized_clusters = await asyncio.gather(*[_synthesize_cluster(cl) for cl in taxonomy.clusters])

        if args.auto_combine_domains:
            skills_by_domain: Dict[str, List[SynthesizedAgentSkill]] = {}
            for sk in synthesized_clusters:
                if sk is not None:
                    skills_by_domain.setdefault(sk.domain, []).append(sk)

            async def _combine_domain_master(dom: str, dom_skills: List[SynthesizedAgentSkill]):
                master_title = f"Comprehensive {dom} Master Skill"
                console.print(f"[bold magenta]Combining & auditing {len(dom_skills)} '{dom}' skills into '{master_title}'...[/bold magenta]")
                master_skill, audit_report = await engine.acombine_skills(
                    dom_skills,
                    merged_name=master_title,
                    domain=dom,
                    batch_size=args.batch_size,
                    enable_audit=enable_audit,
                )
                master_path = SkillExporter.export(master_skill, args.output, audit_report=audit_report)
                console.print(f"  [bold green]-> Saved Audited Master Domain Skill:[/bold green] {master_path}")

            await asyncio.gather(*[
                _combine_domain_master(dom, d_skills)
                for dom, d_skills in skills_by_domain.items()
                if len(d_skills) > 1
            ])

    # -----------------------------------------------------------------
    # MODE C: COMBINE & AUDIT EXISTING SKILLS
    # -----------------------------------------------------------------
    elif args.command == "combine":
        discovered_files: List[Path] = []
        for p_str in args.paths:
            p = Path(p_str)
            if (p / "skill_data.json").exists():
                discovered_files.append(p / "skill_data.json")
            else:
                discovered_files.extend(p.rglob("skill_data.json"))

        if not discovered_files:
            raise FileNotFoundError("No 'skill_data.json' files found in the specified paths.")

        loaded_skills = [
            SynthesizedAgentSkill.model_validate_json(f.read_text(encoding="utf-8"))
            for f in discovered_files
        ]

        if args.by_domain:
            grouped: Dict[str, List[SynthesizedAgentSkill]] = {}
            for sk in loaded_skills:
                grouped.setdefault(sk.domain.strip().title(), []).append(sk)

            async def _merge_domain_worker(dom: str, dom_skills: List[SynthesizedAgentSkill]):
                master_name = f"Comprehensive {dom} Master Skill"
                merged, audit_report = await engine.acombine_skills(
                    dom_skills,
                    merged_name=master_name,
                    domain=dom,
                    batch_size=args.batch_size,
                    enable_audit=enable_audit,
                )
                out_path = SkillExporter.export(merged, args.output, audit_report=audit_report)
                console.print(f"[bold green]Exported Audited '{dom}' Master Skill to:[/bold green] {out_path}")

            await asyncio.gather(*[
                _merge_domain_worker(dom, dom_skills)
                for dom, dom_skills in grouped.items()
                if len(dom_skills) > 1
            ])
        else:
            merged, audit_report = await engine.acombine_skills(
                loaded_skills,
                merged_name=args.name,
                domain=args.domain,
                batch_size=args.batch_size,
                enable_audit=enable_audit,
            )
            out_path = SkillExporter.export(merged, args.output, audit_report=audit_report)
            console.print(f"[bold green]Exported Audited Combined Skill to:[/bold green] {out_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
```

---

## 4. Quick-Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract an Agent Skill from a PDF/EPUB (automatically extracts embedded figures & vector charts + runs Pass 3 Audit)
python main.py ebook "./books/valuation_and_financial_modeling.pdf" \
  --model "gemini/gemini-2.5-flash" \
  --domain "Finance" \
  --name "Institutional Valuation & Financial Modeling" \
  --llm-concurrency 10

# 3. Extract multi-domain skills from a large YouTube playlist (with Whisper fallback, visual snapshots & Pass 3 Audit)
python main.py youtube "[https://www.youtube.com/playlist?list=PL_MIXED_PLAYLIST](https://www.youtube.com/playlist?list=PL_MIXED_PLAYLIST)" \
  --model "anthropic/claude-3-7-sonnet-latest" \
  --domain auto \
  --group-by subtopic \
  --auto-combine-domains \
  --llm-concurrency 6 \
  --rpm-limit 50

# 4. Combine & micro-consolidate all generated skills across domains
python main.py combine "./skills_output" \
  --by-domain \
  --model "openai/gpt-4o"
```
