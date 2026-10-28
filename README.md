# SkillForge

> **Multimodal Ebook & YouTube Playlist to Audited Agent Skill Compiler**

SkillForge transforms dense, unstructured knowledge sources—such as multimodal **PDFs**, **EPUBs**, individual **YouTube videos**, and diverse **YouTube playlists**—into production-ready, standardized agent skills (`SKILL.md`).

SkillForge extracts and transcribes textual and visual artifacts (raster figures, vector diagrams, tables, flowcharts, presentation slides), clusters concepts, and employs a multi-pass hierarchical synthesis pipeline with a **Semantic Quality & Gap Audit** to ensure zero knowledge loss and clean consolidation.

---

## Key Highlights

- **Multimodal Ebook Visual Ingestion (`ingestors/ebook.py`)**:
  - **PDF Extraction via PyMuPDF (`fitz`)**: Extracts both embedded raster images (`page.get_images()`) and vector-drawn figures/charts/tables (by clustering bounding boxes from `page.get_drawings()` and rendering crisp, high-DPI crops via `page.get_pixmap()`).
  - **EPUB Extraction via `ebooklib` & `BeautifulSoup`**: Resolves `<img>` / `<image>` elements against the archive manifest, filters out tiny icons/spacers, and preserves chapter context.
  - **Computer Vision Filtering & Scoring (`visual_utils.py`)**: Evaluates Canny edge density and contrast with OpenCV, skipping blank, decorative, or low-information images.
- **Multimodal YouTube Ingestion (`ingestors/youtube.py`)**:
  - Fast transcript extraction with automatic fallback to **Whisper** (Local via `faster-whisper` / `CTranslate2` or Cloud API via Groq / OpenAI-compatible endpoints).
  - Video keyframe extraction (`yt-dlp` + `opencv-python-headless`) for slides, code walkthroughs, and diagrams.
- **Hierarchical Map-Reduce Synthesis (`engine/pipeline.py`)**:
  - Automatically clusters playlist videos or book chapters by subtopic or domain.
  - Reduces raw knowledge in concurrent tree batches, mitigating token context overflow.
- **Pass 3: Semantic Quality Auditing & Gap Recovery (`engine/auditor.py`)**:
  - Builds a deterministic ground-truth ledger of formulas, equations, dosage thresholds, and protocol names.
  - Verifies candidate skills against source chunks, recovering dropped edge cases or formulas.
  - Performs micro-consolidation of duplicate concepts and overlapping workflows.
- **Universal LLM & Multimodal Vision Support (`llm_client.py`)**:
  - Powered by `litellm`—use any model: Google Gemini, Anthropic Claude, OpenAI GPT, Ollama, DeepSeek, or custom API bases.
  - Built-in token-bucket rate limiting (RPM cap + asyncio semaphore concurrency control) and SQLite WAL persistent caching (`cache.py`).
- **Standardized Agent Skill Output (`exporter.py`)**:
  - Exports standard agent bundles with YAML frontmatter, execution heuristics, step-by-step SOPs, decision trees, Mermaid.js diagrams, Markdown tables, and `audit_report.json`.

---

## Architecture Overview

```text
                     ┌───────────────────────────────┐
                     │    Sources: Ebook / YouTube   │
                     └───────────────┬───────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
   [ Ebook Ingestor ]                                  [ YouTube Ingestor ]
   • PDF text & raster images                          • Transcripts (API)
   • Vector chart/table clustering                     • Whisper fallback (Local/API)
   • EPUB XHTML & manifest visual extraction           • Keyframe snapshot extraction
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     │
                         [ Visual Quality Filter ]
                         • Canny edge detection
                         • Contrast & entropy scoring
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Pass 1: Structured Chunk   │
                      │         Extraction          │
                      │  (Text + Vision Multimodal) │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   Pass 2: Hierarchical      │
                      │   Map-Reduce Synthesis      │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Pass 3: Semantic Audit,  │
                      │  Gap Recovery & Merge Pass  │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Standard Agent Skill Pack  │
                      │  • SKILL.md                 │
                      │  • skill_data.json          │
                      │  • audit_report.json        │
                      │  • references/              │
                      └─────────────────────────────┘
```

---

## Project Structure

```text
skill-forgery/
├── requirements.txt            # Project dependencies
├── main.py                     # CLI entry point (ebook, youtube, combine)
├── skill_forge/
│   ├── __init__.py             # Package init & version metadata
│   ├── models.py               # Pydantic schemas (Concepts, Procedures, Visuals, Skills)
│   ├── cache.py                # SQLite WAL cache for chunk extractions & LLM responses
│   ├── llm_client.py           # Universal LiteLLM async client & rate limiter
│   ├── exporter.py             # Agent skill markdown & reference generator
│   ├── ingestors/
│   │   ├── __init__.py         # Ingestor exports
│   │   ├── visual_utils.py     # Image scoring, Canny edge detection, resizing, badges
│   │   ├── ebook.py            # PDF/EPUB parser (vector drawing clusters + raster images)
│   │   └── youtube.py          # YouTube transcript fetcher, Whisper transcriber & keyframes
│   └── engine/
│       ├── __init__.py         # Engine exports
│       ├── pipeline.py         # Pass 1 extraction & Pass 2 hierarchical synthesizer
│       └── auditor.py          # Pass 3 semantic auditor, gap recoverer & consolidator
```

---

## Installation

### 1. Prerequisites
- Python 3.10+
- (Optional) `ffmpeg` installed on your system if extracting video keyframes or processing audio with local Whisper.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables & Custom LLM Provider

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

#### Custom Omniroute Provider (Preconfigured)
SkillForge natively supports the custom **Omniroute** OpenAI-compatible provider for both multimodal LLM operations and Speech-to-Text transcription:

```env
# Custom Omniroute Provider
OMNIROUTE_API_BASE=https://ubuntu-proot.taild785f2.ts.net/v1
OMNIROUTE_API_KEY=sk-35ab26f1551c058d-6e9297-a9809721
OMNIROUTE_CHAT_MODEL=omniroute/chat
OMNIROUTE_STT_MODEL=omniroute/stt

# Defaults for execution
LLM_MODEL=openai/chat
LLM_API_BASE=https://ubuntu-proot.taild785f2.ts.net/v1
LLM_API_KEY=sk-35ab26f1551c058d-6e9297-a9809721
WHISPER_API_MODEL=openai/stt
WHISPER_API_BASE=https://ubuntu-proot.taild785f2.ts.net/v1
WHISPER_API_KEY=sk-35ab26f1551c058d-6e9297-a9809721
```

- **`chat`**: Multimodal LLM handling structured extraction, synthesis, and image analysis.
- **`stt`**: High-accuracy speech-to-text model for YouTube audio transcription fallback.

#### Standard Cloud Providers (Optional)
Alternatively, you can provide official API keys:
```bash
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
```

---

## Quick-Start CLI Usage

SkillForge provides three primary CLI subcommands: `ebook`, `youtube`, and `combine`.

### 1. Compile Skills from Ebooks (PDF & EPUB)

Extracts chapter texts, embedded raster images, and clustered vector tables/diagrams:

```bash
python main.py ebook "./books/valuation_and_financial_modeling.pdf" \
  --model "gemini/gemini-2.5-flash" \
  --domain "Finance" \
  --name "Institutional Valuation & Financial Modeling" \
  --llm-concurrency 10
```

For EPUB files:
```bash
python main.py ebook "./books/microservices_patterns.epub" \
  --model "gemini/gemini-2.5-flash" \
  --domain "Software Architecture" \
  --name "Microservices Architecture Patterns"
```

### 2. Compile Skills from YouTube (Videos & Diverse Playlists)

Extracts video transcripts (with automatic Whisper fallback) and keyframe visual snapshots:

```bash
# Process a multi-topic playlist with subtopic clustering & Pass 3 audit
python main.py youtube "https://www.youtube.com/playlist?list=PL_YOUR_PLAYLIST_ID" \
  --model "gemini/gemini-2.5-flash" \
  --domain auto \
  --group-by subtopic \
  --auto-combine-domains \
  --llm-concurrency 6 \
  --rpm-limit 50
```

Using Anthropic Claude with local Whisper fallback on GPU:
```bash
python main.py youtube "https://www.youtube.com/watch?v=VIDEO_ID" \
  --model "anthropic/claude-3-7-sonnet-latest" \
  --whisper-mode local \
  --whisper-model small \
  --whisper-device cuda \
  --name "High-Performance Systems Design"
```

Using Groq Whisper Cloud API for audio transcription:
```bash
python main.py youtube "https://www.youtube.com/watch?v=VIDEO_ID" \
  --model "gemini/gemini-2.5-flash" \
  --whisper-mode api \
  --whisper-api-model "groq/whisper-large-v3-turbo" \
  --name "Advanced AI Agents"
```

### 3. Combine & Consolidate Existing Skills

Combines multiple synthesized skills into unified master skills, micro-consolidating redundant workflows and running audit verification:

```bash
# Combine all generated skills in the output directory, grouped by domain
python main.py combine "./skills_output" \
  --by-domain \
  --model "gemini/gemini-2.5-flash"
```

---

## CLI Options & Flags

### Common Flags (available on all subcommands)

| Flag | Default | Description |
|---|---|---|
| `--model` | `gemini/gemini-2.5-flash` | LiteLLM model identifier (`openai/gpt-4o`, `anthropic/claude-3-7-sonnet-latest`, `ollama/qwen2.5:32b`, etc.). |
| `--vision-model` | `None` | Dedicated vision model if `--model` is text-only. |
| `--api-base` | `None` | Custom API base URL (e.g. for vLLM, Ollama, or local proxies). |
| `--llm-concurrency` | `8` | Maximum concurrent in-flight LLM calls (`asyncio.Semaphore`). |
| `--rpm-limit` | `None` | Strict requests-per-minute rate limit cap. |
| `--batch-size` | `6` | Batch size during hierarchical tree reduction. |
| `--no-audit` | `False` | Skips Pass 3 Semantic Quality & Gap Audit. |
| `--cache-db` | `./skill_forge_cache.db` | SQLite cache file path. |
| `--output` | `./skills_output` | Destination directory for exported skills. |

### Ebook-Specific Flags (`ebook`)

| Flag | Default | Description |
|---|---|---|
| `file` | *(required)* | Path to `.pdf` or `.epub` document. |
| `--name` | *(required)* | Target skill title. |
| `--domain` | `auto` | Domain name or `auto` for automated domain detection. |
| `--no-visuals` | `False` | Disables extraction of figures, vector charts, and tables. |
| `--max-snapshots` | `10` | Maximum visuals extracted per chapter. |

### YouTube-Specific Flags (`youtube`)

| Flag | Default | Description |
|---|---|---|
| `url` | *(required)* | YouTube video or playlist URL. |
| `--name` | `Playlist Skill` | Target skill title. |
| `--domain` | `auto` | Domain name or `auto` for automated clustering. |
| `--group-by` | `subtopic` | Playlist clustering mode: `subtopic`, `domain`, or `single`. |
| `--auto-combine-domains`| `False` | Automatically combines subtopic skills within each domain. |
| `--ingest-concurrency` | `4` | Concurrency for downloading transcripts and snapshots. |
| `--yt-delay` | `0.5` | Politeness delay between YouTube queries (seconds). |
| `--no-whisper-fallback`| `False` | Disable Whisper if official/auto captions are missing. |
| `--whisper-mode` | `local` | `local` (faster-whisper) or `api` (LiteLLM audio endpoint). |
| `--whisper-model` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `--whisper-device` | `auto` | Device for local Whisper: `auto`, `cuda`, or `cpu`. |
| `--whisper-api-model` | `groq/whisper-large-v3-turbo` | Model string when `--whisper-mode api`. |
| `--no-visuals` | `False` | Disables video keyframe snapshot extraction. |
| `--max-snapshots` | `12` | Maximum visual snapshots per video. |
| `--snapshot-interval` | `8.0` | Minimum seconds between keyframe captures. |

---

## Output Format

Each exported skill creates a clean, self-contained bundle formatted for LLM agents:

```text
skills_output/
└── <domain>/
    └── <skill-slug>/
        ├── SKILL.md            # Standard agent instructions (YAML frontmatter + SOPs)
        ├── skill_data.json     # Pydantic schema serialization of the skill
        ├── audit_report.json   # Pass 3 audit scores, recovered gaps & merges
        └── references/
            ├── figures_and_tables.md  # Reconstructed Mermaid diagrams & Markdown tables
            └── audit_report.json
```

### Sample `SKILL.md` Structure
```markdown
---
name: valuation-financial-modeling
description: "Comprehensive operational skill for corporate valuation and financial modeling..."
domain: Finance
subtopics: ["DCF", "LBO", "Cost of Capital", "Comps"]
---

# Institutional Valuation & Financial Modeling

## Core Operating Stance
[World-class persona instructions and execution mindset]

## High-Leverage Heuristics
- [Deterministic rule with quantitative threshold]
- [Conditional heuristic: IF market volatility > 30% THEN ...]

## Theoretical Foundations
### Discounted Cash Flow Mechanics
...

## Primary Applied Workflows
### Procedure: 3-Statement Model Integration
...

## Visual Frameworks & Tables
[Mermaid.js charts, system diagrams, and Markdown financial tables]
```

---

## License

MIT License. Feel free to use, modify, and distribute for personal and commercial agent workflows.
