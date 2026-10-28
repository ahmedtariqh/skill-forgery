import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    default_model = (
        os.getenv("LLM_MODEL")
        or os.getenv("OMNIROUTE_CHAT_MODEL")
        or "gemini/gemini-2.5-flash"
    )
    default_api_base = os.getenv("LLM_API_BASE") or os.getenv("OMNIROUTE_API_BASE")
    default_api_key = os.getenv("LLM_API_KEY") or os.getenv("OMNIROUTE_API_KEY")

    parser.add_argument(
        "--model",
        default=default_model,
        help=(
            f"LiteLLM model string (default: '{default_model}'). "
            "Examples: 'openrouter/deepseek/deepseek-r1', 'openai/chat', 'ollama/qwen2.5:32b'."
        ),
    )
    parser.add_argument("--vision-model", default=None, help="Dedicated Vision LLM when --model is text-only.")
    parser.add_argument(
        "--api-base",
        default=default_api_base,
        help="Custom API base URL (defaults to LLM_API_BASE or OMNIROUTE_API_BASE from env).",
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        help="Custom API key (defaults to LLM_API_KEY or OMNIROUTE_API_KEY from env).",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1).")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per LLM completion.")
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
    default_whisper_mode = "api" if (os.getenv("WHISPER_API_MODEL") or os.getenv("OMNIROUTE_STT_MODEL")) else "local"
    default_whisper_api = (
        os.getenv("WHISPER_API_MODEL")
        or os.getenv("OMNIROUTE_STT_MODEL")
        or "groq/whisper-large-v3-turbo"
    )
    yt_parser.add_argument("--whisper-mode", choices=["local", "api"], default=default_whisper_mode)
    yt_parser.add_argument("--whisper-model", default="small")
    yt_parser.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"], default="auto")
    yt_parser.add_argument("--whisper-api-model", default=default_whisper_api)
    yt_parser.add_argument(
        "--whisper-api-base",
        default=os.getenv("WHISPER_API_BASE") or os.getenv("OMNIROUTE_API_BASE"),
        help="Optional API base URL for Whisper API endpoint.",
    )
    yt_parser.add_argument(
        "--whisper-api-key",
        default=os.getenv("WHISPER_API_KEY") or os.getenv("OMNIROUTE_API_KEY"),
        help="Optional API key for Whisper API endpoint.",
    )
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
    custom_model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    custom_headers = json.loads(args.extra_headers) if args.extra_headers else {}

    rate_limiter = AsyncRateLimiter(max_concurrency=args.llm_concurrency, rpm_limit=args.rpm_limit)
    cache = PipelineCache(db_path=args.cache_db)
    llm = UniversalLLMClient(
        model=args.model,
        vision_model=args.vision_model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_kwargs=custom_model_kwargs,
        extra_headers=custom_headers,
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
            api_base=args.whisper_api_base,
            api_key=args.whisper_api_key,
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
