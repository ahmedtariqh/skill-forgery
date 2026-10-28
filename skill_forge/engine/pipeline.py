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
