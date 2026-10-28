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
