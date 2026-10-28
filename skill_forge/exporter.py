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
