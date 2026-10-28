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
