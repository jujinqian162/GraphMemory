from __future__ import annotations

import hashlib

from graph_memory.query_synthesis.provenance.catalog import (
    DEFAULT_TEMPLATE_CATALOG,
)
from graph_memory.query_synthesis.provenance.contracts import (
    MotifSpec,
    QueryIntent,
    ProvenanceQueryExample,
    ProvenanceQueryLabel,
    ProvenanceQueryRecord,
    QueryTemplate,
    TemplateCatalog,
    TemplateGenerationProvenance,
)


def templates_for(
    motif: MotifSpec,
    query_intent: QueryIntent,
    *,
    catalog: TemplateCatalog = DEFAULT_TEMPLATE_CATALOG,
) -> tuple[QueryTemplate, ...]:
    matches = tuple(
        template
        for template in catalog.templates
        if template.motif_type == motif.motif_type
        and template.query_intent == query_intent
    )
    if not matches:
        detail = (
            f"no templates for motif_type={motif.motif_type} query_intent={query_intent}"
        )
        raise ValueError(detail)
    return matches


def _select_template(
    motif: MotifSpec,
    query_intent: QueryIntent,
    *,
    seed: int,
    catalog: TemplateCatalog,
) -> QueryTemplate:
    templates = templates_for(motif, query_intent, catalog=catalog)
    selection_key = "\0".join(
        (catalog.version, str(seed), motif.motif_id, query_intent)
    )
    digest = hashlib.sha256(selection_key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(templates)
    return templates[index]


def verbalize_motif(
    motif: MotifSpec,
    query_intent: QueryIntent,
    *,
    seed: int = 13,
    catalog: TemplateCatalog = DEFAULT_TEMPLATE_CATALOG,
    template_id: str | None = None,
) -> ProvenanceQueryExample:
    target = motif.target_for(query_intent)
    if template_id is None:
        template = _select_template(
            motif, query_intent, seed=seed, catalog=catalog
        )
    else:
        matches = [
            candidate
            for candidate in templates_for(
                motif, query_intent, catalog=catalog
            )
            if candidate.template_id == template_id
        ]
        if len(matches) != 1:
            detail = (
                f"template_id={template_id!r} is not valid for motif={motif.motif_id} "
                f"query_intent={query_intent}"
            )
            raise ValueError(detail)
        template = matches[0]

    missing = set(template.required_slots) - set(target.safe_slots)
    if missing:
        raise ValueError(
            f"motif={motif.motif_id} lacks safe template slots: {sorted(missing)}"
        )
    render_slots = {
        slot: target.safe_slots[slot] for slot in template.required_slots
    }
    query_text = template.text.format_map(render_slots).strip()
    query_identity = "\0".join(
        (
            motif.graph_id,
            motif.motif_id,
            query_intent,
            template.template_id,
            str(seed),
        )
    )
    query_id = f"query:{hashlib.sha256(query_identity.encode()).hexdigest()[:20]}"
    query = ProvenanceQueryRecord(
        query_id=query_id,
        graph_id=motif.graph_id,
        query_text=query_text,
    )
    label = ProvenanceQueryLabel(
        query_id=query_id,
        motif_id=motif.motif_id,
        motif_type=motif.motif_type,
        query_intent=query_intent,
        answer_output_ids=target.answer_output_ids,
        support_output_ids=target.support_output_ids,
        answer_evidence_spans=target.answer_evidence_spans,
        support_evidence_spans=target.support_evidence_spans,
        dependencies=motif.dependencies,
    )
    generation = TemplateGenerationProvenance(
        template_id=template.template_id,
        template_catalog_version=catalog.version,
        style_tags=template.style_tags,
        generation_seed=seed,
    )
    return ProvenanceQueryExample(
        query=query,
        label=label,
        generation=generation,
    )


def verbalize_all_templates(
    motif: MotifSpec,
    query_intent: QueryIntent,
    *,
    seed: int = 13,
    catalog: TemplateCatalog = DEFAULT_TEMPLATE_CATALOG,
) -> tuple[ProvenanceQueryExample, ...]:
    return tuple(
        verbalize_motif(
            motif,
            query_intent,
            seed=seed,
            catalog=catalog,
            template_id=template.template_id,
        )
        for template in templates_for(motif, query_intent, catalog=catalog)
    )


__all__ = [
    "templates_for",
    "verbalize_all_templates",
    "verbalize_motif",
]
