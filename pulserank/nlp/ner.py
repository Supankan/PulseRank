"""Named Entity Recognition (NER) pipeline and MIND Wikidata entity resolution."""

import re
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    text: str
    label: str  # PER, ORG, LOC, MISC
    start: int
    end: int
    confidence: float = 1.0
    wikidata_id: Optional[str] = None


class NamedEntityPipeline:
    """
    Extracts named entities from article content, resolves them against MIND Wikidata metadata,
    and computes precision, recall, and F1 alignment metrics.
    """

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self._hf_pipeline = None

        if use_transformer:
            try:
                from transformers import pipeline
                self._hf_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                )
                logger.info("Loaded transformer NER pipeline (dslim/bert-base-NER).")
            except Exception as e:
                logger.warning(f"Could not load transformer NER pipeline ({e}). Falling back to pattern-based NER.")
                self._hf_pipeline = None

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities with token spans, labels, and confidence scores."""
        if not text or not text.strip():
            return []

        if self._hf_pipeline is not None:
            try:
                raw_entities = self._hf_pipeline(text)
                entities = []
                for item in raw_entities:
                    label = item.get("entity_group", "MISC")
                    # Normalize label names
                    if label in ("PER", "PERSON"):
                        norm_label = "PER"
                    elif label in ("ORG", "ORGANIZATION"):
                        norm_label = "ORG"
                    elif label in ("LOC", "LOCATION", "GPE"):
                        norm_label = "LOC"
                    else:
                        norm_label = "MISC"

                    entities.append(ExtractedEntity(
                        text=item["word"].strip(),
                        label=norm_label,
                        start=item["start"],
                        end=item["end"],
                        confidence=round(float(item["score"]), 4),
                    ))
                return entities
            except Exception as e:
                logger.debug(f"Transformer NER failed on input: {e}")

        # Fast rule-based / regex NER fallback (capitalized sequences, known org suffixes, places)
        return self._extract_regex_entities(text)

    def _extract_regex_entities(self, text: str) -> List[ExtractedEntity]:
        """Deterministic pattern-based entity extractor for rapid local testing."""
        entities: List[ExtractedEntity] = []

        org_indicators = {"Inc", "Corp", "Co", "Ltd", "LLC", "Group", "Agency", "Bank", "Department", "University", "Google", "Microsoft", "Apple", "Amazon", "Facebook", "Meta", "Netflix", "NASA", "FBI", "CIA", "WHO", "CDC", "NFL", "NBA", "FIFA"}
        loc_indicators = {"US", "USA", "UK", "China", "Russia", "Japan", "Germany", "France", "India", "New York", "Washington", "London", "Paris", "Tokyo", "California", "Texas", "Florida", "Beijing", "Berlin", "Europe", "Asia", "America"}

        # Multi-word or single-word capitalized terms
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b"
        for m in re.finditer(pattern, text):
            token = m.group(1).strip()
            if len(token) < 2 or token.lower() in {"the", "a", "an", "in", "on", "at", "by", "for", "with", "about", "new", "first", "last"}:
                continue

            # Determine label
            words = token.split()
            if any(w in org_indicators for w in words) or token in org_indicators:
                label = "ORG"
            elif any(w in loc_indicators for w in words) or token in loc_indicators:
                label = "LOC"
            elif len(words) >= 2 and all(w[0].isupper() for w in words):
                label = "PER"
            else:
                label = "MISC"

            entities.append(ExtractedEntity(
                text=token,
                label=label,
                start=m.start(),
                end=m.end(),
                confidence=0.85,
            ))

        # Deduplicate overlapping spans
        deduped: List[ExtractedEntity] = []
        seen_texts: Set[str] = set()
        for ent in entities:
            if ent.text.lower() not in seen_texts:
                seen_texts.add(ent.text.lower())
                deduped.append(ent)

        return deduped

    def resolve_mind_entities(
        self,
        extracted_entities: List[ExtractedEntity],
        mind_entities_raw: Optional[List[Dict[str, Any]]],
    ) -> List[ExtractedEntity]:
        """
        Resolve model extracted entities against MIND Wikidata annotations.
        Attaches wikidata_id if entity matches a MIND entity surface form.
        """
        if not mind_entities_raw:
            return extracted_entities

        mind_lookup: Dict[str, str] = {}
        for me in mind_entities_raw:
            label = me.get("Label", "")
            wikidata_id = me.get("WikidataId", "")
            if label and wikidata_id:
                mind_lookup[label.lower().strip()] = wikidata_id

        for ent in extracted_entities:
            ent_key = ent.text.lower().strip()
            if ent_key in mind_lookup:
                ent.wikidata_id = mind_lookup[ent_key]
            else:
                # Substring match
                for m_lbl, wid in mind_lookup.items():
                    if m_lbl in ent_key or ent_key in m_lbl:
                        ent.wikidata_id = wid
                        break

        return extracted_entities

    def evaluate_against_mind(
        self,
        articles: List[Any],
    ) -> Dict[str, Any]:
        """
        Evaluate entity detection alignment against MIND dataset Wikidata entities.
        Computes precision, recall, and F1 at the entity surface-form level.
        """
        tp, fp, fn = 0, 0, 0
        total_model_entities = 0
        total_mind_entities = 0

        for art in articles:
            text = f"{art.title} {art.abstract}"
            model_ents = self.extract_entities(text)
            model_surface = {e.text.lower().strip() for e in model_ents}
            total_model_entities += len(model_surface)

            # Extract MIND entities from title_entities + abstract_entities
            mind_surface: Set[str] = set()
            for ent_list in [getattr(art, "title_entities", []), getattr(art, "abstract_entities", [])]:
                if isinstance(ent_list, list):
                    for me in ent_list:
                        lbl = me.get("Label", "") if isinstance(me, dict) else ""
                        if lbl:
                            mind_surface.add(lbl.lower().strip())

            total_mind_entities += len(mind_surface)

            # Calculate overlap
            matched = model_surface & mind_surface
            tp += len(matched)
            fp += len(model_surface - mind_surface)
            fn += len(mind_surface - model_surface)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_model_extracted": total_model_entities,
            "total_mind_annotated": total_mind_entities,
            "num_articles_evaluated": len(articles),
        }
