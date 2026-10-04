"""Image-only API extraction contract; source identity must never be inferred from metadata."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

VISUAL_PROMPT = '''Extract an image-grounded scene hypergraph from this image alone.
Return one JSON object, no markdown. First describe the visible scene, then list
its main visible objects, then derive visible relations from that description.
Do not use external knowledge, tools, or web search. Do not guess a landmark,
city, country, historical date, builder, or proper identity from appearance.
Readable text may be transcribed exactly; otherwise omit it. Do not expand
abbreviations or complete unreadable signs. Exclude hypotheses from positive facts.
Do not infer a named monument from the camera viewpoint, or assume a bridge is
visible merely because a water scene might have been photographed from one.
Object boxes are approximate normalized [x0,y0,x1,y1] coordinates in [0,1].
Confidence scores use (0,10] and are model self-assessments, NOT factual truth.
Use English. Limit to 6 main objects and 3 relations; omit uncertain relations.
Schema (all keys required, no additional keys):
{"image_only":true,
 "scene_description":"detailed literal scene description, at most 1200 characters",
 "objects":[{"id":"o1", "name":"generic visible object name",
   "entity_type":"OBJECT", "description":"at most 220 characters",
   "bbox":[0.0,0.0,1.0,1.0], "confidence":9.0}],
 "relations":[{"statement":"at most 300 characters describing a visible relation",
   "object_ids":["o1","o2"], "confidence":9.0}],
 "visible_text":["exactly readable text; at most 6 entries"],
 "uncertainties":["at most 6 brief limitations"]}
Object entity_type must be OBJECT, LOCATION, PERSON, or TEXT.
Use local IDs o1,o2,...; never put filenames or dataset IDs in the output.
Each relation must reference existing object IDs; every fact will be anchored
to its actual image downstream. A single collective object (boats, people) is
allowed: avoid inventing precise counts for distant or occluded objects.
'''


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class VisualObject(StrictModel):
    id: str = Field(pattern=r'^o[1-6]$')
    name: str = Field(min_length=1, max_length=100)
    entity_type: Literal['OBJECT', 'LOCATION', 'PERSON', 'TEXT']
    description: str = Field(min_length=1, max_length=220)
    bbox: list[float] | None = Field(min_length=4, max_length=4)
    confidence: float = Field(gt=0, le=10)

    @model_validator(mode='after')
    def valid_box(self):
        if self.bbox is None:
            return self  # Whole-image grounding; no region localization claim.
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('invalid normalized visual box')
        return self


class VisualRelation(StrictModel):
    statement: str = Field(min_length=1, max_length=300)
    object_ids: list[str] = Field(min_length=1, max_length=6)
    confidence: float = Field(gt=0, le=10)


class VisualScene(StrictModel):
    image_only: Literal[True]
    scene_description: str = Field(min_length=20, max_length=1200)
    objects: list[VisualObject] = Field(min_length=1, max_length=6)
    relations: list[VisualRelation] = Field(max_length=3)
    visible_text: list[str] = Field(max_length=6)
    uncertainties: list[str] = Field(max_length=6)

    @model_validator(mode='after')
    def valid_references(self):
        ids = [obj.id for obj in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate object IDs')
        names = [obj.name.casefold().strip() for obj in self.objects]
        if len(set(names)) != len(names):
            raise ValueError('duplicate object names')
        for relation in self.relations:
            if not set(relation.object_ids) <= set(ids) or len(set(relation.object_ids)) != len(relation.object_ids):
                raise ValueError('invalid visual relation references')
        return self


def parse_scene(record: dict) -> VisualScene:
    if record.get('status') != 'complete' or record.get('finish_reason') != 'stop':
        raise ValueError('incomplete, truncated, or refused visual API output')
    return VisualScene.model_validate_json(record.get('output') or '')


def apply_source_review(record: dict, review: dict | None) -> VisualScene:
    """Only explicit, hash-pinned review can transform raw output; never silently repair boxes."""
    if not review:
        return parse_scene(record)
    import hashlib
    import json
    if record.get('status') != 'complete' or record.get('finish_reason') != 'stop':
        raise ValueError('cannot review incomplete visual API output')
    if hashlib.sha256(record['output'].encode()).hexdigest() != review['raw_output_sha256']:
        raise ValueError('source review hash mismatch')
    output = json.loads(record['output'])
    allowed = {'raw_output_sha256', 'reviewer', 'human_reviewed', 'reason', 'excluded_object_ids',
               'scene_description', 'object_descriptions', 'object_names', 'bbox_scale',
               'visible_text', 'relation_statements', 'field_aliases', 'attest_image_only_request',
               'omit_box_object_ids'}
    if not set(review) <= allowed or review.get('reviewer') != 'agent_image_source_check' or review.get('human_reviewed') is not False:
        raise ValueError('invalid source review provenance')
    if 'field_aliases' in review:
        if review['field_aliases'] != {'image_description': 'scene_description'} or 'image_description' not in output or 'scene_description' in output:
            raise ValueError('unsupported reviewed field alias')
        output['scene_description'] = output.pop('image_description')
    if review.get('attest_image_only_request') is True:
        if 'image_only' in output:
            raise ValueError('cannot override model-provided modality flag')
        output['image_only'] = True  # Attested request provenance, not a fabricated model statement.
    removed = set(review.get('excluded_object_ids', []))
    if not removed <= {o['id'] for o in output['objects']}:
        raise ValueError('review excludes an unknown object')
    output['objects'] = [o for o in output['objects'] if o['id'] not in removed]
    output['relations'] = [r for r in output['relations'] if not removed & set(r['object_ids'])]
    omitted_boxes = set(review.get('omit_box_object_ids', []))
    if not omitted_boxes <= {o['id'] for o in output['objects']}:
        raise ValueError('review omits a box for an unknown object')
    for obj in output['objects']:
        if obj['id'] in omitted_boxes:
            obj['bbox'] = None
    if 'bbox_scale' in review:
        # Explicit source-inspected scale conversion only, never guess mixed conventions.
        scale = review['bbox_scale']
        if scale != 1000 or any(len(o['bbox']) != 4 or max(o['bbox']) <= 1
                               or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                          and 0 <= x <= scale for x in o['bbox']) for o in output['objects'] if o['bbox'] is not None):
            raise ValueError('ambiguous or unsupported explicit box scale')
        for obj in output['objects']:
            if obj['bbox'] is not None:
                obj['bbox'] = [float(x) / scale for x in obj['bbox']]
    if 'scene_description' in review:
        output['scene_description'] = review['scene_description']
    descriptions = review.get('object_descriptions', {})
    names = review.get('object_names', {})
    if not (set(descriptions) | set(names)) <= {o['id'] for o in output['objects']}:
        raise ValueError('review modifies an unknown object')
    for obj in output['objects']:
        if obj['id'] in descriptions:
            obj['description'] = descriptions[obj['id']]
        if obj['id'] in names:
            obj['name'] = names[obj['id']]
    if 'visible_text' in review:
        output['visible_text'] = review['visible_text']
    for index, statement in review.get('relation_statements', {}).items():
        if not index.isdigit() or not 0 <= int(index) < len(output['relations']):
            raise ValueError('review modifies an unknown relation')
        output['relations'][int(index)]['statement'] = statement
    return VisualScene.model_validate(output)


def normalized_label(name: str) -> str:
    import re
    import unicodedata
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', name).strip('"').strip()).casefold()
