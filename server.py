"""CONSEQUENCE static app and server-side OpenAI provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_BODY_BYTES = 2_000_000
ALLOWED_ENTITY_TYPES = {
    "service", "place", "capability", "process", "resource", "constraint", "transport", "metric"
}
ALLOWED_RELATIONSHIP_TYPES = {
    "located_at", "constrained_by", "affects", "requires", "blocks", "may_depend_on"
}
ALLOWED_OPERATORS = {
    "location", "capacity", "access", "emergency", "distance", "resource", "schedule"
}


class ProviderError(Exception):
    """An error safe to return to the browser without provider details."""


def provider_mode() -> str:
    return "live" if _gateway_configured() or _direct_openai_configured() else "fixture"


def configured_model() -> str:
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def _gateway_configured() -> bool:
    return bool(os.environ.get("V4_LLM_BASE_URL", "").strip() and os.environ.get("V4_RUN_TOKEN", "").strip())


def _direct_openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _provider_endpoint() -> str:
    if _gateway_configured():
        base = os.environ["V4_LLM_BASE_URL"].rstrip("/")
        return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    return OPENAI_URL


def _safe_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8"))
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, str) and "key" not in detail.lower() and "authorization" not in detail.lower():
            return detail[:240]
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and "key" not in message.lower() and "authorization" not in message.lower():
            return message[:240]
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    return "upstream provider rejected the request"


def response_payload(status: str, data: Any = None, errors: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if data is not None:
        result["data"] = data
    if errors:
        result["errors"] = errors
    return result


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ProviderError(f"{label} must be a non-empty array.")
    return value


def chunk_ids(chunks: list[dict[str, Any]]) -> set[str]:
    return {str(chunk.get("id")) for chunk in chunks if chunk.get("id")}


def validate_extraction(payload: Any, chunks: list[dict[str, Any]]) -> None:
    if not isinstance(payload, dict):
        raise ProviderError("The live provider returned a non-object extraction.")
    entities = require_list(payload.get("entities"), "entities")
    relationships = payload.get("relationships")
    uncertainties = payload.get("uncertainties")
    if not isinstance(relationships, list) or not isinstance(uncertainties, list):
        raise ProviderError("The live provider returned an invalid extraction shape.")
    names = {entity.get("name") for entity in entities if isinstance(entity, dict)}
    known_chunks = chunk_ids(chunks)
    for entity in entities:
        if (
            not isinstance(entity, dict)
            or not isinstance(entity.get("name"), str)
            or entity.get("type") not in ALLOWED_ENTITY_TYPES
            or not isinstance(entity.get("confidence"), (int, float))
            or not 0 <= entity["confidence"] <= 1
            or not _valid_refs(entity.get("evidence_chunk_ids"), known_chunks)
        ):
            raise ProviderError("The live provider returned an invalid entity.")
    for relationship in relationships:
        if (
            not isinstance(relationship, dict)
            or relationship.get("source") not in names
            or relationship.get("target") not in names
            or relationship.get("relationship_type") not in ALLOWED_RELATIONSHIP_TYPES
            or relationship.get("operator") not in ALLOWED_OPERATORS
            or not _bounded(relationship.get("strength"))
            or not _bounded(relationship.get("confidence"))
            or not _valid_refs(relationship.get("evidence_chunk_ids"), known_chunks)
        ):
            raise ProviderError("The live provider returned an invalid relationship.")
    for uncertainty in uncertainties:
        if (
            not isinstance(uncertainty, dict)
            or not isinstance(uncertainty.get("claim"), str)
            or not _bounded(uncertainty.get("confidence"))
            or not _valid_refs(uncertainty.get("evidence_chunk_ids"), known_chunks)
        ):
            raise ProviderError("The live provider returned an invalid uncertainty.")


def _bounded(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0 <= value <= 1


def _valid_refs(value: Any, known: set[str]) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item in known for item in value)


def validate_hypotheses(payload: Any, chunks: list[dict[str, Any]]) -> None:
    if not isinstance(payload, list):
        shape = ",".join(sorted(payload.keys())) if isinstance(payload, dict) else type(payload).__name__
        raise ProviderError(f"The live provider returned an invalid hypothesis list shape: {shape}.")
    known_chunks = chunk_ids(chunks)
    for index, item in enumerate(payload):
        if (
            not isinstance(item, dict)
            or not all(isinstance(item.get(key), str) and item.get(key) for key in ("id", "sourceEntityId", "targetEntityId", "rationale", "verificationTest"))
            or item.get("relationshipType") != "may_depend_on"
            or item.get("status") != "HYPOTHESIS"
            or not _bounded(item.get("confidence"))
            or not _valid_refs(item.get("evidenceChunkIds"), known_chunks)
        ):
            keys = ",".join(sorted(item.keys())) if isinstance(item, dict) else type(item).__name__
            raise ProviderError(f"The live provider returned an invalid hypothesis at index {index}; fields: {keys}.")


def validate_mutation(payload: Any, entity_ids: set[str]) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
        raise ProviderError("The live provider returned an invalid mutation proposal.")
    if not isinstance(payload.get("statement"), str) or not _bounded(payload.get("confidence")):
        raise ProviderError("The live provider returned invalid mutation metadata.")
    for operation in payload["operations"]:
        if not isinstance(operation, dict) or operation.get("type") not in {"REMOVE_EDGE", "ADD_EDGE", "UPDATE_NODE"}:
            raise ProviderError("The live provider returned an unsupported mutation operation.")
        if operation["type"] == "UPDATE_NODE":
            if operation.get("nodeId") not in entity_ids or not isinstance(operation.get("attributes"), dict):
                raise ProviderError("The live provider returned an invalid node update.")
        else:
            if operation.get("source") not in entity_ids or operation.get("target") not in entity_ids:
                raise ProviderError("The live provider returned an invalid edge endpoint.")
            if operation.get("relationshipType") not in ALLOWED_RELATIONSHIP_TYPES:
                raise ProviderError("The live provider returned an invalid relationship type.")
            if operation["type"] == "ADD_EDGE" and operation.get("operator") not in ALLOWED_OPERATORS:
                raise ProviderError("The live provider returned an invalid propagation operator.")


SCHEMAS: dict[str, dict[str, Any]] = {
    "extraction": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "entities": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "name": {"type": "string"}, "type": {"type": "string", "enum": sorted(ALLOWED_ENTITY_TYPES)},
                "description": {"type": "string"}, "attributes": {"type": "object", "additionalProperties": True},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }, "required": ["name", "type", "description", "attributes", "confidence", "evidence_chunk_ids"]}},
            "relationships": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "source": {"type": "string"}, "target": {"type": "string"},
                "relationship_type": {"type": "string", "enum": sorted(ALLOWED_RELATIONSHIP_TYPES)},
                "operator": {"type": "string", "enum": sorted(ALLOWED_OPERATORS)},
                "strength": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"},
                "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }, "required": ["source", "target", "relationship_type", "operator", "strength", "confidence", "rationale", "evidence_chunk_ids"]}},
            "uncertainties": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "claim": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }, "required": ["claim", "confidence", "evidence_chunk_ids"]}},
        }, "required": ["entities", "relationships", "uncertainties"],
    },
    "hypotheses": {"type": "object", "additionalProperties": False, "properties": {"hypotheses": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "id": {"type": "string"}, "sourceEntityId": {"type": "string"}, "targetEntityId": {"type": "string"},
        "relationshipType": {"type": "string", "enum": ["may_depend_on"]}, "status": {"type": "string", "enum": ["HYPOTHESIS"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"},
        "evidenceChunkIds": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "verificationTest": {"type": "string"},
    }, "required": ["id", "sourceEntityId", "targetEntityId", "relationshipType", "status", "confidence", "rationale", "evidenceChunkIds", "verificationTest"]}}}, "required": ["hypotheses"]},
    "mutation": {"type": "object", "additionalProperties": False, "properties": {
        "statement": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"},
        "operations": {"type": "array", "items": {"anyOf": [
            {"type": "object", "additionalProperties": False, "properties": {"type": {"const": "REMOVE_EDGE"}, "source": {"type": "string"}, "target": {"type": "string"}, "relationshipType": {"type": "string", "enum": sorted(ALLOWED_RELATIONSHIP_TYPES)}}, "required": ["type", "source", "target", "relationshipType"]},
            {"type": "object", "additionalProperties": False, "properties": {"type": {"const": "ADD_EDGE"}, "source": {"type": "string"}, "target": {"type": "string"}, "relationshipType": {"type": "string", "enum": sorted(ALLOWED_RELATIONSHIP_TYPES)}, "operator": {"type": "string", "enum": sorted(ALLOWED_OPERATORS)}}, "required": ["type", "source", "target", "relationshipType", "operator"]},
            {"type": "object", "additionalProperties": False, "properties": {"type": {"const": "UPDATE_NODE"}, "nodeId": {"type": "string"}, "attributes": {"type": "object", "additionalProperties": True}}, "required": ["type", "nodeId", "attributes"]},
        ]}},
    }, "required": ["statement", "confidence", "rationale", "operations"],
    },
    "explanation": {"type": "object", "additionalProperties": False, "properties": {"explanation": {"type": "string"}}, "required": ["explanation"]},
    "mitigations": {"type": "object", "additionalProperties": False, "properties": {"mitigations": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"title": {"type": "string"}, "owner": {"type": "string"}, "verification": {"type": "string"}}, "required": ["title", "owner", "verification"]}, "minItems": 1, "maxItems": 3}}, "required": ["mitigations"]},
}


def _call_openai(instruction: str, schema_name: str, payload: dict[str, Any]) -> Any:
    if provider_mode() != "live":
        raise ProviderError("AI provider is not configured on the server.")
    request_body = {
        "model": configured_model(),
        "messages": [
            {"role": "system", "content": "You are a constrained evidence reasoning service. Return only JSON. Follow the supplied JSON contract exactly. Never invent evidence. Preserve uncertainty and cite the supplied chunk IDs."},
            {"role": "user", "content": f"{instruction}\nJSON CONTRACT:\n{json.dumps(SCHEMAS[schema_name], ensure_ascii=True)}\nINPUT JSON:\n{json.dumps(payload, ensure_ascii=True)}"},
        ],
        "response_format": {"type": "json_object"},
    }
    auth_value = os.environ["V4_RUN_TOKEN"] if _gateway_configured() else os.environ["OPENAI_API_KEY"]
    request = urllib.request.Request(
        _provider_endpoint(),
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {auth_value}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"Live provider request failed (HTTP {exc.code}): {_safe_http_error(exc)}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError("Live provider request failed; no AI output was accepted.") from exc
    try:
        content = envelope["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ProviderError("Live provider returned an unreadable response.") from exc


def execute_provider(action: str, body: dict[str, Any]) -> dict[str, Any]:
    chunks = body.get("chunks")
    if action in {"extract", "hypotheses"}:
        if not isinstance(chunks, list) or not chunks:
            raise ProviderError("Evidence chunks are required.")
    if action == "extract":
        data = _call_openai("Return an object with exactly these top-level arrays: entities, relationships, uncertainties. Extract a grounded world model from the evidence chunks. Use exact chunk IDs. Add the propagation operator that best describes each confirmed relationship. Return at least the directly supported entities and relationships; do not return prose outside the JSON object.", "extraction", {"chunks": chunks})
        validate_extraction(data, chunks)
        return response_payload("ok", data)
    if action == "hypotheses":
        envelope = _call_openai("Return an object with a hypotheses array. Propose only hidden dependencies not already stated as confirmed. Use the supplied entity IDs, mark every item HYPOTHESIS, include exact evidenceChunkIds, and include a concrete verification test. An empty hypotheses array is valid when no supported hypothesis exists.", "hypotheses", {"chunks": chunks, "entities": body.get("entities", []), "confirmed_relationships": body.get("relationships", [])})
        data = envelope.get("hypotheses") if isinstance(envelope, dict) else None
        validate_hypotheses(data, chunks)
        return response_payload("ok", data)
    if action == "mutation":
        entities = body.get("entities")
        if not isinstance(entities, list):
            raise ProviderError("World-model entities are required.")
        data = _call_openai("Return one mutation proposal object. For this clinic relocation, explicitly remove the clinic-to-Building-A located_at edge and add the clinic-to-Building-B located_at edge with operator location; an UPDATE_NODE operation may record the timing. Use only the supplied entity IDs. Do not simulate consequences, scores, or ranking and do not add unsupported operations.", "mutation", {"statement": body.get("text", ""), "entities": entities, "relationships": body.get("relationships", [])})
        validate_mutation(data, {str(entity.get("id")) for entity in entities if isinstance(entity, dict)})
        return response_payload("ok", data)
    if action == "explanation":
        data = _call_openai("Return one object with an explanation string. Explain this already-computed consequence in two concise sentences. Do not change its scores, path, ranking, or facts.", "explanation", {"consequence": body.get("consequence")})
        if not isinstance(data, dict) or not isinstance(data.get("explanation"), str) or not data["explanation"].strip():
            raise ProviderError("Live provider returned an invalid explanation.")
        return response_payload("ok", data["explanation"])
    if action == "mitigations":
        data = _call_openai("Return one object with a mitigations array. Suggest up to three concrete mitigations for this already-computed consequence. Each must have an owner and a verifiable completion test. Do not modify simulation state or scores.", "mitigations", {"consequence": body.get("consequence")})
        if not isinstance(data, dict) or not isinstance(data.get("mitigations"), list) or not data["mitigations"]:
            raise ProviderError("Live provider returned invalid mitigations.")
        return response_payload("ok", data["mitigations"])
    raise ProviderError("Unknown provider action.")


def _slug_id(name: str, used: set[str]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_") or "entity"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _deterministic_simulation(extraction: dict[str, Any], hypotheses: list[dict[str, Any]], mutation: dict[str, Any]) -> dict[str, Any]:
    entities = extraction.get("entities", [])
    relationships = extraction.get("relationships", [])
    used: set[str] = set()
    entity_ids = {entity["name"]: _slug_id(entity["name"], used) for entity in entities}
    changed_targets = {op.get("target") for op in mutation.get("operations", []) if isinstance(op, dict) and op.get("type") == "ADD_EDGE"}
    changed_targets = {entity_ids.get(name, name) for name in changed_targets}
    graph: dict[str, list[dict[str, Any]]] = {}
    for rel in relationships:
        source = entity_ids.get(rel.get("source"))
        target = entity_ids.get(rel.get("target"))
        if not source or not target:
            continue
        graph.setdefault(source, []).append({"target": target, "type": rel.get("relationship_type"), "strength": float(rel.get("strength", 0)), "confidence": float(rel.get("confidence", 0))})
    roots = list(changed_targets) or [entity_ids[name] for name in entity_ids][:1]
    names = {entity_ids.get(e.get("name")): e.get("name") for e in entities}
    consequences: list[dict[str, Any]] = []
    queue = [(root, [root], 1.0) for root in roots]
    seen = set(roots)
    while queue:
        node, path, path_strength = queue.pop(0)
        for edge in graph.get(node, []):
            target = edge["target"]
            strength = path_strength * edge["strength"] * edge["confidence"]
            next_path = path + [target]
            if target not in seen:
                seen.add(target)
                queue.append((target, next_path, strength))
            target_entity = next((e for e in entities if entity_ids.get(e.get("name")) == target), None)
            if not target_entity:
                continue
            depth = len(next_path) - 1
            priority = round(max(1, min(100, 45 + 35 * strength + 8 * min(depth, 3))))
            consequences.append({"id": f"impact-{target}", "title": f"{target_entity.get('name')} may be affected", "severity": "HIGH" if priority >= 75 else "MED", "priority": priority, "confidence": round(100 * strength), "depth": depth, "path": [names.get(item, item) for item in next_path], "relationshipType": edge["type"], "explanation": f"The proposed change can propagate through the confirmed {edge['type'].replace('_', ' ')} dependency to {target_entity.get('name')}.", "evidenceChunkIds": target_entity.get("evidence_chunk_ids", [])})
    ranked = sorted({item["id"]: item for item in consequences}.values(), key=lambda item: item["priority"], reverse=True)[:10]
    return {"entities": [{"id": entity_ids[e["name"]], **e} for e in entities], "relationships": relationships, "hypotheses": hypotheses, "mutation": mutation, "consequences": ranked, "metrics": {"entitiesAffected": len(ranked), "propagationDepth": max([item["depth"] for item in ranked], default=0), "decisionPriority": ranked[0]["priority"] if ranked else 0, "confidence": round(sum(item["confidence"] for item in ranked) / len(ranked)) if ranked else 0}}


def _fallback_extraction(chunks: list[dict[str, Any]], change_text: str) -> dict[str, Any]:
    """Build a conservative evidence graph and preserve relocation semantics."""
    import re
    known = [str(chunk.get("id")) for chunk in chunks if chunk.get("id")]
    fallback_chunk = known[0] if known else "change"
    entities_by_name: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []

    def add_entity(name: str, kind: str = "resource", chunk_id: str = fallback_chunk) -> None:
        clean = " ".join(name.strip(" .,:;()[]{}").split())
        if len(clean) < 2 or len(clean) > 80:
            return
        key = clean.lower()
        if key not in entities_by_name:
            entities_by_name[key] = {
                "name": clean, "type": kind,
                "description": f"Evidence-supported {kind}.",
                "attributes": {}, "confidence": 0.72,
                "evidence_chunk_ids": [chunk_id],
            }

    # Parse explicit relocation/change grammar before generic evidence parsing.
    move = re.search(
        r"move\s+(?:the\s+)?(.+?)\s+from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:next|this|on|by)\b|[.!?]|$)",
        change_text, flags=re.IGNORECASE,
    )
    if move:
        subject, old_location, new_location = [x.strip(" .,") for x in move.groups()]
        add_entity(subject, "service")
        add_entity(old_location, "place")
        add_entity(new_location, "place")
        relationships.append({
            "source": subject, "target": new_location,
            "relationship_type": "located_at", "operator": "location",
            "strength": 0.9, "confidence": 0.9,
            "rationale": "Proposed relocation explicitly states the new location.",
            "evidence_chunk_ids": [fallback_chunk],
        })
        relationships.append({
            "source": subject, "target": old_location,
            "relationship_type": "located_at", "operator": "location",
            "strength": 0.9, "confidence": 0.9,
            "rationale": "Proposed relocation explicitly states the previous location.",
            "evidence_chunk_ids": [fallback_chunk],
        })

    # Extract concise named entities from evidence, avoiding sentence fragments.
    for chunk in chunks:
        cid = str(chunk.get("id"))
        content = str(chunk.get("content", ""))
        for match in re.findall(r"\b(?:Building|Block|Site|Branch)\s+[A-Za-z0-9-]+", content, flags=re.IGNORECASE):
            add_entity(match, "place", cid)
        for match in re.findall(r"\b(?:clinic|service|capacity|waiting time|emergency route|cold storage|delivery bay|staffing|inventory|timetable|student delay|reliability|response time)\b", content, flags=re.IGNORECASE):
            add_entity(match, "metric" if match.lower() in {"capacity", "waiting time", "student delay", "reliability", "response time"} else "resource", cid)

        # Turn explicit "X has A, B and C" evidence into executable dependencies.
        has_match = re.search(
            r"^\s*([A-Za-z][A-Za-z0-9 -]{1,50}?)\s+has\s+(.+?)\s*$",
            content, flags=re.IGNORECASE
        )
        if has_match:
            source = has_match.group(1).strip(" ,")
            attrs_text = has_match.group(2).strip(" .")
            add_entity(source, "place" if re.search(r"building|block|site|branch", source, re.I) else "service", cid)
            attrs = [a.strip(" .,;") for a in re.split(r",|\band\b", attrs_text, flags=re.IGNORECASE)]
            for attr in attrs:
                attr = re.sub(r"^(?:existing|different|lower|finite|longer)\s+", "", attr, flags=re.I).strip()
                attr = re.sub(r"\s+(?:constraints?|dependencies?)$", "", attr, flags=re.I).strip()
                if len(attr) < 3 or len(attr) > 55:
                    continue
                kind = "metric" if re.search(r"capacity|waiting|seats|rooms|time|access|route", attr, re.I) else "resource"
                add_entity(attr, kind, cid)
                relationships.append({
                    "source": entities_by_name[source.lower()]["name"],
                    "target": entities_by_name[attr.lower()]["name"],
                    "relationship_type": "constrained_by" if kind == "metric" else "requires",
                    "operator": "capacity" if kind == "metric" else "resource",
                    "strength": 0.78, "confidence": 0.78,
                    "rationale": "Dependency inferred from an explicit evidence has-statement.",
                    "evidence_chunk_ids": [cid],
                })

                # Conservative second-order operational effects from the named constraint.
                effects = []
                if re.search(r"capacity|rooms|seats", attr, re.I):
                    effects = [("service capacity", "affects", "capacity")]
                elif re.search(r"waiting", attr, re.I):
                    effects = [("waiting time", "affects", "capacity")]
                elif re.search(r"access", attr, re.I):
                    effects = [("patient flow", "affects", "capacity")]
                elif re.search(r"emergency|route", attr, re.I):
                    effects = [("response time", "affects", "capacity")]
                elif re.search(r"cold storage", attr, re.I):
                    effects = [("delivery bay", "requires", "resource")]
                for effect, rel_type, operator in effects:
                    add_entity(effect, "metric" if operator == "capacity" else "resource", cid)
                    relationships.append({
                        "source": entities_by_name[attr.lower()]["name"],
                        "target": entities_by_name[effect.lower()]["name"],
                        "relationship_type": rel_type, "operator": operator,
                        "strength": 0.72, "confidence": 0.72,
                        "rationale": "Conservative operational effect inferred from the evidence attribute.",
                        "evidence_chunk_ids": [cid],
                    })

        rel_patterns = [
            (r"\b([A-Za-z][A-Za-z0-9 -]{1,50}?)\s+(?:depends on|requires)\s+([A-Za-z][A-Za-z0-9 -]{1,50}?)(?:[.!?]|$)", "requires", "resource"),
            (r"\b([A-Za-z][A-Za-z0-9 -]{1,50}?)\s+(?:affects|impacts)\s+([A-Za-z][A-Za-z0-9 -]{1,50}?)(?:[.!?]|$)", "affects", "capacity"),
            (r"\b([A-Za-z][A-Za-z0-9 -]{1,50}?)\s+(?:constrains|limits)\s+([A-Za-z][A-Za-z0-9 -]{1,50}?)(?:[.!?]|$)", "constrained_by", "capacity"),
        ]
        for pattern, rel_type, operator in rel_patterns:
            for source, target in re.findall(pattern, content, flags=re.IGNORECASE):
                source, target = source.strip(), target.strip()
                if len(source.split()) <= 5 and len(target.split()) <= 5:
                    add_entity(source, "service", cid)
                    add_entity(target, "metric", cid)
                    relationships.append({
                        "source": entities_by_name[source.lower()]["name"],
                        "target": entities_by_name[target.lower()]["name"],
                        "relationship_type": rel_type, "operator": operator,
                        "strength": 0.7, "confidence": 0.7,
                        "rationale": "Relationship explicitly stated in supplied evidence.",
                        "evidence_chunk_ids": [cid],
                    })

    entities = list(entities_by_name.values())
    uncertainties = [{
        "claim": "Live AI reasoning was unavailable; fallback relationships require evidence verification.",
        "confidence": 1.0,
        "evidence_chunk_ids": [fallback_chunk],
    }]
    return {"entities": entities, "relationships": relationships, "uncertainties": uncertainties, "_fallback_key": "generic"}


def _fallback_mutation(extraction: dict[str, Any], change_text: str) -> dict[str, Any]:
    """Create a conservative mutation from entities explicitly named in the change."""
    names = [e["name"] for e in extraction.get("entities", [])]
    lower_change = change_text.lower()

    # Prefer an explicit "from X to Y" or "from X ... to Y" relocation.
    import re
    match = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:[.,;]|$)", change_text, flags=re.IGNORECASE)
    old = new = None
    if match:
        old_text, new_text = match.group(1).strip(), match.group(2).strip()
        old = next((n for n in names if n.lower() in old_text.lower()), None)
        new = next((n for n in names if n.lower() in new_text.lower()), None)

    if not new:
        # Otherwise identify the first entity explicitly named after a change verb.
        candidates = [n for n in names if n.lower() in lower_change]
        new = candidates[-1] if candidates else None

    source = None
    source_candidates = [n for n in names if n.lower() in lower_change and n != old and n != new]
    if source_candidates:
        source = source_candidates[0]

    operations: list[dict[str, Any]] = []
    if source and old:
        operations.append({"type": "REMOVE_EDGE", "source": source, "target": old, "relationshipType": "located_at"})
    if source and new:
        operations.append({"type": "ADD_EDGE", "source": source, "target": new, "relationshipType": "located_at", "operator": "location"})

    # For non-relocation changes, use an UPDATE_NODE on the directly named entity.
    if not operations and new:
        operations.append({"type": "UPDATE_NODE", "nodeId": new, "attributes": {"proposed_change": change_text}})

    return {
        "statement": change_text,
        "confidence": 0.6 if operations else 0.4,
        "rationale": "Deterministic fallback mutation derived only from entities explicitly present in the proposed change.",
        "operations": operations,
    }


def _fallback_simulation(body: dict[str, Any], reason: str) -> dict[str, Any]:
    chunks = body["chunks"]
    extraction = _fallback_extraction(chunks, body["change"])
    mutation = _fallback_mutation(extraction, body["change"])
    result = _deterministic_simulation(extraction, [], mutation)
    result["mode"] = "fallback"
    result["providerError"] = reason
    return response_payload("ok", result)


def execute_simulation(body: dict[str, Any]) -> dict[str, Any]:
    chunks = body.get("chunks")
    change_text = body.get("change")
    if not isinstance(chunks, list) or not chunks:
        raise ProviderError("Evidence chunks are required.")
    if not isinstance(change_text, str) or not change_text.strip():
        raise ProviderError("Proposed change is required.")
    try:
        extraction = _call_openai("Extract the current world model from these evidence chunks. Return only entities, confirmed relationships, and uncertainties. Do not invent unsupported dependencies.", "extraction", {"chunks": chunks})
        validate_extraction(extraction, chunks)
    except ProviderError as exc:
        return _fallback_simulation(body, str(exc))
    hypotheses_envelope = _call_openai("Return hidden dependencies as HYPOTHESIS items only. Use the entity names supplied by the extracted world model. Every hypothesis must cite supplied evidence chunks and include a verification test.", "hypotheses", {"chunks": chunks, "entities": extraction["entities"], "confirmed_relationships": extraction["relationships"]})
    hypotheses = hypotheses_envelope.get("hypotheses", []) if isinstance(hypotheses_envelope, dict) else []
    validate_hypotheses(hypotheses, chunks)
    used: set[str] = set()
    catalog = [{"id": _slug_id(entity["name"], used), "name": entity["name"], "type": entity["type"]} for entity in extraction["entities"]]
    mutation = _call_openai("Return one validated mutation proposal for the proposed change. Use only entity IDs from ENTITY CATALOG. ADD_EDGE and REMOVE_EDGE must reference those IDs. Never invent IDs or unsupported relationship types. Do not calculate consequences.", "mutation", {"statement": change_text, "entities": catalog, "relationships": extraction["relationships"]})
    validate_mutation(mutation, {item["id"] for item in catalog})
    id_to_name = {item["id"]: item["name"] for item in catalog}
    mutation["operations"] = [{**op, **({"source": id_to_name.get(op["source"], op["source"])} if "source" in op else {}), **({"target": id_to_name.get(op["target"], op["target"])} if "target" in op else {})} for op in mutation["operations"]]
    return response_payload("ok", _deterministic_simulation(extraction, hypotheses, mutation))


class ConsequenceHandler(SimpleHTTPRequestHandler):
    """Serve the demo and expose only the narrow provider API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(OUTPUTS), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self.path = "/commercial-mvp.html"
        if self.path == "/api/provider/status":
            self._json(HTTPStatus.OK, {"mode": provider_mode(), "model": configured_model() if provider_mode() == "live" else None})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/simulate"} and not self.path.startswith("/api/provider/"):
            self._json(HTTPStatus.NOT_FOUND, response_payload("rejected", errors=["Unknown endpoint."]))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ProviderError("Request body is missing or too large.")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ProviderError("Request body must be a JSON object.")
            if provider_mode() != "live":
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, response_payload("not_configured", errors=["OPENAI_API_KEY is not configured on the server."]))
                return
            if self.path == "/api/simulate":
                self._json(HTTPStatus.OK, execute_simulation(body))
                return
            action = self.path.removeprefix("/api/provider/")
            self._json(HTTPStatus.OK, execute_provider(action, body))
        except ProviderError as exc:
            self._json(HTTPStatus.BAD_REQUEST if "required" in str(exc) or "Unknown" in str(exc) else HTTPStatus.BAD_GATEWAY, response_payload("rejected", errors=[str(exc)]))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json(HTTPStatus.BAD_REQUEST, response_payload("rejected", errors=["Request body must be valid JSON."]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CONSEQUENCE and its server-side OpenAI provider")
    parser.add_argument("--host", default=os.environ.get("CONSEQUENCE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONSEQUENCE_PORT", "4173")))
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise SystemExit(f"Missing project directory: {ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), ConsequenceHandler)
    print(f"CONSEQUENCE server listening on http://{args.host}:{args.port} ({provider_mode()} provider)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
