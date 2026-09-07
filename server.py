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
OUTPUTS = ROOT / "outputs"
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
        if self.path == "/api/provider/status":
            self._json(HTTPStatus.OK, {"mode": provider_mode(), "model": configured_model() if provider_mode() == "live" else None})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/provider/"):
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
    if not OUTPUTS.is_dir():
        raise SystemExit(f"Missing outputs directory: {OUTPUTS}")
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
