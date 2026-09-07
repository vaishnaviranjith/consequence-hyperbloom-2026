"""Run the real provider stages against an already-started CONSEQUENCE server."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def call_provider(port: int, action: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/provider/{action}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{action} request failed: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"{action} request failed") from exc
    if envelope.get("status") != "ok":
        raise RuntimeError(f"{action} provider response was rejected")
    return envelope["data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4175)
    parser.add_argument("--input", type=Path, default=Path("/tmp/live-input.json"))
    parser.add_argument("--computed", type=Path, default=Path("/tmp/live-computed.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/live-provider-results.json"))
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    chunks = source["chunks"]
    entities = source["entities"]
    relationships = source["relationships"]
    compact_entities = [{key: entity[key] for key in ("id", "label", "type")} for entity in entities]
    compact_relationships = [{key: relationship[key] for key in ("from", "to", "type")} for relationship in relationships]

    extraction = call_provider(args.port, "extract", {"chunks": chunks})
    hypotheses = call_provider(args.port, "hypotheses", {"chunks": chunks, "entities": compact_entities, "relationships": compact_relationships})
    mutation = call_provider(args.port, "mutation", {"text": source["changeProposal"]["statement"] + " next month", "entities": compact_entities, "relationships": compact_relationships})
    known_chunks = {chunk["id"] for chunk in chunks}
    extraction_grounded = all(set(item["evidence_chunk_ids"]) <= known_chunks for item in extraction["entities"] + extraction["relationships"] + extraction["uncertainties"])
    hypotheses_grounded = all(set(item["evidenceChunkIds"]) <= known_chunks for item in hypotheses)
    confirmed_pairs = {(item["from"], item["to"]) for item in relationships}
    hypothesis_pairs_are_separate = all((item["sourceEntityId"], item["targetEntityId"]) not in confirmed_pairs for item in hypotheses)
    relocation_ops = {(item.get("type"), item.get("source"), item.get("target")) for item in mutation["operations"]}
    relocation_validated = {("REMOVE_EDGE", "clinic", "buildingA"), ("ADD_EDGE", "clinic", "buildingB")} <= relocation_ops
    if not extraction_grounded or not hypotheses_grounded or not hypothesis_pairs_are_separate or not relocation_validated:
        raise RuntimeError("live provider output failed grounding or separation checks")
    computed = json.loads(args.computed.read_text())
    if not computed["validation"]["valid"] or not computed["simulation"]["consequences"]:
        raise RuntimeError("deterministic simulation did not accept the live mutation")
    consequence = computed["simulation"]["consequences"][0]
    explanation = call_provider(args.port, "explanation", {"consequence": consequence})
    mitigations = call_provider(args.port, "mitigations", {"consequence": consequence})
    if not isinstance(explanation, str) or not explanation.strip() or not all(all(item.get(key) for key in ("title", "owner", "verification")) for item in mitigations):
        raise RuntimeError("live explanation or mitigation output failed validation")
    args.output.write_text(json.dumps({"extraction": extraction, "hypotheses": hypotheses, "mutation": mutation, "computed": computed, "explanation": explanation, "mitigations": mitigations}, ensure_ascii=True))
    print(json.dumps({"extraction_entities": len(extraction["entities"]), "extraction_relationships": len(extraction["relationships"]), "hypotheses": len(hypotheses), "mutation_operations": len(mutation["operations"]), "grounding": True, "hypotheses_separate": True, "relocation_validated": True, "deterministic_consequences": len(computed["simulation"]["consequences"]), "explanation": True, "mitigations": len(mitigations)}))


if __name__ == "__main__":
    main()
