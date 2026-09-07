import json
import os
import unittest
from unittest.mock import patch

import server


class ServerProviderTests(unittest.TestCase):
    def test_fixture_mode_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(server.provider_mode(), "fixture")

    def test_live_mode_requires_key_and_never_returns_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder", "OPENAI_MODEL": "test-model"}, clear=True):
            self.assertEqual(server.provider_mode(), "live")
            self.assertEqual(server.configured_model(), "test-model")
            status = {"mode": server.provider_mode(), "model": server.configured_model()}
            self.assertNotIn("placeholder", json.dumps(status))

    def test_gateway_mode_uses_confirmed_default_model(self):
        with patch.dict(os.environ, {"V4_LLM_BASE_URL": "https://gateway.invalid/v1", "V4_RUN_TOKEN": "placeholder"}, clear=True):
            self.assertEqual(server.provider_mode(), "live")
            self.assertEqual(server.configured_model(), "gpt-5.6-luna")

    def test_request_uses_gateway_model_and_supported_json_mode(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'

        with patch.dict(os.environ, {"V4_LLM_BASE_URL": "https://gateway.invalid/v1", "V4_RUN_TOKEN": "placeholder"}, clear=True):
            with patch("server.urllib.request.urlopen", return_value=FakeResponse()) as request:
                server._call_openai("test", "explanation", {"x": 1})
                sent = json.loads(request.call_args.args[0].data.decode("utf-8"))
                self.assertEqual(sent["model"], "gpt-5.6-luna")
                self.assertNotIn("temperature", sent)
                self.assertEqual(sent["response_format"], {"type": "json_object"})
                self.assertIn("/chat/completions", request.call_args.args[0].full_url)

    def test_extraction_rejects_unknown_chunk(self):
        chunks = [{"id": "E-1", "content": "claim"}]
        payload = {
            "entities": [{"name": "Clinic", "type": "service", "description": "x", "attributes": {}, "confidence": 0.9, "evidence_chunk_ids": ["UNKNOWN"]}],
            "relationships": [],
            "uncertainties": [],
        }
        with self.assertRaises(server.ProviderError):
            server.validate_extraction(payload, chunks)

    def test_mutation_rejects_unknown_node(self):
        payload = {"statement": "move", "confidence": 0.9, "rationale": "x", "operations": [{"type": "UPDATE_NODE", "nodeId": "missing", "attributes": {}}]}
        with self.assertRaises(server.ProviderError):
            server.validate_mutation(payload, {"clinic"})

    def test_hypothesis_rejects_unknown_chunk(self):
        payload = [{"id": "h1", "sourceEntityId": "clinic", "targetEntityId": "route", "relationshipType": "may_depend_on", "status": "HYPOTHESIS", "confidence": 0.6, "rationale": "x", "evidenceChunkIds": ["UNKNOWN"], "verificationTest": "test"}]
        with self.assertRaises(server.ProviderError):
            server.validate_hypotheses(payload, [{"id": "E-1", "content": "claim"}])

    def test_malformed_openai_response_is_rejected_without_secret(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder"}, clear=True):
            with patch("server.urllib.request.urlopen", side_effect=server.urllib.error.URLError("network")):
                with self.assertRaisesRegex(server.ProviderError, "Live provider request failed"):
                    server._call_openai("test", "explanation", {"x": 1})


if __name__ == "__main__":
    unittest.main()
