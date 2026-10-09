"""Integration tests for the OrcaRouter HTTP surface and the request path.

These exercise the real Flask routes with the real provider/credential/catalog
code, replacing only the two outbound boundaries (the model-list HTTP call and
the inference HTTP call) with local fakes. Nothing here needs a network or a
real credential.

    python -m unittest tests.test_web_integration -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as webapp  # noqa: E402
import orcarouter  # noqa: E402
from tests.test_orcarouter import FAKE_CODE, FAKE_KEY as PKCE_KEY, FakeAuthServer  # noqa: E402

FAKE_KEY = "sk-orca-integration-0000000000000000000000000000"
FAKE_STORED_KEY = "sk-orca-integration-1111111111111111111111111111"

CATALOG_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "vendor/text-only",
            "name": "Text Only",
            "owned_by": "vendor",
            "context_length": 128000,
            "supported_endpoint_types": ["openai", "anthropic"],
            "architecture": {"input_modalities": ["text"]},
        },
        {
            "id": "vendor/vision",
            "name": "Vision",
            "owned_by": "vendor",
            "context_length": 200000,
            "supported_endpoint_types": ["openai"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {"id": "vendor/painter", "supported_endpoint_types": ["image-generation"]},
        {"id": "vendor/embed", "supported_endpoint_types": ["embeddings"]},
    ],
}


class FakeHttpResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self, limit=None):
        return self._body if limit is None else self._body[:limit]

    def __enter__(self):
        return self

    def close(self):
        return None

    def __exit__(self, *args):
        return False


COMPLETION_PAYLOAD = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def fake_urlopen(
    *,
    catalog=CATALOG_PAYLOAD,
    catalog_status=200,
    completion=COMPLETION_PAYLOAD,
    completion_status=200,
    captured=None,
):
    """Stand in for ``urllib.request.urlopen`` on both outbound boundaries.

    Everything above it -- model discovery, capability filtering, caching,
    redaction and the inference request builder -- is the real implementation.
    """
    def opener(request, timeout=None, **kwargs):
        url = request.full_url
        if captured is not None:
            captured.append(request)
        if "/models" in url:
            payload, status = catalog, catalog_status
        else:
            payload, status = completion, completion_status
        if status >= 400:
            body = json.dumps(payload).encode()
            raise urllib.error.HTTPError(url, status, "error", {}, FakeHttpResponse(body, status))
        return FakeHttpResponse(payload)

    return opener


def catalog_calls(captured):
    return [request.full_url for request in captured if "/models" in request.full_url]


def inference_calls(captured):
    return [request.full_url for request in captured if "/models" not in request.full_url]


class WebIntegrationCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = orcarouter.CredentialStore(self.root / "orcarouter.json")
        self.cache = orcarouter.CatalogCache(self.root / "catalog.json")
        self.patches = [
            mock.patch.object(webapp, "CREDENTIAL_STORE", self.store),
            mock.patch.object(webapp, "CATALOG_CACHE", self.cache),
            mock.patch.object(webapp, "CONNECT_MANAGER", orcarouter.ConnectManager(store=self.store)),
            mock.patch.dict(os.environ, {"ORCAROUTER_API_KEY": FAKE_KEY}, clear=False),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.client = webapp.app.test_client()

    # -- registry and credential surface ----------------------------------

    def test_provider_registry_lists_both_orcarouter_entries(self):
        payload = self.client.get("/api/providers").get_json()
        ids = [provider["id"] for provider in payload["providers"]]
        self.assertEqual(ids, ["custom", "orcarouter", "orcarouter-oauth"])
        labels = {provider["id"]: provider["label"] for provider in payload["providers"]}
        self.assertEqual(labels["orcarouter"], "OrcaRouter - API")
        self.assertEqual(labels["orcarouter-oauth"], "OrcaRouter - Auth")
        self.assertFalse(payload["credential"]["configured"] is None)

    def test_credential_status_masks_the_key_and_never_returns_it(self):
        self.store.save(FAKE_STORED_KEY, source="api_key")
        payload = self.client.get("/api/orcarouter/credential").get_json()
        self.assertTrue(payload["configured"])
        self.assertEqual(payload["source"], "api_key")
        self.assertNotIn(FAKE_STORED_KEY, json.dumps(payload))
        self.assertTrue(payload["masked"].startswith("sk-orca-…"))

    def test_api_key_adapter_saves_clears_and_validates(self):
        bad = self.client.post("/api/orcarouter/credential", json={"api_key": "nope"})
        self.assertEqual(bad.status_code, 400)
        self.assertIsNone(self.store.read())
        good = self.client.post("/api/orcarouter/credential", json={"api_key": FAKE_STORED_KEY})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(self.store.read().key, FAKE_STORED_KEY)
        self.assertNotIn(FAKE_STORED_KEY, good.get_data(as_text=True))
        cleared = self.client.delete("/api/orcarouter/credential")
        self.assertEqual(cleared.status_code, 200)
        self.assertIsNone(self.store.read())

    # -- catalog ----------------------------------------------------------

    def test_model_list_comes_from_the_api_and_is_capability_filtered(self):
        captured = []
        with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
            payload = self.client.post(
                "/api/orcarouter/models", json={"provider_id": "orcarouter", "capability": "chat"}
            ).get_json()
        self.assertEqual(catalog_calls(captured), ["https://api.orcarouter.ai/v1/models?capability=chat"])
        self.assertEqual(captured[0].headers["Authorization"], f"Bearer {FAKE_KEY}")
        self.assertEqual([model["id"] for model in payload["models"]], ["vendor/text-only", "vendor/vision"])
        self.assertEqual(payload["source"], "live")
        self.assertFalse(payload["degraded"])

    def test_model_list_never_sends_the_key_to_the_browser(self):
        with mock.patch("urllib.request.urlopen", fake_urlopen()):
            body = self.client.post(
                "/api/orcarouter/models", json={"provider_id": "orcarouter", "capability": "chat"}
            ).get_data(as_text=True)
        self.assertNotIn(FAKE_KEY, body)

    def test_multimodal_request_only_returns_models_declaring_image_input(self):
        with mock.patch("urllib.request.urlopen", fake_urlopen()):
            payload = self.client.post(
                "/api/orcarouter/models",
                json={"provider_id": "orcarouter", "capability": "chat", "modality": "image"},
            ).get_json()
        self.assertEqual([model["id"] for model in payload["models"]], ["vendor/vision"])

    def test_each_capability_uses_its_own_filter_and_query(self):
        captured = []
        with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
            embedding = self.client.post(
                "/api/orcarouter/models", json={"provider_id": "orcarouter", "capability": "embedding"}
            ).get_json()
            image = self.client.post(
                "/api/orcarouter/models", json={"provider_id": "orcarouter", "capability": "image"}
            ).get_json()
        self.assertEqual([model["id"] for model in embedding["models"]], ["vendor/embed"])
        self.assertEqual([model["id"] for model in image["models"]], ["vendor/painter"])
        self.assertEqual(
            catalog_calls(captured),
            [
                "https://api.orcarouter.ai/v1/models?capability=embedding",
                "https://api.orcarouter.ai/v1/models?capability=image",
            ],
        )

    def test_unknown_capability_is_rejected(self):
        response = self.client.post(
            "/api/orcarouter/models", json={"provider_id": "orcarouter", "capability": "telepathy"}
        )
        self.assertEqual(response.status_code, 400)

    def test_catalog_outage_falls_back_to_the_verified_seed_and_says_so(self):
        with mock.patch("urllib.request.urlopen", fake_urlopen(catalog={"error": "boom"}, catalog_status=503)):
            payload = self.client.post(
                "/api/orcarouter/models", json={"provider_id": "orcarouter", "capability": "chat"}
            ).get_json()
        self.assertEqual(payload["source"], "seed")
        self.assertTrue(payload["degraded"])
        self.assertEqual(
            [model["id"] for model in payload["models"]], list(orcarouter.SEED_MODEL_IDS)
        )
        # The seed never reaches a multimodal dropdown it cannot prove.
        with mock.patch("urllib.request.urlopen", fake_urlopen(catalog={"error": "boom"}, catalog_status=503)):
            multimodal = self.client.post(
                "/api/orcarouter/models",
                json={"provider_id": "orcarouter", "capability": "chat", "modality": "image"},
            ).get_json()
        self.assertEqual(multimodal["models"], [])

    def test_pkce_provider_without_login_reports_an_actionable_error(self):
        response = self.client.post(
            "/api/orcarouter/models", json={"provider_id": "orcarouter-oauth", "capability": "chat"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("连接", response.get_json()["error"])

    # -- inference path ---------------------------------------------------

    def test_api_key_provider_routes_inference_to_the_relay(self):
        captured = []
        with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
            response = self.client.post(
                "/api/test/probe",
                json={
                    "provider_id": "orcarouter",
                    "api_model": "vendor/text-only",
                    "prompt": "hi",
                    "expected_count": 10,
                    "temperature": None,
                },
            )
        self.assertEqual(response.status_code, 200)
        request = captured[-1]
        self.assertEqual(request.full_url, "https://api.orcarouter.ai/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], f"Bearer {FAKE_KEY}")
        self.assertNotIn("www.orcarouter.ai", request.full_url)

    def test_pkce_provider_uses_the_same_downstream_request(self):
        """The request path must not care which adapter produced the credential."""
        self.store.save(FAKE_STORED_KEY, source="pkce", account_id="42", scope="api")
        captured = []
        with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
            response = self.client.post(
                "/api/test/probe",
                json={
                    "provider_id": "orcarouter-oauth",
                    "api_model": "vendor/text-only",
                    "prompt": "hi",
                    "expected_count": 10,
                    "temperature": None,
                },
            )
        self.assertEqual(response.status_code, 200)
        request = captured[-1]
        self.assertEqual(request.full_url, "https://api.orcarouter.ai/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], f"Bearer {FAKE_STORED_KEY}")
        self.assertNotIn(FAKE_STORED_KEY, response.get_data(as_text=True))

    def test_custom_provider_still_uses_its_own_base_url(self):
        captured = []
        with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
            response = self.client.post(
                "/api/test/probe",
                json={
                    "provider_id": "custom",
                    "base_url": "https://example.test/v1",
                    "api_key": "whatever",
                    "api_model": "some-model",
                    "prompt": "hi",
                    "expected_count": 10,
                    "temperature": None,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured[-1].full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(captured[-1].headers["Authorization"], "Bearer whatever")

    def test_missing_model_is_refused_before_any_request(self):
        captured = []
        with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
            response = self.client.post(
                "/api/test/probe",
                json={"provider_id": "orcarouter", "api_model": "", "prompt": "hi", "expected_count": 10},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("选择模型", response.get_json()["error"])
        self.assertEqual(captured, [])

    def test_unknown_provider_is_refused(self):
        response = self.client.post(
            "/api/test/probe",
            json={"provider_id": "openai", "api_model": "x", "prompt": "hi", "expected_count": 10},
        )
        self.assertEqual(response.status_code, 400)

    def test_revoked_credential_marks_reauth_without_any_refresh_attempt(self):
        stored = self.store.save(FAKE_STORED_KEY, source="pkce", account_id="42", scope="api")
        captured = []
        with mock.patch(
            "urllib.request.urlopen",
            fake_urlopen(
                completion={"error": {"message": "invalid api key"}},
                completion_status=401,
                captured=captured,
            ),
        ):
            response = self.client.post(
                "/api/test/probe",
                json={
                    "provider_id": "orcarouter-oauth",
                    "api_model": "vendor/text-only",
                    "prompt": "hi",
                    "expected_count": 10,
                    "temperature": None,
                },
            )
        message = response.get_json()["error"]
        self.assertEqual(response.status_code, 502)
        self.assertIn("重新连接", message)
        # Exactly one attempt: the revoked credential is not replayed against a
        # second wire format, no refresh grant is invented, and no request is
        # ever sent to the authentication origin.
        self.assertEqual(inference_calls(captured), ["https://api.orcarouter.ai/v1/chat/completions"])
        self.assertNotIn("/auth", "".join(request.full_url for request in captured))
        after = self.store.read()
        self.assertTrue(after.needs_reauth)
        self.assertEqual(after.key, stored.key)
        # A subsequent call reports reauthentication instead of reusing the key.
        later = self.client.post(
            "/api/test/probe",
            json={
                "provider_id": "orcarouter-oauth",
                "api_model": "vendor/text-only",
                "prompt": "hi",
                "expected_count": 10,
            },
        )
        self.assertEqual(later.status_code, 400)
        self.assertIn("重新连接", later.get_json()["error"])

    def test_stale_generation_401_cannot_poison_a_newer_credential(self):
        response = self.client.post(
            "/api/orcarouter/reauth", json={"generation": 99, "account_id": "42"}
        )
        self.assertFalse(response.get_json()["applied"])
        stored = self.store.save(FAKE_STORED_KEY, source="pkce", account_id="42")
        self.assertFalse(
            self.client.post(
                "/api/orcarouter/reauth", json={"generation": stored.generation + 5, "account_id": "42"}
            ).get_json()["applied"]
        )
        self.assertFalse(self.store.read().needs_reauth)

    # -- connect endpoints ------------------------------------------------

    def test_connect_start_returns_an_s256_authorize_url_without_the_verifier(self):
        payload = self.client.post("/api/orcarouter/connect/start", json={}).get_json()
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(payload["authorize_url"]).query)
        self.assertEqual(urllib.parse.urlsplit(payload["authorize_url"]).netloc, "www.orcarouter.ai")
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["app_name"], ["ModelTrace"])
        self.assertNotIn("code_verifier", payload["authorize_url"])
        self.assertEqual(payload["status"], "pending")
        attempt = webapp.CONNECT_MANAGER.get(payload["attempt"])
        self.assertNotIn(attempt.verifier, json.dumps(payload))

    def test_connect_cancel_releases_the_lock_and_a_new_login_can_start(self):
        first = self.client.post("/api/orcarouter/connect/start", json={}).get_json()
        cancelled = self.client.post(
            "/api/orcarouter/connect/cancel", json={"attempt": first["attempt"]}
        ).get_json()
        self.assertTrue(cancelled["cancelled"])
        second = self.client.post("/api/orcarouter/connect/start", json={}).get_json()
        self.assertEqual(second["status"], "pending")
        self.assertNotEqual(second["attempt"], first["attempt"])

    def test_connect_status_and_cancel_require_a_known_attempt(self):
        self.assertEqual(self.client.get("/api/orcarouter/connect/status").status_code, 400)
        self.assertEqual(
            self.client.get("/api/orcarouter/connect/status?attempt=987654").status_code, 404
        )
        self.assertEqual(
            self.client.post("/api/orcarouter/connect/cancel", json={"attempt": 987654}).status_code, 404
        )

    def test_pkce_login_runs_through_the_web_adapter_and_stores_one_credential(self):
        """authorize -> exchange -> persist -> inference, through the real endpoints."""
        auth = FakeAuthServer()
        self.addCleanup(auth.close)
        with mock.patch.dict(os.environ, {"ORCA_AUTH_BASE_URL": auth.origin}):
            started = self.client.post("/api/orcarouter/connect/start", json={}).get_json()
            self.assertTrue(started["callback_url"].startswith("http://127.0.0.1:"))
            self.assertEqual(
                urllib.parse.urlsplit(started["authorize_url"]).netloc,
                urllib.parse.urlsplit(auth.origin).netloc,
            )
            attempt = webapp.CONNECT_MANAGER.get(started["attempt"])
            auth.server.expected_verifier = attempt.verifier

            # A human pasting the code the consent screen displayed ("Show me a code").
            done = self.client.post(
                "/api/orcarouter/connect/code",
                json={"attempt": started["attempt"], "code": FAKE_CODE},
            ).get_json()
            self.assertEqual(done["status"], "authorized")
            self.assertNotIn(PKCE_KEY, json.dumps(done))
            self.assertEqual(self.store.read().key, PKCE_KEY)
            self.assertEqual(self.store.read().source, "pkce")

            # The stored credential is then used without any key being passed in.
            captured = []
            with mock.patch("urllib.request.urlopen", fake_urlopen(captured=captured)):
                response = self.client.post(
                    "/api/test/probe",
                    json={
                        "provider_id": "orcarouter-oauth",
                        "api_model": "vendor/text-only",
                        "prompt": "hi",
                        "expected_count": 10,
                    },
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured[-1].headers["Authorization"], f"Bearer {PKCE_KEY}")

    def test_pkce_denial_leaves_the_credential_untouched(self):
        auth = FakeAuthServer()
        self.addCleanup(auth.close)
        with mock.patch.dict(os.environ, {"ORCA_AUTH_BASE_URL": auth.origin}):
            started = self.client.post("/api/orcarouter/connect/start", json={}).get_json()
            attempt = webapp.CONNECT_MANAGER.get(started["attempt"])
            webapp.CONNECT_MANAGER.handle_callback(
                attempt, state=attempt.state, error="access_denied", code=None
            )
            denied = self.client.get(
                f"/api/orcarouter/connect/status?attempt={started['attempt']}"
            ).get_json()
        self.assertEqual(denied["status"], "denied")
        self.assertIsNone(self.store.read())
        self.assertEqual(auth.server.requests, [])

    def test_index_page_renders_both_orcarouter_entries(self):
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn("OrcaRouter - API", body)
        self.assertIn("OrcaRouter - Auth", body)
        self.assertIn("orca-logo.png", body)
        self.assertIn("orcarouter-provider.js", body)
        self.assertIn("orcarouter-connect.js", body)


if __name__ == "__main__":
    unittest.main()
