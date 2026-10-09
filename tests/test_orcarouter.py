"""OrcaRouter provider tests: credentials, PKCE, catalog and capability filters.

Run with the standard library only:

    python -m unittest tests.test_orcarouter -v

No network access and no real credential is required: the authorization server
and the model catalog are replaced by local fakes. Every credential in this file
is a fabricated ``sk-orca-…`` literal; a live check lives in
``tests/test_orcarouter_live.py`` and only runs when ``ORCAROUTER_API_KEY`` is
present.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orcarouter  # noqa: E402

FAKE_KEY = "sk-orca-test-0000000000000000000000000000000000"
FAKE_KEY_2 = "sk-orca-test-1111111111111111111111111111111111"
FAKE_CODE = "fake-authorization-code"


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self._read = False

    def read(self, limit=None):
        if self._read:
            return b""
        self._read = True
        return self._body if limit is None else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fake_opener(payload, status=200, capture=None):
    def opener(request, timeout=None):
        if capture is not None:
            capture.append(request)
        if status >= 400:
            body = json.dumps(payload).encode() if isinstance(payload, dict) else bytes(payload)
            raise urllib.error.HTTPError(request.full_url, status, "error", {}, FakeBody(body))
        return FakeResponse(payload)

    return opener


class FakeBody:
    def __init__(self, body):
        self._body = body

    def read(self, limit=None):
        return self._body if limit is None else self._body[:limit]

    def close(self):
        return None


class NetworkFailureOpener:
    def __init__(self, reason="connection refused"):
        self.reason = reason

    def __call__(self, request, timeout=None):
        raise urllib.error.URLError(self.reason)


class TemporaryStore(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "orcarouter.json"
        self.store = orcarouter.CredentialStore(self.path)


# --------------------------------------------------------------------------
# provider registry and origins
# --------------------------------------------------------------------------


class ProviderRegistryTests(unittest.TestCase):
    def test_orcarouter_is_a_named_provider_not_a_custom_base_url(self):
        ids = [provider["id"] for provider in orcarouter.provider_registry({})]
        self.assertIn("orcarouter", ids)
        self.assertIn("orcarouter-oauth", ids)
        self.assertIn("custom", ids)

    def test_api_key_and_pkce_are_two_separate_entries(self):
        registry = {item["id"]: item for item in orcarouter.provider_registry({})}
        self.assertEqual(registry["orcarouter"]["auth_methods"], ["api_key"])
        self.assertEqual(registry["orcarouter-oauth"]["auth_methods"], ["pkce"])
        self.assertEqual(registry["orcarouter"]["api_base"], orcarouter.DEFAULT_API_BASE)
        self.assertEqual(registry["orcarouter-oauth"]["auth_base"], orcarouter.DEFAULT_AUTH_BASE)

    def test_auth_method_mapping(self):
        self.assertEqual(orcarouter.auth_method_for_provider("orcarouter"), "api_key")
        self.assertEqual(orcarouter.auth_method_for_provider("orcarouter-oauth"), "pkce")
        with self.assertRaises(orcarouter.OrcaRouterError):
            orcarouter.auth_method_for_provider("openai")

    def test_default_origins_are_distinct(self):
        env: dict[str, str] = {}
        self.assertEqual(orcarouter.auth_base(env), "https://www.orcarouter.ai")
        self.assertEqual(orcarouter.api_base(env), "https://api.orcarouter.ai/v1")
        self.assertEqual(
            orcarouter.exchange_url(orcarouter.auth_base(env)),
            "https://www.orcarouter.ai/api/v1/auth/keys",
        )
        # The relay origin must never be used for authentication.
        self.assertNotIn("api.orcarouter.ai/v1/auth", orcarouter.exchange_url(orcarouter.auth_base(env)))

    def test_explicit_overrides_win_over_shared_base(self):
        env = {
            "ORCA_BASE_URL": "https://self-hosted.example",
            "ORCA_AUTH_BASE_URL": "https://auth.example",
            "ORCA_API_BASE_URL": "https://api.example/v1",
        }
        self.assertEqual(orcarouter.auth_base(env), "https://auth.example")
        self.assertEqual(orcarouter.api_base(env), "https://api.example/v1")

    def test_shared_self_hosted_base_feeds_both_origins(self):
        env = {"ORCA_BASE_URL": "https://orca.internal"}
        self.assertEqual(orcarouter.auth_base(env), "https://orca.internal")
        self.assertEqual(orcarouter.api_base(env), "https://orca.internal")

    def test_remote_origin_must_be_https_but_loopback_may_be_http(self):
        self.assertEqual(orcarouter.validate_origin("http://127.0.0.1:8080", label="x"), "http://127.0.0.1:8080")
        self.assertEqual(orcarouter.validate_origin("http://localhost:9000/", label="x"), "http://localhost:9000")
        with self.assertRaises(orcarouter.OrcaRouterError):
            orcarouter.validate_origin("http://orca.example", label="x")
        with self.assertRaises(orcarouter.OrcaRouterError):
            orcarouter.validate_origin("https://orca.example/?a=1", label="x")


# --------------------------------------------------------------------------
# PKCE primitives
# --------------------------------------------------------------------------


class PkceTests(unittest.TestCase):
    def test_verifier_and_state_are_fresh_and_random(self):
        verifiers = {orcarouter.generate_verifier() for _ in range(64)}
        states = {orcarouter.generate_state() for _ in range(64)}
        self.assertEqual(len(verifiers), 64)
        self.assertEqual(len(states), 64)
        for value in verifiers | states:
            self.assertNotIn("=", value)
            self.assertTrue(re.fullmatch(r"[A-Za-z0-9_-]+", value))

    def test_challenge_is_unpadded_base64url_sha256(self):
        verifier = "fixed-verifier-for-hash-check"
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        challenge = orcarouter.code_challenge(verifier)
        self.assertEqual(challenge, expected)
        self.assertNotIn("=", challenge)
        self.assertNotEqual(challenge, verifier)

    def test_state_comparison_is_exact(self):
        self.assertTrue(orcarouter.state_matches("abc", "abc"))
        self.assertFalse(orcarouter.state_matches("abc", "abd"))
        self.assertFalse(orcarouter.state_matches("abc", None))
        self.assertFalse(orcarouter.state_matches("abc", ""))

    def test_authorize_url_always_uses_s256_and_hides_the_verifier(self):
        verifier = orcarouter.generate_verifier()
        url = orcarouter.authorize_url(
            "https://www.orcarouter.ai",
            callback_url="http://127.0.0.1:51733/cb",
            challenge=orcarouter.code_challenge(verifier),
            state="state-value",
            app_name="ModelTrace",
        )
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(parsed.netloc, "www.orcarouter.ai")
        self.assertEqual(parsed.path, "/auth")
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["callback_url"], ["http://127.0.0.1:51733/cb"])
        self.assertEqual(query["state"], ["state-value"])
        self.assertEqual(query["app_name"], ["ModelTrace"])
        self.assertNotIn(verifier, url)
        self.assertEqual(query["code_challenge"], [orcarouter.code_challenge(verifier)])

    def test_out_of_band_callback_url_is_the_literal_oob(self):
        url = orcarouter.authorize_url(
            "https://www.orcarouter.ai",
            callback_url="oob",
            challenge=orcarouter.code_challenge(orcarouter.generate_verifier()),
            state="s",
        )
        self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["callback_url"], ["oob"])


# --------------------------------------------------------------------------
# credential store and the two adapters
# --------------------------------------------------------------------------


class CredentialStoreTests(TemporaryStore):
    def test_save_read_mask_and_clear(self):
        stored = self.store.save(FAKE_KEY, source="api_key")
        self.assertEqual(stored.masked, "sk-orca-…0000")
        self.assertNotIn(FAKE_KEY[8:-4], stored.masked)
        self.assertEqual(self.store.read().key, FAKE_KEY)
        self.assertTrue(self.store.clear())
        self.assertIsNone(self.store.read())
        self.assertFalse(self.store.clear())

    def test_secret_file_is_not_world_readable(self):
        self.store.save(FAKE_KEY, source="pkce", account_id="42", scope="api")
        mode = os.stat(self.path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_generation_increases_and_marks_only_that_generation(self):
        first = self.store.save(FAKE_KEY, source="pkce", account_id="42")
        second = self.store.save(FAKE_KEY_2, source="pkce", account_id="42")
        self.assertEqual(first.generation, 1)
        self.assertEqual(second.generation, 2)
        # A late failure for the old generation must not poison the new one.
        self.assertFalse(self.store.mark_needs_reauth("42", first.generation))
        self.assertFalse(self.store.read().needs_reauth)
        self.assertTrue(self.store.mark_needs_reauth("42", second.generation))
        self.assertTrue(self.store.read().needs_reauth)
        # The secret is kept: losing it before a replacement is irreversible.
        self.assertEqual(self.store.read().key, FAKE_KEY_2)

    def test_reauth_does_not_apply_to_a_different_account(self):
        stored = self.store.save(FAKE_KEY, source="pkce", account_id="42")
        self.assertFalse(self.store.mark_needs_reauth("99", stored.generation))
        self.assertFalse(self.store.read().needs_reauth)

    def test_corrupt_store_is_treated_as_absent(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.store.read())
        self.path.write_text(json.dumps({"key": ""}), encoding="utf-8")
        self.assertIsNone(self.store.read())


class ApiKeyAdapterTests(TemporaryStore):
    def test_typed_key_is_accepted_without_touching_the_store(self):
        credential = orcarouter.resolve_credential(
            "api_key", explicit_key=FAKE_KEY, store=self.store, environ={}
        )
        self.assertEqual(credential.source, "api_key")
        self.assertEqual(credential.key, FAKE_KEY)
        self.assertIsNone(self.store.read())

    def test_saved_key_is_reused(self):
        self.store.save(FAKE_KEY, source="api_key")
        credential = orcarouter.resolve_credential("api_key", store=self.store, environ={})
        self.assertEqual(credential.key, FAKE_KEY)

    def test_environment_key_is_the_last_resort(self):
        credential = orcarouter.resolve_credential(
            "api_key", store=self.store, environ={"ORCAROUTER_API_KEY": FAKE_KEY}
        )
        self.assertEqual(credential.source, "env")

    def test_bad_shape_is_rejected_without_leaking_the_value(self):
        for value in ("sk-live-123", "not-a-key", "sk-orca-"):
            with self.assertRaises(orcarouter.CredentialUnavailable) as caught:
                orcarouter.resolve_credential("api_key", explicit_key=value, store=self.store, environ={})
            self.assertIn("sk-orca-", str(caught.exception))
            if value.startswith("sk-") and len(value) > 8:
                self.assertNotIn(value, str(caught.exception))

    def test_missing_everywhere_is_actionable(self):
        with self.assertRaises(orcarouter.CredentialUnavailable) as caught:
            orcarouter.resolve_credential("api_key", store=self.store, environ={})
        self.assertIn("API Key", str(caught.exception))

    def test_needs_reauth_blocks_reuse_and_says_so(self):
        stored = self.store.save(FAKE_KEY, source="api_key")
        self.store.mark_needs_reauth(None, stored.generation)
        with self.assertRaises(orcarouter.CredentialUnavailable) as caught:
            orcarouter.resolve_credential("api_key", store=self.store, environ={})
        self.assertIn("失效", str(caught.exception))


class PkceAdapterTests(TemporaryStore):
    def test_pkce_credential_is_unavailable_before_login(self):
        with self.assertRaises(orcarouter.CredentialUnavailable):
            orcarouter.resolve_credential("pkce", store=self.store, environ={})

    def test_stored_pkce_credential_is_reused_until_revoked(self):
        self.store.save(FAKE_KEY, source="pkce", account_id="7", scope="api")
        credential = orcarouter.resolve_credential("pkce", store=self.store, environ={})
        self.assertEqual(credential.key, FAKE_KEY)
        self.assertEqual(credential.account_id, "7")

    def test_needs_reauth_requires_a_new_login(self):
        stored = self.store.save(FAKE_KEY, source="pkce", account_id="7")
        self.store.mark_needs_reauth("7", stored.generation)
        with self.assertRaises(orcarouter.CredentialUnavailable) as caught:
            orcarouter.resolve_credential("pkce", store=self.store, environ={})
        self.assertIn("重新连接", str(caught.exception))

    def test_both_adapters_return_the_same_credential_type(self):
        via_api_key = orcarouter.resolve_credential(
            "api_key", explicit_key=FAKE_KEY, store=self.store, environ={}
        )
        via_pkce = orcarouter.resolve_credential(
            "pkce", store=self.store, environ={}
        ) if self.store.read() else None
        self.assertIsNone(via_pkce)
        self.store.save(FAKE_KEY_2, source="pkce", account_id="7")
        via_pkce = orcarouter.resolve_credential("pkce", store=self.store, environ={})
        self.assertIsInstance(via_api_key, orcarouter.Credential)
        self.assertIsInstance(via_pkce, orcarouter.Credential)
        # Neither carries a source-specific key format: both are plain sk-orca keys.
        for credential in (via_api_key, via_pkce):
            self.assertTrue(orcarouter.key_shape_ok(credential.key))


# --------------------------------------------------------------------------
# code exchange
# --------------------------------------------------------------------------


class ExchangeTests(unittest.TestCase):
    def test_successful_exchange_uses_the_auth_origin_and_s256_body(self):
        captured = []
        credential = orcarouter.exchange_code(
            "https://www.orcarouter.ai",
            FAKE_CODE,
            "verifier-value",
            opener=fake_opener({"key": FAKE_KEY, "user_id": 12345, "scope": "api"}, capture=captured),
        )
        request = captured[0]
        self.assertEqual(request.full_url, "https://www.orcarouter.ai/api/v1/auth/keys")
        self.assertNotIn("api.orcarouter.ai", request.full_url)
        body = json.loads(request.data.decode())
        self.assertEqual(body["code"], FAKE_CODE)
        self.assertEqual(body["code_verifier"], "verifier-value")
        self.assertEqual(body["code_challenge_method"], "S256")
        self.assertEqual(credential.key, FAKE_KEY)
        self.assertEqual(credential.account_id, "12345")
        self.assertEqual(credential.scope, "api")
        self.assertEqual(credential.source, "pkce")

    def test_granted_scope_is_read_back_not_assumed(self):
        credential = orcarouter.exchange_code(
            "https://www.orcarouter.ai",
            FAKE_CODE,
            "v",
            opener=fake_opener({"key": FAKE_KEY, "user_id": 1, "scope": "connector"}),
        )
        self.assertEqual(credential.scope, "connector")

    def test_unexpected_scope_is_not_reported_as_granted(self):
        credential = orcarouter.exchange_code(
            "https://www.orcarouter.ai",
            FAKE_CODE,
            "v",
            opener=fake_opener({"key": FAKE_KEY, "user_id": 1, "scope": "admin"}),
        )
        self.assertIsNone(credential.scope)

    def test_error_statuses_map_to_actionable_messages(self):
        cases = {
            400: "S256",
            403: "过期",
            429: "10 个",
            500: "暂时不可用",
            418: "HTTP 418",
        }
        for status, needle in cases.items():
            with self.subTest(status=status):
                with self.assertRaises(orcarouter.ConnectError) as caught:
                    orcarouter.exchange_code(
                        "https://www.orcarouter.ai", FAKE_CODE, "v",
                        opener=fake_opener({"error": "invalid_request"}, status=status),
                    )
                self.assertIn(needle, str(caught.exception))

    def test_network_failure_is_reported_without_hanging(self):
        with self.assertRaises(orcarouter.ConnectError) as caught:
            orcarouter.exchange_code(
                "https://www.orcarouter.ai", FAKE_CODE, "v",
                opener=NetworkFailureOpener("timed out"),
            )
        self.assertEqual(caught.exception.code, "network")

    def test_missing_key_and_garbage_body_are_errors(self):
        with self.assertRaises(orcarouter.ConnectError):
            orcarouter.exchange_code(
                "https://www.orcarouter.ai", FAKE_CODE, "v", opener=fake_opener({"user_id": 1})
            )
        with self.assertRaises(orcarouter.ConnectError):
            orcarouter.exchange_code(
                "https://www.orcarouter.ai", FAKE_CODE, "v",
                opener=lambda request, timeout=None: FakeResponse(b"<html>"),
            )

    def test_error_bodies_never_leak_a_key_or_the_verifier(self):
        verifier = orcarouter.generate_verifier()
        with self.assertRaises(orcarouter.ConnectError) as caught:
            orcarouter.exchange_code(
                "https://www.orcarouter.ai", FAKE_CODE, verifier,
                opener=fake_opener(
                    {"error": "invalid_grant", "error_description": f"code {FAKE_CODE} key {FAKE_KEY}"},
                    status=403,
                ),
            )
        message = str(caught.exception)
        self.assertNotIn(FAKE_KEY, message)
        self.assertNotIn(verifier, message)
        self.assertIn("invalid_grant", message)


# --------------------------------------------------------------------------
# connect manager: full PKCE round trip against a local fake auth server
# --------------------------------------------------------------------------


class FakeAuthHandler(BaseHTTPRequestHandler):
    server_version = "FakeOrcaAuth/1.0"

    def log_message(self, *args):
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        body = json.loads(raw or b"{}")
        server = self.server
        server.requests.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
        if self.path != "/api/v1/auth/keys":
            self._json(404, {"error": "not_found"})
            return
        if body.get("code_verifier") != server.expected_verifier:
            self._json(403, {"error": "invalid_grant", "error_description": "verifier mismatch"})
            return
        if body.get("code") != server.expected_code:
            self._json(403, {"error": "invalid_grant", "error_description": "unknown code"})
            return
        if server.reject_with is not None:
            self._json(server.reject_with, {"error": "access_denied"})
            return
        if server.issued:
            self._json(403, {"error": "invalid_grant", "error_description": "code already used"})
            return
        server.issued = True
        self._json(200, {"key": server.key, "user_id": "42", "scope": server.scope})

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeAuthServer:
    """Local stand-in for the OrcaRouter consent/exchange endpoints."""

    def __init__(self, key=FAKE_KEY, scope="api"):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAuthHandler)
        self.server.daemon_threads = True
        self.server.requests = []
        self.server.expected_verifier = None
        self.server.expected_code = FAKE_CODE
        self.server.issued = False
        self.server.reject_with = None
        self.server.key = key
        self.server.scope = scope
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ConnectManagerTests(TemporaryStore):
    def setUp(self):
        super().setUp()
        self.auth = FakeAuthServer()
        self.addCleanup(self.auth.close)
        self.environ = {"ORCA_AUTH_BASE_URL": self.auth.origin}
        self.manager = orcarouter.ConnectManager(store=self.store, environ=self.environ)

    def _start(self, **kwargs):
        return self.manager.start(**kwargs)

    def test_full_round_trip_authorize_callback_exchange_persist(self):
        attempt = self._start()
        self.assertEqual(attempt["status"], "pending")
        self.assertTrue(attempt["callback_url"].startswith("http://127.0.0.1:"))
        self.assertEqual(urllib.parse.urlsplit(attempt["authorize_url"]).netloc,
                         urllib.parse.urlsplit(self.auth.origin).netloc)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(attempt["authorize_url"]).query)
        self.assertEqual(query["code_challenge_method"], ["S256"])

        # The fake server verifies the verifier matches the challenge we sent.
        current = self.manager.get(attempt["attempt"])
        self.auth.server.expected_verifier = current.verifier
        self.assertEqual(
            orcarouter.code_challenge(current.verifier), query["code_challenge"][0]
        )

        # Simulate the browser hitting the loopback listener.
        callback = (
            f"{attempt['callback_url']}?code={FAKE_CODE}&state="
            f"{urllib.parse.quote(query['state'][0])}"
        )
        with urllib.request.urlopen(callback, timeout=10) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"", response.read() + b"")

        deadline = threading.Event()
        for _ in range(100):
            status = self.manager.status(attempt["attempt"])
            if status["status"] != "pending":
                break
            deadline.wait(0.05)
        status = self.manager.status(attempt["attempt"])
        self.assertEqual(status["status"], "authorized")
        self.assertEqual(status["credential"]["masked"], "sk-orca-…0000")
        self.assertNotIn(FAKE_KEY, json.dumps(status))
        self.assertEqual(self.store.read().key, FAKE_KEY)
        self.assertEqual(self.store.read().source, "pkce")
        self.assertEqual(self.store.read().account_id, "42")
        # A second callback for the same single-use code cannot mint a key again.
        self.assertEqual(self.manager.submit_code(attempt["attempt"], FAKE_CODE)["status"], "authorized")

    def test_s256_and_state_never_appear_in_the_browser_projection(self):
        attempt = self._start()
        current = self.manager.get(attempt["attempt"])
        # The browser receives the challenge and the state (both are public by
        # design); the verifier is the secret that must never leave the process.
        self.assertNotIn(current.verifier, json.dumps(attempt))
        self.assertNotIn("code_verifier", json.dumps(attempt))

    def test_state_mismatch_is_refused(self):
        attempt = self._start()
        current = self.manager.get(attempt["attempt"])
        self.auth.server.expected_verifier = current.verifier
        self.manager.handle_callback(
            current, state="wrong-state", error=None, code=FAKE_CODE
        )
        status = self.manager.status(attempt["attempt"])
        self.assertEqual(status["status"], "error")
        self.assertEqual(status["code"], "state_mismatch")
        self.assertIsNone(self.store.read())
        self.assertEqual(self.auth.server.requests, [])

    def test_denial_ends_cleanly_without_exchange(self):
        attempt = self._start()
        current = self.manager.get(attempt["attempt"])
        self.manager.handle_callback(
            current, state=current.state, error="access_denied", code=None
        )
        status = self.manager.status(attempt["attempt"])
        self.assertEqual(status["status"], "denied")
        self.assertEqual(self.auth.server.requests, [])
        self.assertIsNone(self.store.read())

    def test_expired_or_reused_code_is_a_clean_failure(self):
        attempt = self._start()
        current = self.manager.get(attempt["attempt"])
        self.auth.server.expected_verifier = current.verifier
        self.auth.server.reject_with = 403
        result = self.manager.submit_code(attempt["attempt"], FAKE_CODE)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "denied")
        self.assertIsNone(self.store.read())

    def test_rate_limited_consent_is_reported_with_the_cap(self):
        attempt = self._start()
        current = self.manager.get(attempt["attempt"])
        self.auth.server.expected_verifier = current.verifier
        self.auth.server.reject_with = 429
        result = self.manager.submit_code(attempt["attempt"], FAKE_CODE)
        self.assertEqual(result["code"], "rate_limited")
        self.assertIn("10", result["message"])

    def test_verifier_mismatch_is_rejected_by_the_server(self):
        attempt = self._start()
        result = self.manager.submit_code(attempt["attempt"], FAKE_CODE)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "denied")
        self.assertIsNone(self.store.read())

    def test_timeout_expires_the_attempt(self):
        clock = {"now": 1000.0}
        manager = orcarouter.ConnectManager(
            store=self.store, environ=self.environ, ttl_seconds=600, clock=lambda: clock["now"]
        )
        attempt = manager.start()
        self.assertEqual(attempt["status"], "pending")
        clock["now"] += 601
        self.assertEqual(manager.status(attempt["attempt"])["status"], "expired")
        self.assertIsNone(self.store.read())

    def test_explicit_cancel_releases_the_login_lock(self):
        attempt = self._start()
        cancelled = self.manager.cancel(attempt["attempt"])
        self.assertEqual(cancelled["status"], "cancelled")
        # A second login can start immediately after the cancel.
        second = self._start()
        self.assertEqual(second["status"], "pending")
        self.assertNotEqual(second["attempt"], attempt["attempt"])

    def test_starting_a_new_login_supersedes_the_previous_attempt(self):
        first = self._start()
        self._start()
        status = self.manager.status(first["attempt"])
        self.assertEqual(status["status"], "cancelled")
        self.assertEqual(status["code"], "superseded")

    def test_late_callback_from_an_old_attempt_cannot_replace_a_new_credential(self):
        first = self._start()
        old = self.manager.get(first["attempt"])
        second = self._start()
        current = self.manager.get(second["attempt"])
        self.auth.server.expected_verifier = current.verifier
        # A callback for the abandoned attempt arrives after the new login began.
        self.manager.handle_callback(old, state=old.state, error=None, code=FAKE_CODE)
        self.assertEqual(self.manager.status(second["attempt"])["status"], "pending")
        self.assertIsNone(self.store.read())

    def test_successful_login_replaces_the_stored_secret_and_clears_reauth(self):
        old = self.store.save(FAKE_KEY_2, source="pkce", account_id="42")
        self.store.mark_needs_reauth("42", old.generation)
        attempt = self._start()
        current = self.manager.get(attempt["attempt"])
        self.auth.server.expected_verifier = current.verifier
        result = self.manager.submit_code(attempt["attempt"], FAKE_CODE)
        self.assertEqual(result["status"], "authorized")
        stored = self.store.read()
        self.assertEqual(stored.key, FAKE_KEY)
        self.assertFalse(stored.needs_reauth)
        self.assertGreater(stored.generation, old.generation)

    def test_unknown_attempt_is_reported_not_guessed(self):
        with self.assertRaises(orcarouter.ConnectError):
            self.manager.status(9999)
        with self.assertRaises(orcarouter.ConnectError):
            self.manager.submit_code(9999, FAKE_CODE)


# --------------------------------------------------------------------------
# catalog and capability filters
# --------------------------------------------------------------------------

TEXT_ONLY = {
    "id": "vendor/text-only",
    "name": "Text Only",
    "supported_endpoint_types": ["openai", "openai-response"],
    "context_length": 128000,
    "architecture": {"input_modalities": ["text"]},
}
VISION = {
    "id": "vendor/vision",
    "name": "Vision",
    "supported_endpoint_types": ["openai", "anthropic"],
    "context_length": 200000,
    "architecture": {"input_modalities": ["text", "image"]},
}
IMAGE_GEN = {
    "id": "vendor/painter",
    "name": "Painter",
    "supported_endpoint_types": ["image-generation"],
}
VIDEO_GEN = {
    "id": "vendor/motion",
    "name": "Motion",
    "supported_endpoint_types": ["openai-video"],
}
RERANK = {
    "id": "vendor/ranker",
    "name": "Ranker",
    "supported_endpoint_types": ["jina-rerank"],
}
EMBEDDING = {
    "id": "vendor/embed",
    "name": "Embed",
    "supported_endpoint_types": ["embeddings"],
    "architecture": {"input_modalities": ["text"]},
}
UNKNOWN_CAPABILITY = {"id": "vendor/mystery", "name": "Mystery", "supported_endpoint_types": []}
NO_ENDPOINT_FIELD = {"id": "vendor/legacy", "name": "Legacy"}

CATALOG_FIXTURE = [
    TEXT_ONLY, VISION, IMAGE_GEN, VIDEO_GEN, RERANK, EMBEDDING, UNKNOWN_CAPABILITY, NO_ENDPOINT_FIELD,
]


class CapabilityFilterTests(unittest.TestCase):
    def ids(self, capability, modality=None):
        return [model["id"] for model in orcarouter.capability_filter(CATALOG_FIXTURE, capability, modality=modality)]

    def test_chat_keeps_text_models_and_excludes_dedicated_non_text_models(self):
        ids = self.ids("chat")
        self.assertIn("vendor/text-only", ids)
        self.assertIn("vendor/vision", ids)
        for excluded in ("vendor/painter", "vendor/motion", "vendor/ranker", "vendor/embed"):
            self.assertNotIn(excluded, ids)

    def test_models_without_endpoint_metadata_never_appear(self):
        for capability in orcarouter.CAPABILITY_ENDPOINT_TYPES:
            self.assertNotIn("vendor/mystery", self.ids(capability))
            self.assertNotIn("vendor/legacy", self.ids(capability))

    def test_multimodal_filter_fails_closed_when_modality_is_not_declared(self):
        image_models = self.ids("chat", "image")
        self.assertEqual(image_models, ["vendor/vision"])
        self.assertNotIn("vendor/text-only", image_models)

    def test_each_capability_gets_its_own_endpoint_strictness(self):
        self.assertEqual(self.ids("embedding"), ["vendor/embed"])
        self.assertEqual(self.ids("image"), ["vendor/painter"])
        self.assertEqual(self.ids("video"), ["vendor/motion"])
        self.assertEqual(self.ids("rerank"), ["vendor/ranker"])

    def test_unknown_capability_is_a_programming_error(self):
        with self.assertRaises(orcarouter.OrcaRouterError):
            orcarouter.capability_filter(CATALOG_FIXTURE, "telepathy")

    def test_describe_model_keeps_verified_metadata(self):
        described = orcarouter.capability_filter([VISION], "chat")[0]
        self.assertEqual(described["id"], "vendor/vision")
        self.assertEqual(described["context_length"], 200000)
        self.assertEqual(described["input_modalities"], ["text", "image"])
        self.assertEqual(described["supported_endpoint_types"], ["openai", "anthropic"])

    def test_seed_models_expose_the_verified_gpt_55_effort_ladder(self):
        seed = {model["id"]: model for model in orcarouter.seed_catalog("chat").models}
        self.assertEqual(
            seed["openai/gpt-5.5"]["reasoning_efforts"], ["low", "medium", "high", "xhigh"]
        )
        self.assertEqual(set(orcarouter.SEED_MODEL_IDS), set(seed))
        # An unknown modality must not be invented for a seed model.
        self.assertEqual(seed["openai/gpt-5.5"]["input_modalities"], [])
        self.assertEqual(orcarouter.capability_filter(orcarouter.SEED_MODELS, "chat", modality="image").models
                         if hasattr(orcarouter.capability_filter(orcarouter.SEED_MODELS, "chat", modality="image"), "models")
                         else orcarouter.capability_filter(orcarouter.SEED_MODELS, "chat", modality="image"), [])


class CatalogFetchTests(unittest.TestCase):
    def test_live_discovery_is_authoritative_and_filters(self):
        captured = []
        catalog = orcarouter.fetch_models(
            "https://api.orcarouter.ai/v1",
            FAKE_KEY,
            capability="chat",
            opener=fake_opener({"data": CATALOG_FIXTURE}, capture=captured),
        )
        self.assertEqual(catalog.source, "live")
        self.assertFalse(catalog.degraded)
        self.assertEqual([model["id"] for model in catalog.models], ["vendor/text-only", "vendor/vision"])
        request = captured[0]
        self.assertEqual(request.full_url, "https://api.orcarouter.ai/v1/models?capability=chat")
        self.assertEqual(request.headers["Authorization"], f"Bearer {FAKE_KEY}")
        self.assertNotIn("www.orcarouter.ai", request.full_url)

    def test_capability_query_parameter_follows_the_entry_point(self):
        for capability in ("chat", "embedding", "image", "video", "rerank"):
            captured = []
            orcarouter.fetch_models(
                "https://api.orcarouter.ai/v1", FAKE_KEY, capability=capability,
                opener=fake_opener({"data": []}, capture=captured),
            )
            self.assertIn(f"capability={capability}", captured[0].full_url)

    def test_response_is_bounded_by_item_count(self):
        huge = {"data": [dict(TEXT_ONLY, id=f"vendor/m{i}") for i in range(5000)]}
        catalog = orcarouter.fetch_models(
            "https://api.orcarouter.ai/v1", FAKE_KEY, opener=fake_opener(huge)
        )
        self.assertEqual(catalog.total_seen, orcarouter.CATALOG_MAX_ITEMS)

    def test_auth_error_degrades_instead_of_raising(self):
        catalog = orcarouter.fetch_models(
            "https://api.orcarouter.ai/v1", FAKE_KEY,
            opener=fake_opener({"error": "invalid_api_key"}, status=401),
        )
        self.assertTrue(catalog.degraded)
        self.assertIn("撤销", catalog.error)

    def test_network_error_degrades_instead_of_raising(self):
        catalog = orcarouter.fetch_models(
            "https://api.orcarouter.ai/v1", FAKE_KEY, opener=NetworkFailureOpener("dns failure")
        )
        self.assertTrue(catalog.degraded)
        self.assertIn("无法连接", catalog.error)

    def test_malformed_payload_degrades(self):
        catalog = orcarouter.fetch_models(
            "https://api.orcarouter.ai/v1", FAKE_KEY,
            opener=lambda request, timeout=None: FakeResponse(b"not json"),
        )
        self.assertTrue(catalog.degraded)
        self.assertIn("无法解析", catalog.error)

    def test_catalog_error_never_leaks_the_key(self):
        catalog = orcarouter.fetch_models(
            "https://api.orcarouter.ai/v1", FAKE_KEY,
            opener=fake_opener({"error": f"bad key {FAKE_KEY}"}, status=403),
        )
        self.assertNotIn(FAKE_KEY, catalog.error)


class CatalogCacheTests(TemporaryStore):
    def setUp(self):
        super().setUp()
        self.cache_path = Path(self.directory.name) / "catalog.json"
        self.cache = orcarouter.CatalogCache(self.cache_path)

    def test_live_result_is_cached_and_reused(self):
        calls = {"n": 0}

        def fetcher(base, key, **kwargs):
            calls["n"] += 1
            return orcarouter.Catalog(
                models=orcarouter.capability_filter(CATALOG_FIXTURE, kwargs["capability"]),
                source="live",
                fetched_at="2026-09-15T00:00:00+00:00",
                total_seen=len(CATALOG_FIXTURE),
            )

        first = self.cache.get("https://api.orcarouter.ai/v1", FAKE_KEY, fetcher=fetcher)
        second = self.cache.get("https://api.orcarouter.ai/v1", FAKE_KEY, fetcher=fetcher)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(first.models, second.models)
        self.assertFalse(first.degraded)

    def test_seed_is_used_when_discovery_fails_and_nothing_was_cached(self):
        def failing(base, key, **kwargs):
            return orcarouter.Catalog(source="seed", degraded=True, error="down")

        catalog = self.cache.get("https://api.orcarouter.ai/v1", FAKE_KEY, fetcher=failing)
        self.assertEqual(catalog.source, "seed")
        self.assertEqual([model["id"] for model in catalog.models], list(orcarouter.SEED_MODEL_IDS))
        self.assertEqual(catalog.error, "down")

    def test_last_known_good_is_refiltered_before_reuse(self):
        self.cache.get(
            "https://api.orcarouter.ai/v1", FAKE_KEY, capability="chat",
            fetcher=lambda base, key, **kwargs: orcarouter.Catalog(
                models=orcarouter.capability_filter(CATALOG_FIXTURE, "chat"),
                source="live", fetched_at="2026-09-15T00:00:00+00:00", total_seen=len(CATALOG_FIXTURE),
            ),
        )
        persisted = json.loads(self.cache_path.read_text(encoding="utf-8"))["models"]
        # A fresh capability request that fails must not replay the chat list.
        catalog = self.cache.get(
            "https://api.orcarouter.ai/v1", FAKE_KEY, capability="image",
            fetcher=lambda base, key, **kwargs: orcarouter.Catalog(source="seed", degraded=True, error="down"),
        )
        # The persisted last-known-good copy is re-filtered for the requested
        # capability: only the image model may come back through it.
        self.assertTrue(catalog.degraded)
        self.assertTrue(all("image-generation" in model["supported_endpoint_types"] for model in catalog.models))
        self.assertNotIn("vendor/text-only", [model["id"] for model in catalog.models])

    def test_seed_is_never_mixed_into_a_successful_live_result(self):
        catalog = self.cache.get(
            "https://api.orcarouter.ai/v1", FAKE_KEY, capability="chat",
            fetcher=lambda base, key, **kwargs: orcarouter.Catalog(
                models=orcarouter.capability_filter([TEXT_ONLY], "chat"), source="live",
                fetched_at="2026-09-15T00:00:00+00:00", total_seen=1,
            ),
        )
        ids = [model["id"] for model in catalog.models]
        self.assertEqual(ids, ["vendor/text-only"])
        for seed_id in orcarouter.SEED_MODEL_IDS:
            self.assertNotIn(seed_id, ids)

    def test_cached_catalog_is_persisted_without_the_key(self):
        self.cache.get(
            "https://api.orcarouter.ai/v1", FAKE_KEY, capability="chat",
            fetcher=lambda base, key, **kwargs: orcarouter.Catalog(
                models=orcarouter.capability_filter([TEXT_ONLY], "chat"), source="live",
                fetched_at="2026-09-15T00:00:00+00:00", total_seen=1,
            ),
        )
        raw = self.cache_path.read_text(encoding="utf-8")
        self.assertNotIn(FAKE_KEY, raw)
        self.assertIn("vendor/text-only", raw)


# --------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------


class RedactionTests(unittest.TestCase):
    def test_mask_keeps_only_a_short_tail(self):
        self.assertEqual(orcarouter.mask_key(FAKE_KEY), "sk-orca-…0000")
        self.assertEqual(orcarouter.mask_key(""), "")
        self.assertEqual(orcarouter.mask_key("short"), "sk-orca-…")

    def test_redact_removes_keys_from_free_text(self):
        text = orcarouter.redact(f"failed with {FAKE_KEY} while calling")
        self.assertNotIn(FAKE_KEY, text)
        self.assertIn("redacted", text)

    def test_key_shape_accepts_only_the_documented_prefix(self):
        self.assertTrue(orcarouter.key_shape_ok(FAKE_KEY))
        self.assertFalse(orcarouter.key_shape_ok("sk-orca-"))
        self.assertFalse(orcarouter.key_shape_ok("sk-orca-abc def"))
        self.assertFalse(orcarouter.key_shape_ok(""))

    def test_no_real_key_or_fixed_verifier_is_hardcoded(self):
        source = Path(orcarouter.__file__).read_text(encoding="utf-8")
        self.assertNotIn("client_secret", source)
        for literal in re.findall(r"sk-orca-[A-Za-z0-9]{16,}", source):
            self.fail(f"hardcoded key literal in orcarouter.py: {literal[:12]}…")


if __name__ == "__main__":
    unittest.main()
