"""Live OrcaRouter checks: real catalog and one real inference request.

Skipped unless ``ORCAROUTER_API_KEY`` is present in the environment. The key is
only ever read from the environment; it is never printed, persisted by this test
module or included in an assertion message.

    ORCAROUTER_API_KEY=… python -m unittest tests.test_orcarouter_live -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import enrollment  # noqa: E402
import orcarouter  # noqa: E402
from fingerprint import generate_challenges, parse_numbers  # noqa: E402

LIVE_KEY = orcarouter.normalize_key(os.environ.get("ORCAROUTER_API_KEY"))


@unittest.skipUnless(LIVE_KEY, "ORCAROUTER_API_KEY is not set")
class LiveCatalogTests(unittest.TestCase):
    def test_live_catalog_comes_from_the_official_endpoint(self):
        catalog = orcarouter.fetch_models(orcarouter.api_base(), LIVE_KEY, capability="chat")
        self.assertEqual(catalog.source, "live", msg=f"discovery degraded: {catalog.error}")
        self.assertGreater(len(catalog.models), 0)
        for model in catalog.models:
            # Vendor/model namespaces are preserved verbatim.
            self.assertIn("/", model["id"])
            self.assertTrue(
                set(model["supported_endpoint_types"]) & set(orcarouter.TEXT_ENDPOINT_TYPES)
            )
            self.assertFalse(
                set(model["supported_endpoint_types"]) <= set(orcarouter.NON_TEXT_ENDPOINT_TYPES)
            )

    def test_multimodal_filter_only_returns_declared_image_models(self):
        catalog = orcarouter.fetch_models(
            orcarouter.api_base(), LIVE_KEY, capability="chat", modality="image"
        )
        self.assertEqual(catalog.source, "live", msg=f"discovery degraded: {catalog.error}")
        for model in catalog.models:
            self.assertIn("image", model["input_modalities"])

    def test_live_result_is_never_padded_with_the_seed(self):
        catalog = orcarouter.fetch_models(orcarouter.api_base(), LIVE_KEY, capability="chat")
        if catalog.source != "live":
            self.skipTest(f"catalog unavailable: {catalog.error}")
        ids = {model["id"] for model in catalog.models}
        for seed_id in orcarouter.SEED_MODEL_IDS:
            if seed_id in ids:
                # Only acceptable when the live catalog really advertises it.
                advertised = {
                    model["id"]
                    for model in orcarouter.fetch_models(
                        orcarouter.api_base(), LIVE_KEY, capability="chat"
                    ).models
                }
                self.assertIn(seed_id, advertised)


@unittest.skipUnless(LIVE_KEY, "ORCAROUTER_API_KEY is not set")
class LiveInferenceTests(unittest.TestCase):
    """One real request through the provider code path, not through curl."""

    @classmethod
    def setUpClass(cls):
        catalog = orcarouter.fetch_models(orcarouter.api_base(), LIVE_KEY, capability="chat")
        if catalog.source != "live" or not catalog.models:
            raise unittest.SkipTest(f"catalog unavailable: {catalog.error}")
        cls.catalog = catalog
        cls.model = catalog.models[0]["id"]

    def test_api_key_adapter_and_request_path_reach_the_relay(self):
        credential = orcarouter.resolve_credential(
            "api_key", explicit_key=LIVE_KEY, environ={}
        )
        self.assertEqual(credential.source, "api_key")
        # An explicit key is never written to disk by this path.
        store = orcarouter.CredentialStore(Path(os.devnull))
        self.assertIsNone(store.read())

        # A workspace key may be scoped to a subset of the catalog, so try the
        # advertised models in order and use the first one it may actually call.
        challenge = generate_challenges(1)[0]
        attempted = []
        for model in self.catalog.models:
            try:
                text = enrollment.request_completion(
                    base_url="",
                    api_key=credential.key,
                    api_model=model["id"],
                    prompt=challenge["prompt"],
                    temperature=None,
                    api_format="openai",
                    provider_id=orcarouter.PROVIDER_ID,
                    credential=credential,
                )
            except RuntimeError as error:
                attempted.append(f"{model['id']}: {error}")
                continue
            self.assertIsInstance(text, str)
            self.assertGreater(len(text.strip()), 0)
            self.assertGreater(len(parse_numbers(text)), 0)
            return
        self.fail(
            "no advertised chat model accepted this workspace key through the "
            f"provider path; attempts: {'; '.join(attempted[:5])}"
        )

    def test_provider_configuration_supplies_the_relay_origin(self):
        target = enrollment.resolve_provider(orcarouter.PROVIDER_ID, {})
        self.assertEqual(target["base_url"], "https://api.orcarouter.ai/v1")
        self.assertEqual(target["auth_method"], "api_key")
        pkce_target = enrollment.resolve_provider(orcarouter.PKCE_PROVIDER_ID, {})
        self.assertEqual(pkce_target["base_url"], "https://api.orcarouter.ai/v1")
        self.assertEqual(pkce_target["auth_method"], "pkce")
        # The authentication origin is never used for inference.
        self.assertNotIn("www.orcarouter.ai", target["base_url"])


if __name__ == "__main__":
    unittest.main()
