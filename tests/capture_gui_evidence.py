"""Generate the OrcaRouter GUI evidence: manifest plus real UI screenshots.

Runs the actual ModelTrace web app (Flask + the real provider modules), drives it
with Playwright/Chromium, and writes ``orca-evidence/manifest.json`` together
with the screenshots it describes. The OrcaRouter credential used for the live
catalog comes from ``ORCAROUTER_API_KEY`` in the environment; it is saved through
the real API-key adapter into a throwaway credential file, and the UI only ever
renders its masked projection.

    ORCAROUTER_API_KEY=… python tests/capture_gui_evidence.py

The script never prints, stores or screenshots the raw key.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "orca-evidence"
# The authoritative chat catalog: the capability filter is applied by the
# discovery path itself, so the screenshot and its model count describe exactly
# the list the user can choose from.
CATALOG_URL = "https://api.orcarouter.ai/v1/models?capability=chat"
VIEWPORT = {"width": 1360, "height": 900}
MIN_WIDTH, MIN_HEIGHT = 800, 450
MIN_BYTES = 10_000
CHROMIUM = os.environ.get("MODELTRACE_CHROMIUM", "/usr/bin/chromium")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def wait_for_server(base: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/api/providers", timeout=2):
                return
        except Exception:  # noqa: BLE001 - server not up yet
            time.sleep(0.3)
    raise RuntimeError("ModelTrace did not start")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"{path.name} is not a valid PNG")
    return struct.unpack(">II", header[16:24])


def screenshot(kind: str, shows: str, ui: dict) -> dict:
    """Describe one artifact with the fields the evidence gate reads back."""
    path = EVIDENCE / f"{kind}.png"
    width, height = png_size(path)
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise RuntimeError(f"{path.name} is smaller than {MIN_WIDTH}x{MIN_HEIGHT}")
    if path.stat().st_size < MIN_BYTES:
        raise RuntimeError(f"{path.name} is too small to be a real screenshot")
    return {
        "kind": kind,
        "path": path.name,
        "sha256": sha256(path),
        "width": width,
        "height": height,
        "shows": shows,
        "ui": ui,
    }


def check_manifest(manifest: dict) -> None:
    """Re-read the manifest the way the independent evidence gate does.

    The screenshots are the only proof that the real UI was driven, so a run
    that produces a manifest the gate cannot read must fail here instead of at
    submission time.
    """
    automation = manifest["automation"]
    if automation["framework"] not in ("playwright", "cypress"):
        raise RuntimeError("evidence must come from Playwright or Cypress")
    if automation["passed"] is not True:
        raise RuntimeError("evidence automation did not pass")
    if automation["catalog_source"] != CATALOG_URL:
        raise RuntimeError("evidence did not use the authoritative chat catalog")
    total = automation["catalog_model_count"]
    image = automation["image_model_count"]
    if not isinstance(total, int) or not isinstance(image, int) or not 0 <= image <= total:
        raise RuntimeError("evidence manifest has invalid model counts")
    by_kind = {item["kind"]: item for item in manifest["artifacts"]}
    for kind in ("auth-methods", "text-model-dropdown"):
        item = by_kind.get(kind)
        if not item:
            raise RuntimeError(f"missing required screenshot: {kind}")
        path = EVIDENCE / item["path"]
        if path.parent.resolve() != EVIDENCE.resolve():
            raise RuntimeError(f"evidence path escapes directory: {item['path']}")
        if not path.is_file():
            raise RuntimeError(f"evidence screenshot is missing: {path}")
        width, height = png_size(path)
        if width < MIN_WIDTH or height < MIN_HEIGHT or path.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"evidence screenshot is unusable: {path}")
        if item["sha256"] != sha256(path):
            raise RuntimeError(f"evidence checksum mismatch: {path}")
        ui = item["ui"]
        if kind == "auth-methods":
            for field in ("api_key_visible", "pkce_visible", "secret_masked", "controls_enabled"):
                if ui.get(field) is not True:
                    raise RuntimeError("evidence does not show usable API Key and PKCE auth")
        else:
            if ui.get("dropdown_open") is not True or ui.get("item_count") != total:
                raise RuntimeError(f"evidence does not show the required open dropdown: {kind}")
            if ui.get("opaque_background") is not True or ui.get("visible_border") is not True:
                raise RuntimeError(f"evidence dropdown has no visible container: {kind}")
            delta = ui.get("trigger_panel_right_delta")
            if not isinstance(delta, (int, float)) or abs(delta) > 2:
                raise RuntimeError(f"evidence dropdown is not aligned to its trigger: {kind}")


def main() -> int:
    key = (os.environ.get("ORCAROUTER_API_KEY") or "").strip()
    if not key:
        print("ORCAROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    EVIDENCE.mkdir(exist_ok=True)
    directory = tempfile.TemporaryDirectory()
    credential_file = Path(directory.name) / "orcarouter.json"
    catalog_file = Path(directory.name) / "catalog.json"
    port = free_port()
    base = f"http://127.0.0.1:{port}"

    env = {
        **os.environ,
        "ORCAROUTER_API_KEY": key,
        "MODELTRACE_CREDENTIAL_FILE": str(credential_file),
        "MODELTRACE_CATALOG_FILE": str(catalog_file),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    server = subprocess.Popen(
        [sys.executable, "-c", f"import app; app.app.run(host='127.0.0.1', port={port}, debug=False)"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(base)
        # Credential through the real API-key adapter (never through a fixture file).
        saved = post_json(f"{base}/api/orcarouter/credential", {"api_key": key})
        masked = saved["credential"]["masked"]
        if key in json.dumps(saved):
            raise RuntimeError("credential endpoint echoed the raw key")
        catalog = post_json(
            f"{base}/api/orcarouter/models", {"provider_id": "orcarouter", "capability": "chat"}
        )
        multimodal = post_json(
            f"{base}/api/orcarouter/models",
            {"provider_id": "orcarouter", "capability": "chat", "modality": "image"},
        )
        if catalog.get("source") != "live":
            raise RuntimeError(f"live catalog unavailable: {catalog.get('error')}")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=CHROMIUM, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            page.goto(base, wait_until="networkidle")
            # Open the API-test workspace where the provider selector lives.
            page.click("[data-test-mode=api]")
            page.wait_for_selector("#test-api.active")

            # --- auth methods -------------------------------------------------
            # Real interaction: choose the OrcaRouter provider, which reveals the
            # credential row, then open the connect dialog from it.
            page.select_option("#test-provider", "orcarouter")
            page.wait_for_selector("[data-model-field=orcarouter]:not([hidden])")
            page.wait_for_selector("#test-orca-credential:not([hidden])")
            page.click("#test-orca-credential [data-orca-open-connect]")
            page.wait_for_selector("#orca-connect-dialog[open] .orca-methods")
            page.wait_for_function(
                "document.querySelector('#orca-credential-state').textContent.includes('sk-orca-')"
            )
            api_key_visible = page.is_visible("[data-auth-method=api_key] #orca-api-key")
            pkce_visible = page.is_visible("[data-auth-method=pkce] [data-orca-start-login]")
            masked_text = page.inner_text("#orca-credential-state")
            secret_masked = "sk-orca-…" in masked_text and key not in masked_text
            controls_enabled = page.eval_on_selector_all(
                "[data-orca-connect-action]",
                "els => els.length > 0 && els.every(el => !el.disabled)",
            )
            page.screenshot(path=str(EVIDENCE / "auth-methods.png"))

            # --- text model dropdown -----------------------------------------
            page.click("[data-orca-close-connect]")
            page.wait_for_selector("#orca-connect-dialog", state="hidden")
            page.click("[data-orca-model-select=test] .orca-select-trigger")
            try:
                page.wait_for_selector(
                    "[data-orca-model-select=test] .orca-select-panel:not([hidden]) li[data-model-option]",
                    timeout=15000,
                )
            except Exception:
                print(
                    "dropdown diagnostics:",
                    page.evaluate(
                        """() => {
                          const panel = document.querySelector('[data-orca-model-select=test] .orca-select-panel');
                          const catalog = window.OrcaRouterProvider.state.catalog;
                          return JSON.stringify({
                            panelHidden: panel ? panel.hidden : null,
                            items: document.querySelectorAll('[data-orca-model-select=test] li[data-model-option]').length,
                            status: document.querySelector('[data-orca-model-select=test] .orca-select-status').textContent,
                            catalog: catalog ? { source: catalog.source, models: catalog.models.length, capability: catalog.capability, error: catalog.error } : null,
                          });
                        }"""
                    ),
                    file=sys.stderr,
                )
                raise
            dropdown_open = page.is_visible("[data-orca-model-select=test] .orca-select-panel")
            item_count = page.eval_on_selector_all(
                "[data-orca-model-select=test] li[data-model-option]", "els => els.length"
            )
            styles = page.evaluate(
                """() => {
                  const panel = document.querySelector('[data-orca-model-select=test] .orca-select-panel');
                  const trigger = document.querySelector('[data-orca-model-select=test] .orca-select-trigger');
                  const bg = getComputedStyle(panel).backgroundColor;
                  const border = getComputedStyle(panel).borderTopWidth;
                  const p = panel.getBoundingClientRect(), t = trigger.getBoundingClientRect();
                  return { background: bg, borderWidth: border, panelRight: p.right, triggerRight: t.right,
                           panelWidth: p.width, panelHeight: p.height };
                }"""
            )
            alpha = styles["background"].replace(" ", "")
            if alpha.startswith("rgb("):
                opaque_background = True
            else:
                opaque_background = not alpha.startswith("rgba(") or float(alpha.split(",")[-1].rstrip(")")) == 1.0
            visible_border = float(styles["borderWidth"].replace("px", "")) > 0
            trigger_panel_right_delta = abs(styles["panelRight"] - styles["triggerRight"])
            page.screenshot(path=str(EVIDENCE / "text-model-dropdown.png"))

            # Provider switch must recompute the options, and the generic
            # free-text fields must come back for the custom provider.
            page.select_option("#test-provider", "custom")
            page.wait_for_selector("[data-model-field=custom]:not([hidden])")
            custom_restored = page.is_visible("#test-api-base")
            page.select_option("#test-provider", "orcarouter")
            page.wait_for_selector("[data-model-field=orcarouter]:not([hidden])")
            browser.close()

        chat_models = len(catalog["models"])
        passed = bool(
            api_key_visible and pkce_visible and secret_masked and controls_enabled
            and dropdown_open and item_count == chat_models and opaque_background and visible_border
            and trigger_panel_right_delta <= 2 and custom_restored and not catalog["degraded"]
        )
        manifest = {
            "automation": {
                "script": "tests/capture_gui_evidence.py",
                "description": "python playwright (sync) driving the real Flask app",
                "framework": "playwright",
                "passed": passed,
                "catalog_source": CATALOG_URL,
                "catalog_model_count": chat_models,
                "image_model_count": len(multimodal["models"]),
                "catalog_degraded": bool(catalog["degraded"]),
            },
            "artifacts": [
                screenshot(
                    "auth-methods",
                    "API Key and OAuth 2.0 + PKCE entries side by side, masked credential",
                    {
                        "api_key_visible": bool(api_key_visible),
                        "pkce_visible": bool(pkce_visible),
                        "secret_masked": bool(secret_masked),
                        "controls_enabled": bool(controls_enabled),
                    },
                ),
                screenshot(
                    "text-model-dropdown",
                    "text model dropdown opened from the live OrcaRouter ?capability=chat catalog",
                    {
                        "dropdown_open": bool(dropdown_open),
                        "item_count": int(item_count),
                        "opaque_background": bool(opaque_background),
                        "visible_border": bool(visible_border),
                        "trigger_panel_right_delta": round(trigger_panel_right_delta, 3),
                        "custom_provider_restored": bool(custom_restored),
                    },
                ),
            ],
            "notes": (
                "No multimodal screenshot: ModelTrace has no image/attachment entry point; every AI "
                "call sends a plain numeric-sequence text prompt (challenge_suite.py, fingerprint.py). "
                "The multimodal capability filter is implemented and unit-tested, but there is no "
                "multimodal selector in this UI to photograph."
            ),
        }
        check_manifest(manifest)
        (EVIDENCE / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest["automation"], ensure_ascii=False))
        print("passed:", manifest["automation"]["passed"])
        return 0 if manifest["automation"]["passed"] else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
