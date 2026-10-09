from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import orcarouter
from enrollment import (
    bank_summary,
    enroll_automatic,
    request_completion,
    resolve_orcarouter_credential,
    test_automatic,
)
from fingerprint import analyze_global_outputs, generate_challenges, load_bank, parse_numbers
from bank_builder import build_bank, read_rows


app = Flask(__name__)
PROJECT = Path(__file__).resolve().parent
CUSTOM_BANKS_FILE = PROJECT / "data" / "custom_banks.json"
UNIFIED_BANK_FILE = PROJECT / "data" / "unified_bank.json"
DEFAULT_BANK_ID = "claude"

# One process-wide credential store, catalog cache and connect manager: the
# login lock and the discovery cache must be shared by every request.
CREDENTIAL_STORE = orcarouter.CredentialStore()
CATALOG_CACHE = orcarouter.CatalogCache()
CONNECT_MANAGER = orcarouter.ConnectManager(store=CREDENTIAL_STORE)


def provider_ids() -> tuple[str, ...]:
    return (orcarouter.CUSTOM_PROVIDER_ID, *orcarouter.ORCAROUTER_PROVIDER_IDS)


def request_provider(payload: dict) -> dict:
    """Resolve provider + credential for one AI entry point.

    This is the only place that knows how a credential was obtained. The
    inference functions below receive a plain key, so OrcaRouter behaves the
    same whether it came from the API-key adapter or the PKCE adapter.
    """
    provider_id = str(payload.get("provider_id") or orcarouter.CUSTOM_PROVIDER_ID).strip()
    if provider_id not in provider_ids():
        raise ValueError(f"未知的服务提供方：{provider_id}")

    model = str(payload.get("api_model") or "").strip()
    if not model:
        raise ValueError("请先选择模型")

    if provider_id == orcarouter.CUSTOM_PROVIDER_ID:
        base_url = str(payload.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("请填写 Base URL")
        return {
            "provider_id": provider_id,
            "base_url": base_url,
            "api_key": str(payload.get("api_key") or ""),
            "credential": None,
            "model": model,
        }

    credential = resolve_orcarouter_credential(
        provider_id, payload.get("api_key"), store=CREDENTIAL_STORE
    )
    return {
        "provider_id": provider_id,
        "base_url": orcarouter.api_base(),
        "api_key": credential.key,
        "credential": credential,
        "model": model,
    }


def credential_state() -> dict:
    """Browser-safe credential status. Never contains the key itself."""
    stored = CREDENTIAL_STORE.read()
    if stored is not None:
        state = stored.describe()
        state["configured"] = not stored.needs_reauth
        state["has_stored_key"] = True
        return state
    env_key = orcarouter.normalize_key(os.environ.get("ORCAROUTER_API_KEY"))
    if env_key:
        return {
            "configured": True,
            "source": "env",
            "masked": orcarouter.mask_key(env_key),
            "account_id": None,
            "scope": None,
            "generation": 0,
            "needs_reauth": False,
            "created_at": None,
            "has_stored_key": False,
        }
    return {
        "configured": False,
        "source": None,
        "masked": "",
        "account_id": None,
        "scope": None,
        "generation": 0,
        "needs_reauth": False,
        "created_at": None,
        "has_stored_key": False,
    }


def open_authorize_url(url: str) -> bool:
    """Open the consent screen in the user's browser, like start.py does."""
    def launch() -> None:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a missing browser must not fail the request
            pass

    threading.Thread(target=launch, name="orca-open-browser", daemon=True).start()
    return True


def builtin_configs() -> dict[str, dict]:
    return {
        "gpt": {
            "label": "GPT",
            "bank_file": PROJECT / "data" / "gpt_bank.json",
            "data_file": PROJECT / "data" / "gpt_reference.jsonl",
        },
        "claude": {
            "label": "Claude",
            "bank_file": PROJECT / "data" / "claude_bank.json",
            "data_file": PROJECT / "data" / "claude_reference.jsonl",
        },
    }


def load_configs() -> dict[str, dict]:
    configs = builtin_configs()
    if CUSTOM_BANKS_FILE.exists():
        for item in json.loads(CUSTOM_BANKS_FILE.read_text(encoding="utf-8")):
            bank_id = item["id"]
            configs[bank_id] = {
                "label": item["label"],
                "bank_file": PROJECT / "data" / f"{bank_id}_bank.json",
                "data_file": PROJECT / "data" / f"{bank_id}_reference.jsonl",
                "custom": True,
            }
    return configs


def save_custom_configs() -> None:
    items = [
        {"id": bank_id, "label": config["label"]}
        for bank_id, config in BANK_CONFIGS.items()
        if config.get("custom")
    ]
    CUSTOM_BANKS_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_configured_bank(bank_id: str) -> dict | None:
    config = BANK_CONFIGS[bank_id]
    if not config["bank_file"].exists():
        return None
    bank = load_bank(config["bank_file"])
    bank["family_name"] = config["label"]
    return bank


BANK_CONFIGS = load_configs()
banks = {bank_id: load_configured_bank(bank_id) for bank_id in BANK_CONFIGS}


def active_banks() -> dict[str, dict]:
    return {
        bank_id: bank
        for bank_id, bank in banks.items()
        if bank is not None and bank.get("models")
    }


def global_reference_rows() -> list[dict]:
    rows = []
    for bank_id in active_banks():
        config = BANK_CONFIGS[bank_id]
        rows.extend(
            {
                **row,
                "family_id": bank_id,
                "family_name": config["label"],
            }
            for row in read_rows(config["data_file"])
        )
    return rows


def rebuild_global_bank() -> dict:
    global unified_bank
    unified_bank = build_bank(global_reference_rows())
    UNIFIED_BANK_FILE.write_text(
        json.dumps(unified_bank, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return unified_bank


unified_bank = load_bank(UNIFIED_BANK_FILE) if UNIFIED_BANK_FILE.exists() else None
if unified_bank is None:
    rebuild_global_bank()


def requested_bank_id(payload: dict | None = None) -> str:
    bank_id = (payload or {}).get("bank_id") or request.args.get("bank_id") or DEFAULT_BANK_ID
    if bank_id not in BANK_CONFIGS:
        raise ValueError(f"未知指纹库：{bank_id}")
    return bank_id


def requested_temperature(payload: dict) -> float | None:
    value = payload.get("temperature")
    return None if value in (None, "") else float(value)


def summarized_bank(bank_id: str) -> dict:
    config = BANK_CONFIGS[bank_id]
    bank = banks.get(bank_id)
    summary = bank_summary(bank) if bank is not None else {
        "model_count": 0,
        "response_count": 0,
        "number_count": 0,
        "models": [],
        "calibration": {},
    }
    summary.update({"id": bank_id, "label": config["label"]})
    return summary


def summarized_unified_bank() -> dict:
    summaries = {bank_id: summarized_bank(bank_id) for bank_id in active_banks()}
    return {
        "id": "unified",
        "label": "全部指纹",
        "model_count": sum(item["model_count"] for item in summaries.values()),
        "response_count": sum(item["response_count"] for item in summaries.values()),
        "family_count": len(summaries),
        "families": [
            {"id": bank_id, "label": item["label"], "model_count": item["model_count"]}
            for bank_id, item in summaries.items()
        ],
    }


def replace_bank(bank_id: str) -> None:
    banks[bank_id] = load_configured_bank(bank_id)
    rebuild_global_bank()


@app.get("/")
def index():
    summaries = {bank_id: summarized_bank(bank_id) for bank_id in BANK_CONFIGS}
    return render_template(
        "index.html",
        banks=summaries,
        bank=summaries[DEFAULT_BANK_ID],
        unified=summarized_unified_bank(),
        default_bank_id=DEFAULT_BANK_ID,
    )


@app.get("/api/challenges")
def challenges():
    return jsonify({"challenges": generate_challenges(3)})


@app.post("/api/analyze")
def analyze():
    try:
        payload = request.get_json()
        result = analyze_global_outputs(payload["outputs"], unified_bank)
        result["bank"] = summarized_unified_bank()
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/test/auto")
def automatic_test():
    payload = request.get_json()
    try:
        context = request_provider(payload)
        result = test_automatic(
            base_url=context["base_url"],
            api_key=context["api_key"],
            api_model=context["model"],
            temperature=requested_temperature(payload),
            bank=unified_bank,
            api_format="auto",
            provider_id=context["provider_id"],
            credential=context["credential"],
        )
        result["bank"] = summarized_unified_bank()
        result["credential"] = credential_state()
        return jsonify(result)
    except (ValueError, orcarouter.OrcaRouterError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/test/probe")
def automatic_test_probe():
    payload = request.get_json()
    try:
        context = request_provider(payload)
        text = request_completion(
            base_url=context["base_url"],
            api_key=context["api_key"],
            api_model=context["model"],
            prompt=payload["prompt"],
            temperature=requested_temperature(payload),
            api_format="auto",
            provider_id=context["provider_id"],
            credential=context["credential"],
        )
        expected_count = int(payload["expected_count"])
        parsed_numbers = len(parse_numbers(text))
        minimum_numbers = max(80, math.ceil(expected_count * 0.55))
        return jsonify(
            {
                "text": text,
                "parsed_numbers": parsed_numbers,
                "minimum_numbers": minimum_numbers,
                "accepted": parsed_numbers >= minimum_numbers,
                "credential": credential_state(),
            }
        )
    except (ValueError, orcarouter.OrcaRouterError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:  # noqa: BLE001 - upstream failures are reported as-is
        return jsonify({"error": orcarouter.redact(str(error))}), 502


@app.get("/api/bank")
def get_bank():
    try:
        return jsonify(summarized_bank(requested_bank_id()))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/banks")
def get_banks():
    return jsonify({bank_id: summarized_bank(bank_id) for bank_id in BANK_CONFIGS})


@app.post("/api/banks")
def create_bank():
    payload = request.get_json()
    label = payload["label"].strip()
    if not label:
        return jsonify({"error": "请输入指纹库名称"}), 400
    bank_id = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or f"bank-{secrets.token_hex(3)}"
    if bank_id in BANK_CONFIGS:
        return jsonify({"error": "同名指纹库已存在"}), 400
    config = {
        "label": label,
        "bank_file": PROJECT / "data" / f"{bank_id}_bank.json",
        "data_file": PROJECT / "data" / f"{bank_id}_reference.jsonl",
        "custom": True,
    }
    config["data_file"].parent.mkdir(parents=True, exist_ok=True)
    config["data_file"].touch()
    BANK_CONFIGS[bank_id] = config
    banks[bank_id] = None
    save_custom_configs()
    return jsonify(
        {
            "bank": summarized_bank(bank_id),
            "banks": {item: summarized_bank(item) for item in BANK_CONFIGS},
            "unified": summarized_unified_bank(),
        }
    )


@app.post("/api/enroll/auto")
def automatic_enrollment():
    payload = request.get_json()
    try:
        context = request_provider(payload)
        bank_id = requested_bank_id(payload)
        config = BANK_CONFIGS[bank_id]
        result = enroll_automatic(
            base_url=context["base_url"],
            api_key=context["api_key"],
            api_model=context["model"],
            model_label=payload["model_label"].strip(),
            sample_count=int(payload.get("sample_count", 36)),
            temperature=requested_temperature(payload),
            api_format="auto",
            data_file=config["data_file"],
            bank_file=config["bank_file"],
            bank_id=bank_id,
            provider="api",
            provider_id=context["provider_id"],
            credential=context["credential"],
        )
        replace_bank(bank_id)
        result["bank"] = summarized_bank(bank_id)
        result["unified"] = summarized_unified_bank()
        result["credential"] = credential_state()
        return jsonify(result)
    except (RuntimeError, ValueError, orcarouter.OrcaRouterError) as error:
        return jsonify({"error": orcarouter.redact(str(error))}), 400


@app.get("/api/providers")
def get_providers():
    """Provider registry: OrcaRouter is two named entries, not a custom base URL."""
    return jsonify(
        {
            "providers": orcarouter.provider_registry(),
            "credential": credential_state(),
        }
    )


@app.post("/api/orcarouter/models")
def orcarouter_models():
    """Capability-filtered model list for one AI entry point.

    The key never reaches the browser: the backend holds it and returns only
    minimal model metadata.
    """
    payload = request.get_json(silent=True) or {}
    capability = str(payload.get("capability") or "chat").strip()
    modality = payload.get("modality")
    modality = str(modality).strip() if modality else None
    if capability not in orcarouter.CAPABILITY_ENDPOINT_TYPES:
        return jsonify({"error": f"未知的模型能力：{capability}"}), 400
    try:
        provider_id = str(payload.get("provider_id") or orcarouter.PROVIDER_ID).strip()
        if provider_id not in orcarouter.ORCAROUTER_PROVIDER_IDS:
            raise ValueError(f"未知的服务提供方：{provider_id}")
        credential = resolve_orcarouter_credential(
            provider_id, payload.get("api_key"), store=CREDENTIAL_STORE
        )
    except (ValueError, orcarouter.OrcaRouterError) as error:
        return jsonify({"error": str(error), "models": [], "degraded": True, "source": "none"}), 400

    catalog = CATALOG_CACHE.get(
        orcarouter.api_base(),
        credential.key,
        capability=capability,
        modality=modality,
        force=bool(payload.get("refresh")),
    )
    body = catalog.public()
    body["capability"] = capability
    body["modality"] = modality
    body["credential"] = credential_state()
    return jsonify(body)


@app.get("/api/orcarouter/credential")
def orcarouter_credential():
    return jsonify(credential_state())


@app.post("/api/orcarouter/credential")
def save_orcarouter_credential():
    """API-key adapter: paste an existing sk-orca-... key."""
    payload = request.get_json(silent=True) or {}
    key = orcarouter.normalize_key(payload.get("api_key"))
    if not orcarouter.key_shape_ok(key):
        return jsonify({
            "error": "API Key 格式不正确，应以 sk-orca- 开头",
            "credential": credential_state(),
        }), 400
    stored = CREDENTIAL_STORE.save(key, source="api_key")
    return jsonify({"credential": stored.describe(), "message": "API Key 已保存（仅保存在本机）"})


@app.delete("/api/orcarouter/credential")
def clear_orcarouter_credential():
    CREDENTIAL_STORE.clear()
    return jsonify({"credential": credential_state(), "message": "已清除本机保存的 OrcaRouter 凭据"})


@app.post("/api/orcarouter/connect/start")
def orcarouter_connect_start():
    """PKCE adapter, step 1: build the S256 authorize URL for a fresh verifier."""
    payload = request.get_json(silent=True) or {}
    try:
        attempt = CONNECT_MANAGER.start(app_name=orcarouter.APP_NAME)
    except orcarouter.OrcaRouterError as error:
        return jsonify({"error": str(error)}), 400
    attempt["app_name"] = orcarouter.APP_NAME
    attempt["dashboard_url"] = orcarouter.DASHBOARD_URL
    attempt["logo_url"] = orcarouter.LOGO_URL
    if payload.get("open_browser"):
        open_authorize_url(attempt["authorize_url"])
    return jsonify(attempt)


@app.get("/api/orcarouter/connect/status")
def orcarouter_connect_status():
    attempt_id = request.args.get("attempt", type=int)
    if attempt_id is None:
        return jsonify({"error": "缺少 attempt 参数"}), 400
    try:
        return jsonify(CONNECT_MANAGER.status(attempt_id))
    except orcarouter.ConnectError as error:
        return jsonify({"error": str(error), "code": error.code}), 404


@app.post("/api/orcarouter/connect/code")
def orcarouter_connect_code():
    """Exchange the authorization code (from the loopback redirect or pasted)."""
    payload = request.get_json(silent=True) or {}
    attempt_id = payload.get("attempt")
    code = str(payload.get("code") or "")
    if attempt_id is None:
        return jsonify({"error": "缺少 attempt 参数"}), 400
    try:
        result = CONNECT_MANAGER.submit_code(int(attempt_id), code)
    except orcarouter.ConnectError as error:
        return jsonify({"error": str(error), "code": error.code}), 400
    result["credential"] = credential_state()
    return jsonify(result)


@app.post("/api/orcarouter/connect/cancel")
def orcarouter_connect_cancel():
    """Release the login lock: explicit cancel, modal close, provider switch."""
    payload = request.get_json(silent=True) or {}
    attempt_id = payload.get("attempt")
    if attempt_id is None:
        return jsonify({"cancelled": False, "reason": "missing_attempt"}), 400
    try:
        result = CONNECT_MANAGER.cancel(int(attempt_id))
    except orcarouter.ConnectError as error:
        return jsonify({"cancelled": False, "error": str(error), "code": error.code}), 404
    result["cancelled"] = True
    return jsonify(result)


@app.post("/api/orcarouter/reauth")
def orcarouter_reauth():
    """Terminal reauthentication: mark the exact rejected credential generation."""
    payload = request.get_json(silent=True) or {}
    generation = int(payload.get("generation") or 0)
    account_id = payload.get("account_id")
    applied = CREDENTIAL_STORE.mark_needs_reauth(account_id, generation)
    return jsonify({"applied": applied, "credential": credential_state()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
