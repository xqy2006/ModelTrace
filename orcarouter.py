"""OrcaRouter provider: origins, credentials, OAuth 2.0 + PKCE and model catalog.

ModelTrace talks to OpenAI-compatible endpoints by hand (see ``enrollment.py``).
This module adds OrcaRouter as a first-class provider next to the existing
free-form "custom" endpoint:

* two independent ways to obtain the *same* kind of ``sk-orca-...`` key --
  a pasted API key and an OAuth 2.0 + PKCE browser authorization;
* one credential interface (``resolve_credential``) that both adapters feed, so
  the request path and the catalog path never learn where a key came from;
* bounded model discovery with a small verified fallback seed.

Authentication and inference intentionally use different origins:

* auth / code exchange: ``https://www.orcarouter.ai`` (paths ``/auth`` and
  ``/api/v1/auth/keys``);
* inference / model list: ``https://api.orcarouter.ai/v1``.

Nothing here derives one origin from the other, and
``https://api.orcarouter.ai/v1/auth/keys`` is never used: that path does not
exist on the relay.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable

PROVIDER_ID = "orcarouter"
PKCE_PROVIDER_ID = "orcarouter-oauth"
CUSTOM_PROVIDER_ID = "custom"

PROVIDER_LABELS = {
    PROVIDER_ID: "OrcaRouter - API",
    PKCE_PROVIDER_ID: "OrcaRouter - Auth",
    CUSTOM_PROVIDER_ID: "自定义（OpenAI 兼容）",
}
ORCAROUTER_PROVIDER_IDS = (PROVIDER_ID, PKCE_PROVIDER_ID)

DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
AUTHORIZE_PATH = "/auth"
EXCHANGE_PATH = "/api/v1/auth/keys"

KEY_PREFIX = "sk-orca-"
KEY_PATTERN = re.compile(r"sk-orca-[A-Za-z0-9._\-]{4,}")
DASHBOARD_URL = "https://www.orcarouter.ai/console/authorized-apps"
LOGO_URL = "https://www.orcarouter.ai/orca-logo-classic.png"
APP_NAME = "ModelTrace"

DEFAULT_SCOPE = "api"
ALLOWED_SCOPES = ("api", "connector")
CONSENT_TTL_SECONDS = 600
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

# Catalog limits: a discovery response must never be able to consume unbounded
# memory or advertise capabilities this client cannot speak.
CATALOG_TIMEOUT_SECONDS = 8
CATALOG_MAX_BYTES = 1024 * 1024
CATALOG_MAX_ITEMS = 200
CATALOG_CACHE_SECONDS = 300
CATALOG_FAILURE_CACHE_SECONDS = 15

TEXT_ENDPOINT_TYPES = ("openai", "anthropic", "gemini", "openai-response")
# Endpoint types that only make sense for a dedicated non-text route. A model is
# eligible for the text dropdown only when it also speaks a text endpoint.
NON_TEXT_ENDPOINT_TYPES = (
    "image-generation",
    "openai-video",
    "jina-rerank",
    "embeddings",
    "audio",
    "tts",
    "rerank",
)
CAPABILITY_ENDPOINT_TYPES = {
    "chat": TEXT_ENDPOINT_TYPES,
    "embedding": ("embeddings",),
    "image": ("image-generation",),
    "video": ("openai-video",),
    "rerank": ("jina-rerank",),
}

# Verified cold-start seed. These IDs come from the integration campaign's
# verified evidence set; only the reasoning ladder is asserted (GPT-5.5 was
# verified with the four documented effort levels). Context length and input
# modalities are deliberately left unknown: guessing them from a model name is
# the failure mode the capability filter exists to prevent, and an unknown
# modality fails closed in the multimodal filter instead of advertising a
# capability that cannot be proven.
SEED_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "openai/gpt-5.5",
        "name": "OpenAI: GPT-5.5",
        "supported_endpoint_types": list(TEXT_ENDPOINT_TYPES),
        "reasoning_efforts": ["low", "medium", "high", "xhigh"],
    },
    {
        "id": "anthropic/claude-opus-4.8",
        "name": "Anthropic: Claude Opus 4.8",
        "supported_endpoint_types": list(TEXT_ENDPOINT_TYPES),
    },
    {
        "id": "google/gemini-3.5-flash",
        "name": "Google: Gemini 3.5 Flash",
        "supported_endpoint_types": list(TEXT_ENDPOINT_TYPES),
    },
    {
        "id": "deepseek/deepseek-v4-pro",
        "name": "DeepSeek: DeepSeek V4 Pro",
        "supported_endpoint_types": list(TEXT_ENDPOINT_TYPES),
    },
    {
        "id": "orcarouter/auto",
        "name": "OrcaRouter: Auto",
        "supported_endpoint_types": list(TEXT_ENDPOINT_TYPES),
    },
)
SEED_MODEL_IDS = tuple(model["id"] for model in SEED_MODELS)


class OrcaRouterError(RuntimeError):
    """Base class for provider errors that are safe to show to the user."""


class CredentialUnavailable(OrcaRouterError):
    """No usable OrcaRouter credential is configured for the requested method."""


class ConnectError(OrcaRouterError):
    """An authorization attempt ended without a usable key."""

    def __init__(self, message: str, *, code: str = "error") -> None:
        super().__init__(message)
        self.code = code


class CredentialRejected(RuntimeError):
    """The relay answered 401: terminal reauthentication, never a retry loop.

    Kept as a ``RuntimeError`` so every existing caller keeps working, while the
    wire-format auto detection can stop immediately instead of replaying the
    same rejected credential against a second endpoint.
    """


# --------------------------------------------------------------------------
# origins
# --------------------------------------------------------------------------


def _strip(value: str | None) -> str:
    return (value or "").strip().rstrip("/")


def is_loopback_host(host: str) -> bool:
    return host.lower() in LOOPBACK_HOSTS


def validate_origin(value: str, *, label: str) -> str:
    """Return a normalized origin, rejecting anything but HTTPS (or loopback HTTP)."""
    candidate = _strip(value)
    if not candidate:
        raise OrcaRouterError(f"{label} 未配置")
    parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise OrcaRouterError(f"{label} 必须是完整的 http(s) 地址")
    if parsed.scheme == "http" and not is_loopback_host(parsed.hostname):
        raise OrcaRouterError(f"{label} 只允许对回环地址使用 http，远程地址必须使用 https")
    if parsed.query or parsed.fragment:
        raise OrcaRouterError(f"{label} 不能包含查询串或片段")
    return candidate


def auth_base(environ: dict[str, str] | None = None) -> str:
    """Authentication origin. Explicit override, then shared base, then default."""
    env = environ if environ is not None else os.environ
    candidate = _strip(env.get("ORCA_AUTH_BASE_URL")) or _strip(env.get("ORCA_BASE_URL"))
    return validate_origin(candidate or DEFAULT_AUTH_BASE, label="OrcaRouter 认证地址")


def api_base(environ: dict[str, str] | None = None) -> str:
    """Inference origin, including the API prefix. Explicit override wins."""
    env = environ if environ is not None else os.environ
    candidate = _strip(env.get("ORCA_API_BASE_URL")) or _strip(env.get("ORCA_BASE_URL"))
    return validate_origin(candidate or DEFAULT_API_BASE, label="OrcaRouter 接口地址")


def authorize_url(
    base: str,
    *,
    callback_url: str,
    challenge: str,
    state: str,
    app_name: str = APP_NAME,
    scope: str = DEFAULT_SCOPE,
) -> str:
    """Build the consent URL. S256 is always sent, even for a loopback redirect.

    The consent screen can hand the user a displayed code even when a real
    ``callback_url`` was supplied, and no authorize parameter prevents that, so
    the challenge must never be weaker than S256.
    """
    parts = urllib.parse.urlsplit(validate_origin(base, label="OrcaRouter 认证地址"))
    query = urllib.parse.urlencode(
        {
            "callback_url": callback_url,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "app_name": app_name,
            "scope": scope,
        }
    )
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, AUTHORIZE_PATH, query, ""))


def exchange_url(base: str) -> str:
    parts = urllib.parse.urlsplit(validate_origin(base, label="OrcaRouter 认证地址"))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, EXCHANGE_PATH, "", ""))


# --------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------


def mask_key(key: str | None) -> str:
    """Never render a credential in full; keep only a short tail for support."""
    if not key:
        return ""
    tail = key[-4:] if len(key) > 8 else ""
    return f"{KEY_PREFIX}…{tail}"


def redact(text: str | None) -> str:
    """Strip anything shaped like an OrcaRouter key from user-visible output."""
    if not text:
        return ""
    return KEY_PATTERN.sub(f"{KEY_PREFIX}…redacted", text)


def normalize_key(value: str | None) -> str:
    return (value or "").strip()


def key_shape_ok(value: str) -> bool:
    """Cheap shape check only.

    A matching prefix is not proof of validity: OrcaRouter exposes no
    non-billing validation call, so validity is established by the first real
    request. This only catches obvious paste mistakes.
    """
    key = normalize_key(value)
    return key.startswith(KEY_PREFIX) and len(key) > len(KEY_PREFIX) + 3 and " " not in key


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


@dataclass
class Credential:
    """The single credential shape both adapters produce."""

    key: str
    source: str  # "api_key" | "pkce" | "env"
    account_id: str | None = None
    scope: str | None = None
    generation: int = 0
    needs_reauth: bool = False
    created_at: str | None = None
    # Where this credential came from, so a terminal 401 can flag exactly this
    # stored generation without the request path having to know about storage.
    store: Any = field(default=None, repr=False, compare=False)

    @property
    def masked(self) -> str:
        return mask_key(self.key)

    def describe(self) -> dict[str, Any]:
        """Browser-safe projection: metadata only, never the key itself."""
        return {
            "source": self.source,
            "account_id": self.account_id,
            "scope": self.scope,
            "generation": self.generation,
            "needs_reauth": self.needs_reauth,
            "masked": self.masked,
            "created_at": self.created_at,
        }


def default_credential_path(environ: dict[str, str] | None = None) -> Path:
    """Where the durable key lives.

    The repository commits ``data/`` and keeps ``.gitignore`` out of scope for
    contributions, so a secret must not be written anywhere under the project
    tree. The file therefore lives with the user's other local configuration.
    """
    env = environ if environ is not None else os.environ
    override = (env.get("MODELTRACE_CREDENTIAL_FILE") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".modeltrace" / "orcarouter.json"


def default_catalog_cache_path(environ: dict[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    override = (env.get("MODELTRACE_CATALOG_FILE") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".modeltrace" / "orcarouter_catalog.json"


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class CredentialStore:
    """Persisted OrcaRouter credential with generation-safe reauthentication.

    A PKCE-issued key is durable, but it is *not* a refresh token: there is no
    refresh grant to call, so a revoked key stays stored until a new login
    replaces it. ``mark_needs_reauth`` therefore only ever flags the exact
    account and credential generation that made the rejected request, so a late
    failure from an old request cannot poison a freshly authorized credential.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_credential_path()
        self._lock = threading.Lock()

    def _read_raw(self) -> dict[str, Any] | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or not normalize_key(raw.get("key")):
            return None
        return raw

    def read(self) -> Credential | None:
        with self._lock:
            raw = self._read_raw()
        if raw is None:
            return None
        return Credential(
            key=normalize_key(raw.get("key")),
            source=str(raw.get("source") or "pkce"),
            account_id=raw.get("account_id"),
            scope=raw.get("scope"),
            generation=int(raw.get("generation") or 1),
            needs_reauth=bool(raw.get("needs_reauth")),
            created_at=raw.get("created_at"),
            store=self,
        )

    def save(
        self,
        key: str,
        *,
        source: str,
        account_id: str | None = None,
        scope: str | None = None,
    ) -> Credential:
        key = normalize_key(key)
        if not key:
            raise OrcaRouterError("凭据为空，未保存")
        with self._lock:
            previous = self._read_raw() or {}
            payload = {
                "version": 1,
                "provider": PROVIDER_ID,
                "key": key,
                "source": source,
                "account_id": account_id,
                "scope": scope,
                "needs_reauth": False,
                "generation": int(previous.get("generation") or 0) + 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_private_json(self.path, payload)
        return Credential(
            key=key,
            source=source,
            account_id=account_id,
            scope=scope,
            generation=payload["generation"],
            needs_reauth=False,
            created_at=payload["created_at"],
            store=self,
        )

    def clear(self) -> bool:
        with self._lock:
            try:
                self.path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError:
                return False

    def mark_needs_reauth(self, account_id: str | None, generation: int) -> bool:
        """Flag the rejected credential generation. Returns True when applied.

        The stored secret is deliberately kept: removing it before a successful
        replacement would turn a transient or misclassified 401 into
        irreversible loss of the account. There is no refresh grant, and none is
        emulated here.
        """
        with self._lock:
            raw = self._read_raw()
            if raw is None:
                return False
            if int(raw.get("generation") or 0) != int(generation):
                return False
            stored_account = raw.get("account_id")
            if account_id and stored_account and stored_account != account_id:
                return False
            if raw.get("needs_reauth"):
                return True
            raw["needs_reauth"] = True
            _write_private_json(self.path, raw)
            return True


def resolve_credential(
    auth_method: str,
    explicit_key: str | None = None,
    store: CredentialStore | None = None,
    environ: dict[str, str] | None = None,
) -> Credential:
    """The credential seam shared by both adapters.

    ``auth_method`` is ``"api_key"`` (provider ``orcarouter``) or ``"pkce"``
    (provider ``orcarouter-oauth``). Both return the same ``Credential``; the
    request path and the catalog path only ever receive ``credential.key``.
    """
    env = environ if environ is not None else os.environ
    store = store or CredentialStore()

    if auth_method == "api_key":
        typed = normalize_key(explicit_key)
        if typed:
            if not key_shape_ok(typed):
                raise CredentialUnavailable("API Key 格式不正确，应以 sk-orca- 开头")
            return Credential(key=typed, source="api_key")
        stored = store.read()
        if stored is not None and not stored.needs_reauth:
            return stored
        env_key = normalize_key(env.get("ORCAROUTER_API_KEY"))
        if env_key:
            return Credential(key=env_key, source="env")
        if stored is not None and stored.needs_reauth:
            raise CredentialUnavailable("已保存的 OrcaRouter 凭据已失效，请重新连接或填写 API Key")
        raise CredentialUnavailable("请填写 OrcaRouter API Key，或使用 OAuth 2.0 + PKCE 连接账户")

    if auth_method == "pkce":
        stored = store.read()
        if stored is not None and stored.needs_reauth:
            raise CredentialUnavailable("OrcaRouter 授权已失效或被撤销，请重新连接账户")
        if stored is not None:
            return stored
        raise CredentialUnavailable("尚未连接 OrcaRouter 账户，请先点击“连接 OrcaRouter”")

    raise CredentialUnavailable(f"未知的 OrcaRouter 接入方式：{auth_method}")


def mark_unauthorized(
    credential: Credential | None, store: CredentialStore | None = None
) -> bool:
    """Terminal reauthentication: flag the exact rejected credential generation.

    Called when the relay answers 401. There is no refresh grant to fall back
    on, so the only correct reaction is to stop using this credential and ask
    for a new login -- scoped to the account and generation that was rejected,
    so a late failure cannot invalidate a credential the user has just renewed.
    """
    if credential is None or credential.generation <= 0:
        return False
    target = store or credential.store or CredentialStore()
    return target.mark_needs_reauth(credential.account_id, credential.generation)


# --------------------------------------------------------------------------
# PKCE primitives
# --------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_verifier() -> str:
    """Fresh cryptographic verifier for one attempt. Never logged, never in a URL."""
    return _b64url(secrets.token_bytes(32))


def code_challenge(verifier: str) -> str:
    """base64url(sha256(verifier)) without padding."""
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def generate_state() -> str:
    return _b64url(secrets.token_bytes(16))


def state_matches(expected: str, received: str | None) -> bool:
    """Constant-time comparison of the CSRF token."""
    if not received:
        return False
    return hmac.compare_digest(str(expected), str(received))


def _read_bounded(response: Any, limit: int = CATALOG_MAX_BYTES) -> bytes:
    return response.read(limit + 1)[:limit]


def _safe_error_body(raw: bytes) -> str:
    """Extract only a whitelisted error code/description.

    Error bodies can echo request data, so only ``error``/``error_description``
    are surfaced, and every OrcaRouter-shaped key is redacted.
    """
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(payload[field])
        for field in ("error", "error_description")
        if isinstance(payload.get(field), str) and payload.get(field)
    ]
    return redact(" ".join(parts))[:300]


def exchange_code(
    base: str,
    code: str,
    verifier: str,
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] | None = None,
) -> Credential:
    """Exchange an authorization code for a durable API key.

    Posts to ``{auth_base}/api/v1/auth/keys`` -- never to the relay path.
    """
    target = exchange_url(base)
    body = json.dumps(
        {"code": code, "code_verifier": verifier, "code_challenge_method": "S256"}
    ).encode("utf-8")
    request = urllib.request.Request(
        target,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_NAME,
        },
        method="POST",
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:
            payload = json.loads(_read_bounded(response).decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = _safe_error_body(error.read(4096))
        raise _exchange_error(error.code, detail) from None
    except urllib.error.URLError as error:
        reason = redact(str(getattr(error, "reason", error)))
        raise ConnectError(
            f"无法连接 OrcaRouter 认证服务（{reason}），请检查网络后重试", code="network"
        ) from None
    except (json.JSONDecodeError, ValueError):
        raise ConnectError("OrcaRouter 认证服务返回了无法解析的响应", code="invalid_response") from None

    key = normalize_key(payload.get("key") if isinstance(payload, dict) else None)
    if not key:
        raise ConnectError("OrcaRouter 未返回可用凭据", code="invalid_response")
    scope = payload.get("scope") if isinstance(payload.get("scope"), str) else None
    # Read back the *granted* scope: a downgrade is reported, never assumed away.
    granted = scope if scope in ALLOWED_SCOPES else None
    return Credential(
        key=key,
        source="pkce",
        account_id=str(payload.get("user_id")) if payload.get("user_id") is not None else None,
        scope=granted,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _exchange_error(status: int, detail: str) -> ConnectError:
    suffix = f"（{detail}）" if detail else ""
    if status == 400:
        return ConnectError(
            "授权校验方式不被接受：code_challenge_method 必须为 S256" + suffix,
            code="bad_request",
        )
    if status == 403:
        return ConnectError(
            "授权码无效、已过期（有效期 10 分钟）或已被使用，请重新连接" + suffix, code="denied"
        )
    if status == 429:
        return ConnectError(
            "OrcaRouter 拒绝了本次授权（每用户 24 小时内最多签发 10 个 PKCE 密钥）。"
            "请稍后再试，或改用 API Key 接入" + suffix,
            code="rate_limited",
        )
    if status >= 500:
        return ConnectError(
            f"OrcaRouter 认证服务暂时不可用（HTTP {status}）{suffix}", code="upstream"
        )
    return ConnectError(f"OrcaRouter 授权失败（HTTP {status}）{suffix}", code=f"http_{status}")


# --------------------------------------------------------------------------
# callback page
# --------------------------------------------------------------------------

_CALLBACK_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ModelTrace · OrcaRouter</title>
<style>
body{{margin:0;display:grid;place-items:center;min-height:100vh;background:#f5f6f8;
color:#18212f;font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:440px;padding:28px 30px;background:#fff;border:1px solid #dfe3e8;border-radius:10px;text-align:center}}
h1{{margin:0 0 8px;font-size:17px}}p{{margin:0;color:#667085;font-size:13px;line-height:1.6}}
</style></head>
<body><main><h1>{title}</h1><p>{detail}</p></main></body></html>
"""


def callback_page(title: str, detail: str) -> bytes:
    return _CALLBACK_PAGE.format(title=title, detail=detail).encode("utf-8")


# --------------------------------------------------------------------------
# connect attempts (Flow A loopback, with a manual-code fallback)
# --------------------------------------------------------------------------


@dataclass
class ConnectAttempt:
    attempt_id: int
    verifier: str
    state: str
    authorize_url: str
    loopback_port: int | None
    callback_url: str
    created_at: float
    expires_at: float
    status: str = "pending"  # pending|authorized|denied|error|cancelled|expired
    code: str = "pending"
    message: str = ""
    credential: Credential | None = None
    scope: str = DEFAULT_SCOPE
    server: Any = None

    def public(self) -> dict[str, Any]:
        """Browser-safe projection. The verifier and state never leave the server."""
        return {
            "attempt": self.attempt_id,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "authorize_url": self.authorize_url,
            "expires_in": max(0, int(self.expires_at - time.time())),
            "callback_url": self.callback_url,
            "scope": self.credential.scope if self.credential else self.scope,
            "credential": self.credential.describe() if self.credential else None,
        }


class _CallbackHandler(BaseHTTPRequestHandler):
    server_version = "ModelTraceOrca/1.0"

    def log_message(self, *args: Any) -> None:  # keep secrets out of stderr
        return

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        manager: ConnectManager | None = getattr(self.server, "orca_manager", None)
        attempt: ConnectAttempt | None = getattr(self.server, "orca_attempt", None)
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/cb" or manager is None or attempt is None:
            self._respond(404, callback_page("路径无效", "该地址仅用于接收 OrcaRouter 授权回调。"))
            return
        query = urllib.parse.parse_qs(parsed.query)
        # Answer immediately: the user must never stare at a blank tab while the
        # exchange runs in the background.
        self._respond(200, callback_page("已收到授权", "可以关闭此页面并回到 ModelTrace。"))
        manager.handle_callback(
            attempt,
            state=(query.get("state") or [None])[0],
            error=(query.get("error") or [None])[0],
            code=(query.get("code") or [None])[0],
        )

    def _respond(self, status: int, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass


class ConnectManager:
    """Server-side authorization attempts for the local ModelTrace process.

    Flow A (loopback redirect) is used because ModelTrace runs as a local
    process with a browser on the same machine and free loopback ports. The
    consent screen can still hand the user a displayed code ("Show me a code"),
    which is why every attempt also accepts a manually pasted code, and why
    S256 is sent unconditionally.
    """

    def __init__(
        self,
        *,
        store: CredentialStore | None = None,
        environ: dict[str, str] | None = None,
        ttl_seconds: float = CONSENT_TTL_SECONDS,
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.environ = environ if environ is not None else os.environ
        self.store = store or CredentialStore()
        self.ttl_seconds = ttl_seconds
        self._opener = opener
        self._clock = clock
        self._lock = threading.Lock()
        self._attempts: dict[int, ConnectAttempt] = {}
        self._current_id = 0
        self._counter = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, app_name: str = APP_NAME, scope: str = DEFAULT_SCOPE) -> dict[str, Any]:
        if scope not in ALLOWED_SCOPES:
            raise OrcaRouterError(f"不支持的 scope：{scope}")
        with self._lock:
            self._invalidate_locked()
            self._counter += 1
            attempt = self._new_attempt(app_name=app_name, scope=scope)
            self._attempts[attempt.attempt_id] = attempt
            self._current_id = attempt.attempt_id
        return attempt.public()

    def _new_attempt(self, *, app_name: str, scope: str) -> ConnectAttempt:
        server, port = self._bind_listener()
        verifier = generate_verifier()
        state = generate_state()
        callback_url = f"http://{LOOPBACK_HOST}:{port}/cb" if port else "oob"
        url = authorize_url(
            auth_base(self.environ),
            callback_url=callback_url,
            challenge=code_challenge(verifier),
            state=state,
            app_name=app_name,
            scope=scope,
        )
        now = self._clock()
        attempt = ConnectAttempt(
            attempt_id=self._counter,
            verifier=verifier,
            state=state,
            authorize_url=url,
            loopback_port=port,
            callback_url=callback_url,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            scope=scope,
            server=server,
        )
        if server is not None:
            # Attach before serving so the first request already sees the attempt.
            server.orca_attempt = attempt
            server.orca_manager = self
            threading.Thread(
                target=server.serve_forever,
                name=f"orca-callback-{attempt.attempt_id}",
                daemon=True,
            ).start()
        return attempt

    def _bind_listener(self) -> tuple[Any, int | None]:
        try:
            server = ThreadingHTTPServer((LOOPBACK_HOST, 0), _CallbackHandler)
        except OSError:
            # No loopback listener available: the attempt still works, the user
            # pastes the code the consent screen shows.
            return None, None
        server.daemon_threads = True
        return server, int(server.server_address[1])

    def _invalidate_locked(self) -> None:
        """Drop the previous pending attempt so a late response cannot land."""
        current = self._attempts.get(self._current_id)
        self._current_id = 0
        if current is not None and current.status == "pending":
            self._close(current)
            current.status = "cancelled"
            current.code = "superseded"
            current.message = "已被新的授权请求取代"

    def _close(self, attempt: ConnectAttempt) -> None:
        server = attempt.server
        attempt.server = None
        if server is None:
            return

        def stop() -> None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001 - shutdown must never raise into a request
                pass
            try:
                server.server_close()
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=stop, name="orca-callback-stop", daemon=True).start()

    def _expire_if_needed(self, attempt: ConnectAttempt | None) -> ConnectAttempt | None:
        if attempt is None:
            return None
        if attempt.status == "pending" and self._clock() >= attempt.expires_at:
            self._close(attempt)
            attempt.status = "expired"
            attempt.code = "expired"
            attempt.message = "授权等待超时（10 分钟），请重新发起连接"
        return attempt

    def get(self, attempt_id: int) -> ConnectAttempt | None:
        with self._lock:
            return self._expire_if_needed(self._attempts.get(int(attempt_id)))

    def status(self, attempt_id: int) -> dict[str, Any]:
        attempt = self.get(attempt_id)
        if attempt is None:
            raise ConnectError("未知的授权请求，请重新发起连接", code="unknown_attempt")
        self._cleanup()
        return attempt.public()

    def _cleanup(self) -> None:
        with self._lock:
            stale = [
                key
                for key, item in self._attempts.items()
                if item.status != "pending" and key != self._current_id
            ]
            for key in stale[:-8] if len(stale) > 8 else []:
                self._attempts.pop(key, None)

    # -- terminal paths ----------------------------------------------------

    def cancel(self, attempt_id: int) -> dict[str, Any]:
        with self._lock:
            attempt = self._attempts.get(int(attempt_id))
            if attempt is None:
                raise ConnectError("未知的授权请求，请重新发起连接", code="unknown_attempt")
            if attempt.status == "pending":
                self._close(attempt)
                attempt.status = "cancelled"
                attempt.code = "cancelled"
                attempt.message = "已取消授权"
            if self._current_id == attempt.attempt_id:
                self._current_id = 0
            return attempt.public()

    def handle_callback(
        self,
        attempt: ConnectAttempt,
        *,
        state: str | None,
        error: str | None,
        code: str | None,
    ) -> None:
        """Entry point for the loopback redirect. Runs off the request thread."""
        if not state_matches(attempt.state, state):
            self._settle(
                attempt, "error", "state_mismatch", "授权回调的 state 校验失败，已忽略本次回调"
            )
            return
        if error:
            self._settle(attempt, "denied", str(error), f"授权被拒绝（{error}）")
            return
        if not code:
            self._settle(attempt, "error", "missing_code", "授权回调未包含授权码")
            return
        self.submit_code(attempt.attempt_id, code)

    def submit_code(self, attempt_id: int, code: str) -> dict[str, Any]:
        attempt = self.get(attempt_id)
        if attempt is None:
            raise ConnectError("未知的授权请求，请重新发起连接", code="unknown_attempt")
        if attempt.status != "pending":
            return attempt.public()
        if not code or not code.strip():
            raise ConnectError("授权码为空", code="missing_code")
        try:
            credential = exchange_code(
                auth_base(self.environ), code.strip(), attempt.verifier, opener=self._opener
            )
        except ConnectError as failure:
            self._settle(attempt, "error", failure.code, str(failure))
            return attempt.public()
        try:
            stored = self.store.save(
                credential.key,
                source="pkce",
                account_id=credential.account_id,
                scope=credential.scope,
            )
        except OrcaRouterError as failure:
            self._settle(attempt, "error", "storage_failed", str(failure))
            return attempt.public()
        self._settle(
            attempt, "authorized", "authorized", "已连接 OrcaRouter 账户", credential=stored
        )
        return attempt.public()

    def _settle(
        self,
        attempt: ConnectAttempt,
        status: str,
        code: str,
        message: str,
        *,
        credential: Credential | None = None,
    ) -> None:
        """Apply a terminal result only if this attempt is still the current one.

        This is the server half of the generation guard: a callback or exchange
        belonging to an attempt the user has already replaced (or cancelled)
        cannot write a credential over the newer one.
        """
        with self._lock:
            if attempt.status != "pending" or attempt.attempt_id != self._current_id:
                return
            self._close(attempt)
            attempt.status = status
            attempt.code = code
            attempt.message = redact(message)
            attempt.credential = credential
            self._current_id = 0


# --------------------------------------------------------------------------
# model catalog
# --------------------------------------------------------------------------


@dataclass
class Catalog:
    """A bounded, filtered view of the OrcaRouter model list."""

    models: list[dict[str, Any]] = field(default_factory=list)
    source: str = "live"  # live | cache | seed
    degraded: bool = False
    error: str | None = None
    fetched_at: str | None = None
    total_seen: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "source": self.source,
            "degraded": self.degraded,
            "error": self.error,
            "fetched_at": self.fetched_at,
            "count": len(self.models),
            "total_seen": self.total_seen,
        }


def _endpoint_types(model: dict[str, Any]) -> list[str]:
    raw = model.get("supported_endpoint_types")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _input_modalities(model: dict[str, Any]) -> list[str]:
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return []
    raw = architecture.get("input_modalities")
    if not isinstance(raw, list):
        return []
    return [item.lower() for item in raw if isinstance(item, str)]


def describe_model(model: dict[str, Any]) -> dict[str, Any]:
    """Minimal browser-safe model metadata: no credentials, no pricing internals."""
    architecture = model.get("architecture")
    return {
        "id": model.get("id"),
        "name": model.get("name") or model.get("id"),
        "owned_by": model.get("owned_by"),
        "context_length": model.get("context_length")
        if isinstance(model.get("context_length"), int)
        else None,
        "max_completion_tokens": model.get("max_completion_tokens")
        if isinstance(model.get("max_completion_tokens"), int)
        else None,
        "input_modalities": _input_modalities(model),
        "supported_endpoint_types": _endpoint_types(model),
        "reasoning_efforts": model.get("reasoning_efforts") or [],
        "description": (model.get("description") or "")[:400]
        if isinstance(model.get("description"), str)
        else "",
    }


def capability_filter(
    models: Iterable[dict[str, Any]],
    capability: str = "chat",
    *,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    """Central capability filter. Every AI entry point goes through this.

    ``chat`` accepts a model only when it speaks a text endpoint, which also
    excludes models dedicated to ``image-generation`` / ``openai-video`` /
    ``jina-rerank``. A modality filter additionally requires the catalog to
    *declare* that input modality: unknown modalities fail closed, so a model
    whose capabilities cannot be proven never reaches a multimodal dropdown.
    """
    endpoints = CAPABILITY_ENDPOINT_TYPES.get(capability)
    if endpoints is None:
        raise OrcaRouterError(f"未知的模型能力：{capability}")
    selected: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        types = _endpoint_types(model)
        if not types or not set(types) & set(endpoints):
            continue
        if capability == "chat" and set(types) <= set(NON_TEXT_ENDPOINT_TYPES):
            continue
        if modality and modality.lower() not in _input_modalities(model):
            continue
        selected.append(describe_model(model))
    return selected


def parse_catalog(payload: Any) -> list[dict[str, Any]]:
    """Accept only the documented OpenAI ``{"data": [...]}`` shape, bounded."""
    if isinstance(payload, dict):
        items = payload.get("data")
    elif isinstance(payload, list):
        items = payload
    else:
        items = None
    if not isinstance(items, list):
        raise OrcaRouterError("模型目录响应格式无法识别")
    return [item for item in items[:CATALOG_MAX_ITEMS] if isinstance(item, dict)]


def fetch_models(
    base: str,
    key: str,
    *,
    capability: str = "chat",
    modality: str | None = None,
    timeout: float = CATALOG_TIMEOUT_SECONDS,
    opener: Callable[..., Any] | None = None,
) -> Catalog:
    """Live discovery against ``{api_base}/models``. Never raises to the caller."""
    endpoint = capability if capability in CAPABILITY_ENDPOINT_TYPES else "chat"
    query = urllib.parse.urlencode({"capability": endpoint})
    target = f"{validate_origin(base, label='OrcaRouter 接口地址')}/models?{query}"
    request = urllib.request.Request(
        target,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": APP_NAME,
        },
        method="GET",
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:
            raw = _read_bounded(response)
    except urllib.error.HTTPError as error:
        detail = _safe_error_body(error.read(4096))
        message = f"模型目录请求失败（HTTP {error.code}）"
        if error.code == 401:
            message = "模型目录请求被拒绝：凭据无效或已被撤销"
        return Catalog(
            source="seed", degraded=True, error=message + (f" {detail}" if detail else "")
        )
    except urllib.error.URLError as error:
        reason = redact(str(getattr(error, "reason", error)))
        return Catalog(source="seed", degraded=True, error=f"无法连接模型目录：{reason}")
    except (TimeoutError, OSError) as error:
        return Catalog(
            source="seed", degraded=True, error=f"模型目录请求失败：{type(error).__name__}"
        )
    try:
        items = parse_catalog(json.loads(raw.decode("utf-8")))
    except (json.JSONDecodeError, ValueError, OrcaRouterError) as error:
        return Catalog(source="seed", degraded=True, error=f"模型目录响应无法解析：{error}")
    return Catalog(
        models=capability_filter(items, capability, modality=modality),
        source="live",
        degraded=False,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        total_seen=len(items),
    )


def seed_catalog(capability: str = "chat", modality: str | None = None) -> Catalog:
    """Verified cold-start fallback, used only when live discovery fails."""
    return Catalog(
        models=capability_filter(SEED_MODELS, capability, modality=modality),
        source="seed",
        degraded=True,
        error=None,
        total_seen=len(SEED_MODELS),
    )


class CatalogCache:
    """Tiny in-process cache plus an on-disk last-known-good catalog.

    Live discovery is authoritative whenever it succeeds; the persisted copy is
    only consulted after a failure, and it is re-filtered before use, so a stale
    entry that no longer matches the requested capability cannot come back.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        ttl_seconds: float = CATALOG_CACHE_SECONDS,
        failure_ttl_seconds: float = CATALOG_FAILURE_CACHE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path or default_catalog_cache_path()
        self.ttl_seconds = ttl_seconds
        self.failure_ttl_seconds = failure_ttl_seconds
        self._clock = clock
        self._memory: dict[str, tuple[float, Catalog]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def cache_key(base: str, key: str, capability: str, modality: str | None) -> str:
        digest = hashlib.sha256(
            f"{base}|{key}|{capability}|{modality}".encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()

    def _load_disk(self) -> Catalog | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
            return None
        return Catalog(
            models=[item for item in raw["models"] if isinstance(item, dict)],
            source="cache",
            degraded=True,
            fetched_at=raw.get("fetched_at"),
            total_seen=int(raw.get("total_seen") or len(raw["models"])),
        )

    def _save_disk(self, catalog: Catalog) -> None:
        try:
            _write_private_json(
                self.path,
                {
                    "version": 1,
                    "fetched_at": catalog.fetched_at,
                    "total_seen": catalog.total_seen,
                    "models": catalog.models,
                },
            )
        except OSError:
            pass

    def get(
        self,
        base: str,
        key: str,
        *,
        capability: str = "chat",
        modality: str | None = None,
        force: bool = False,
        fetcher: Callable[..., Catalog] | None = None,
    ) -> Catalog:
        cache_key = self.cache_key(base, key, capability, modality)
        now = self._clock()
        if not force:
            with self._lock:
                cached = self._memory.get(cache_key)
            if cached is not None:
                ttl = self.ttl_seconds if not cached[1].degraded else self.failure_ttl_seconds
                if now - cached[0] < ttl:
                    return cached[1]

        live = (fetcher or fetch_models)(base, key, capability=capability, modality=modality)
        if live.source == "live":
            with self._lock:
                self._memory[cache_key] = (now, live)
            self._save_disk(live)
            return live

        stored = self._load_disk()
        if stored is not None:
            filtered = capability_filter(stored.models, capability, modality=modality)
            if filtered:
                result = Catalog(
                    models=filtered,
                    source="cache",
                    degraded=True,
                    error=live.error,
                    fetched_at=stored.fetched_at,
                    total_seen=stored.total_seen,
                )
                with self._lock:
                    self._memory[cache_key] = (now, result)
                return result

        fallback = seed_catalog(capability, modality)
        fallback.error = live.error
        with self._lock:
            self._memory[cache_key] = (now, fallback)
        return fallback


def provider_registry(environ: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Provider metadata for the UI. OrcaRouter appears as two named entries."""
    env = environ if environ is not None else os.environ
    return [
        {
            "id": CUSTOM_PROVIDER_ID,
            "label": PROVIDER_LABELS[CUSTOM_PROVIDER_ID],
            "kind": "custom",
            "auth_methods": ["base_url"],
            "requires_model_input": True,
        },
        {
            "id": PROVIDER_ID,
            "label": PROVIDER_LABELS[PROVIDER_ID],
            "kind": "orcarouter",
            "auth_methods": ["api_key"],
            "requires_model_input": False,
            "auth_base": auth_base(env),
            "api_base": api_base(env),
            "dashboard_url": DASHBOARD_URL,
            "logo_url": LOGO_URL,
        },
        {
            "id": PKCE_PROVIDER_ID,
            "label": PROVIDER_LABELS[PKCE_PROVIDER_ID],
            "kind": "orcarouter",
            "auth_methods": ["pkce"],
            "requires_model_input": False,
            "auth_base": auth_base(env),
            "api_base": api_base(env),
            "dashboard_url": DASHBOARD_URL,
            "logo_url": LOGO_URL,
        },
    ]


def auth_method_for_provider(provider_id: str) -> str:
    if provider_id == PROVIDER_ID:
        return "api_key"
    if provider_id == PKCE_PROVIDER_ID:
        return "pkce"
    raise OrcaRouterError(f"未知的服务提供方：{provider_id}")
