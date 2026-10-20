"""Webhook consumers of the daily job: registration files under ``./webhooks`` and the POST.

A consumer registers by dropping ``<name>.json`` into the webhooks directory (git-ignored;
``*.example.json`` files are documentation and are skipped). The daily job posts the run
*summary* (``hedgefund.jobs.daily.summary_payload``) to every registration whose
``events`` list includes the event. The payload is a signal, never an order, and carries
no paths, no credentials and no environment values.

An optional ``token_env`` names an environment variable whose **value** is sent as the
``X-Hedgefund-Token`` header. The name must start with ``HEDGEFUND_WEBHOOK_`` so a
registration can never point at another credential (the Alpaca keys in particular).
Only the variable's name is ever stored; the value is read at post time and never
appears in a result, an error message or a file. Redirects are refused, so the header
only ever reaches the registered URL, and the URL itself (which may carry a secret
path, as Slack and Discord hooks do) is never printed: results show the host only.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

REGISTRATION_SUFFIX = ".json"
EXAMPLE_SUFFIX = ".example.json"  # documentation only, never posted to
DEFAULT_EVENTS = ("daily",)  # a registration without ``events`` receives the daily summary
TOKEN_HEADER = "X-Hedgefund-Token"
CONTENT_TYPE = "application/json"
USER_AGENT = "hedgefund-webhook"
DEFAULT_TIMEOUT_SECONDS = 10.0  # a slow consumer must not stall the job for long
ERROR_MAX_CHARS = 200  # keep ``WebhookResult.error`` short and free of response bodies
# HTTPS everywhere except loopback, where plain HTTP is allowed for local testing.
SECURE_SCHEME = "https"
LOOPBACK_SCHEME = "http"
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")  # exact hostname matches only
# A token variable must carry this prefix so a registration cannot name another secret.
TOKEN_ENV_PREFIX = "HEDGEFUND_WEBHOOK_"
FORBIDDEN_TOKEN_ENV_PREFIXES = ("APCA_",)  # the Alpaca credentials, never forwarded anywhere
REQUIRED_KEYS = ("name", "url")


@dataclass(frozen=True)
class Webhook:
    """One registered consumer. ``token_env`` is a variable *name*, never a value."""

    name: str
    url: str
    events: tuple[str, ...]
    token_env: str | None


@dataclass
class WebhookResult:
    """The outcome of one POST; ``error`` never contains the token."""

    name: str
    url: str  # kept for callers; never serialised, a hook URL may carry a secret path
    ok: bool
    status: int | None
    error: str | None

    @property
    def host(self) -> str | None:
        """The URL's hostname, the only part of it that is ever printed."""
        return url_host(self.url)

    def to_dict(self) -> dict:
        """JSON-ready, without the URL: name, host, ok, status, error."""
        return {
            "name": self.name,
            "host": self.host,
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
        }


def url_host(url: str) -> str | None:
    """``urlsplit(url).hostname`` or ``None`` when the URL has none."""
    try:
        return urllib.parse.urlsplit(url).hostname
    except ValueError:
        return None


def _validate_url(url: object, path: Path) -> str:
    """``https://<any host>`` or ``http://`` to an exact loopback hostname, else ``ValueError``."""
    expected = (
        f"'url' must be {SECURE_SCHEME}://<host>/... or {LOOPBACK_SCHEME}:// to exactly one of "
        f"{LOOPBACK_HOSTS} for local testing"
    )
    if not isinstance(url, str):
        raise ValueError(f"webhook file {path}: {expected}, got {url!r}")
    try:
        parts = urllib.parse.urlsplit(url)
        hostname = parts.hostname
    except ValueError as exc:
        raise ValueError(f"webhook file {path}: {expected}, got {url!r} ({exc})") from exc
    if not hostname:
        raise ValueError(f"webhook file {path}: {expected}, got {url!r} (no host)")
    if parts.scheme == SECURE_SCHEME:
        return url
    if parts.scheme == LOOPBACK_SCHEME and hostname in LOOPBACK_HOSTS:
        return url
    raise ValueError(f"webhook file {path}: {expected}, got {url!r}")


def _validate_token_env(token_env: object, path: Path) -> str | None:
    """A ``HEDGEFUND_WEBHOOK_*`` variable name, never an Alpaca one, else ``ValueError``."""
    if token_env is None:
        return None
    if not isinstance(token_env, str) or not token_env:
        raise ValueError(
            f"webhook file {path}: 'token_env' must be an environment variable name, "
            f"got {token_env!r}"
        )
    if token_env.startswith(FORBIDDEN_TOKEN_ENV_PREFIXES):
        raise ValueError(
            f"webhook file {path}: 'token_env' {token_env!r} names a broker credential; "
            f"only {TOKEN_ENV_PREFIX}* variables may be sent to a webhook"
        )
    if not token_env.startswith(TOKEN_ENV_PREFIX):
        raise ValueError(
            f"webhook file {path}: 'token_env' must start with {TOKEN_ENV_PREFIX}, "
            f"got {token_env!r}"
        )
    return token_env


def _validate_events(events: object, path: Path) -> tuple[str, ...]:
    if events is None:
        return DEFAULT_EVENTS
    if (
        not isinstance(events, list)
        or not events
        or not all(isinstance(e, str) and e for e in events)
    ):
        raise ValueError(
            f"webhook file {path}: 'events' must be a non-empty list of strings, got {events!r}"
        )
    return tuple(events)


def parse_webhook(data: object, path: Path) -> Webhook:
    """Build a ``Webhook`` from one registration mapping; ``ValueError`` names ``path``."""
    if not isinstance(data, dict):
        raise ValueError(f"webhook file {path} must hold a JSON object, got {type(data).__name__}")
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"webhook file {path} is missing required key {key!r}")
    name = data["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"webhook file {path}: 'name' must be a non-empty string, got {name!r}")
    return Webhook(
        name=name,
        url=_validate_url(data["url"], path),
        events=_validate_events(data.get("events"), path),
        token_env=_validate_token_env(data.get("token_env"), path),
    )


def load_webhooks(directory: Path) -> list[Webhook]:
    """Every ``*.json`` registration under ``directory`` (sorted by file name).

    ``*.example.json`` files are skipped. A missing directory yields ``[]``. A malformed
    file raises ``ValueError`` naming it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    webhooks = []
    for path in sorted(directory.glob(f"*{REGISTRATION_SUFFIX}")):
        if path.name.endswith(EXAMPLE_SUFFIX):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"webhook file {path} is not valid JSON: {exc}") from exc
        webhooks.append(parse_webhook(data, path))
    return webhooks


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: a 3xx is an ``HTTPError`` with that status, never followed.

    urllib's default handler would replay the request (and the token header) against the
    redirect target and turn the POST into a GET; the only host that may see the token is
    the one in the registration file.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def build_opener() -> urllib.request.OpenerDirector:
    """An opener that never follows a redirect. The stdlib env proxy handler still applies."""
    return urllib.request.build_opener(NoRedirectHandler())


def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: urllib.request.OpenerDirector | None = None,
) -> int:
    """POST ``payload`` as JSON with ``headers`` via the standard library; return the status.

    Redirects are not followed: a 3xx raises ``urllib.error.HTTPError`` like any other
    non-2xx response (its message carries the status code only, never the request
    headers). ``opener`` defaults to ``build_opener()``; tests pass a fake.
    """
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": CONTENT_TYPE, "User-Agent": USER_AGENT, **headers},
        method="POST",
    )
    with (opener if opener is not None else build_opener()).open(request, timeout=timeout) as resp:
        return int(resp.status)


def _headers_for(webhook: Webhook, environ: Mapping[str, str]) -> dict[str, str]:
    if webhook.token_env is None:
        return {}
    token = environ.get(webhook.token_env)
    if token is None:
        raise LookupError(
            f"webhook {webhook.name!r} needs environment variable {webhook.token_env} "
            "for its token; it is not set"
        )
    return {TOKEN_HEADER: token}


def _scrub(text: str, secrets: list[str]) -> str:
    """Remove every secret value from ``text`` so no error message can leak a token."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def post_summary(
    webhooks: list[Webhook],
    payload: dict,
    *,
    event: str = DEFAULT_EVENTS[0],
    environ: Mapping[str, str] | None = None,
    post: Callable[..., int] = post_json,
) -> list[WebhookResult]:
    """POST ``payload`` to every webhook subscribed to ``event``; one result each.

    Never raises: an exception from ``post`` (or a missing token variable) becomes a
    result with ``ok=False`` and a short ``error`` that has the token value scrubbed out.
    ``environ`` defaults to ``os.environ`` and is read only for ``token_env`` lookups.
    """
    if environ is None:
        import os

        environ = os.environ
    results = []
    for webhook in webhooks:
        if event not in webhook.events:
            continue
        secrets: list[str] = [webhook.url]  # a hook URL may carry a secret path
        try:
            headers = _headers_for(webhook, environ)
            secrets.extend(headers.values())
            status = post(webhook.url, payload, headers=headers)
            results.append(
                WebhookResult(
                    name=webhook.name, url=webhook.url, ok=True, status=int(status), error=None
                )
            )
        except Exception as exc:
            status = getattr(exc, "code", None) if isinstance(exc, urllib.error.HTTPError) else None
            message = _scrub(f"{type(exc).__name__}: {exc}", secrets)[:ERROR_MAX_CHARS]
            results.append(
                WebhookResult(
                    name=webhook.name, url=webhook.url, ok=False, status=status, error=message
                )
            )
    return results
