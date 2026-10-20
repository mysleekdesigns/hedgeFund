"""``hedgefund.jobs.webhooks``: registration parsing, event filtering, token handling."""

import email.message
import io
import json
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path

import pytest

from hedgefund.jobs.webhooks import (
    DEFAULT_EVENTS,
    ERROR_MAX_CHARS,
    TOKEN_ENV_PREFIX,
    TOKEN_HEADER,
    NoRedirectHandler,
    Webhook,
    WebhookResult,
    build_opener,
    load_webhooks,
    post_json,
    post_summary,
)

TOKEN_VAR = "HEDGEFUND_WEBHOOK_TEST"
SENTINEL_TOKEN = "sentinel-token-7f3a9c"  # a value no real file or payload would contain
SECRET_PATH = "T0SECRET/B0PATH/xyzSENTINEL"  # the secret part of a Slack / Discord hook URL
HTTPS_URL = "https://hooks.example.invalid/daily"
LOCAL_URL = "http://localhost:8000/daily"
PAYLOAD = {"event": "daily", "orders_placed": False}


def _write(directory: Path, name: str, data: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def test_load_webhooks_parses_every_registration_sorted_by_file_name(tmp_path: Path) -> None:
    _write(tmp_path, "b-second", {"name": "second", "url": LOCAL_URL, "events": ["weekly"]})
    _write(tmp_path, "a-first", {"name": "first", "url": HTTPS_URL, "token_env": TOKEN_VAR})
    hooks = load_webhooks(tmp_path)
    assert hooks == [
        Webhook(name="first", url=HTTPS_URL, events=DEFAULT_EVENTS, token_env=TOKEN_VAR),
        Webhook(name="second", url=LOCAL_URL, events=("weekly",), token_env=None),
    ]


def test_load_webhooks_skips_example_files(tmp_path: Path) -> None:
    (tmp_path / "demo.example.json").write_text(json.dumps({"name": "demo", "url": HTTPS_URL}))
    _write(tmp_path, "real", {"name": "real", "url": HTTPS_URL})
    assert [h.name for h in load_webhooks(tmp_path)] == ["real"]


def test_load_webhooks_missing_directory_is_empty(tmp_path: Path) -> None:
    assert load_webhooks(tmp_path / "nope") == []


def test_load_webhooks_rejects_plain_http_url_naming_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad", {"name": "bad", "url": "http://hooks.example.invalid/x"})
    with pytest.raises(ValueError, match=str(path)):
        load_webhooks(tmp_path)


def test_load_webhooks_rejects_missing_required_key_naming_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "nourl", {"name": "nourl"})
    with pytest.raises(ValueError, match=f"{path}.*'url'"):
        load_webhooks(tmp_path)


def test_load_webhooks_rejects_invalid_json_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match=str(path)):
        load_webhooks(tmp_path)


def test_repo_example_registration_is_skipped_and_valid() -> None:
    """The checked-in example parses as a registration but is never loaded for posting."""
    from hedgefund.jobs.webhooks import parse_webhook

    directory = Path(__file__).resolve().parents[1] / "webhooks"
    example = directory / "example.example.json"
    assert example.is_file()
    hook = parse_webhook(json.loads(example.read_text()), example)
    assert hook.url.startswith("https://")
    assert all(h.name != "example" for h in load_webhooks(directory))


def _recorder(status: int = 200) -> tuple[list, object]:
    calls: list[tuple[str, dict, dict]] = []

    def post(url: str, payload: dict, *, headers: dict[str, str]) -> int:
        calls.append((url, payload, headers))
        return status

    return calls, post


def test_post_summary_only_posts_to_webhooks_subscribed_to_the_event() -> None:
    hooks = [
        Webhook("daily-only", HTTPS_URL, ("daily",), None),
        Webhook("weekly-only", LOCAL_URL, ("weekly",), None),
        Webhook("both", HTTPS_URL, ("daily", "weekly"), None),
    ]
    calls, post = _recorder()
    results = post_summary(hooks, PAYLOAD, event="daily", environ={}, post=post)
    assert [r.name for r in results] == ["daily-only", "both"]
    assert all(r.ok and r.status == 200 and r.error is None for r in results)
    assert [c[1] for c in calls] == [PAYLOAD, PAYLOAD]


def test_post_summary_sends_the_token_value_as_a_header_and_never_stores_it() -> None:
    hook = Webhook("tok", HTTPS_URL, DEFAULT_EVENTS, TOKEN_VAR)
    calls, post = _recorder()
    results = post_summary([hook], PAYLOAD, environ={TOKEN_VAR: SENTINEL_TOKEN}, post=post)
    assert calls[0][2] == {TOKEN_HEADER: SENTINEL_TOKEN}
    assert SENTINEL_TOKEN not in json.dumps(results[0].to_dict())
    assert SENTINEL_TOKEN not in json.dumps(calls[0][1])


def test_post_summary_missing_token_variable_is_a_failed_result_not_an_exception() -> None:
    hook = Webhook("tok", HTTPS_URL, DEFAULT_EVENTS, TOKEN_VAR)
    calls, post = _recorder()
    results = post_summary([hook], PAYLOAD, environ={}, post=post)
    assert calls == []
    assert results[0].ok is False and results[0].status is None
    assert TOKEN_VAR in results[0].error


def test_post_summary_exception_becomes_a_scrubbed_short_error() -> None:
    hook = Webhook("boom", HTTPS_URL, DEFAULT_EVENTS, TOKEN_VAR)

    def post(url: str, payload: dict, *, headers: dict[str, str]) -> int:
        # A sloppy transport that echoes the headers into its error message.
        raise OSError(f"connection refused with headers {headers} " + "x" * 500)

    results = post_summary([hook], PAYLOAD, environ={TOKEN_VAR: SENTINEL_TOKEN}, post=post)
    result = results[0]
    assert isinstance(result, WebhookResult)
    assert result.ok is False and result.status is None
    assert result.error.startswith("OSError: ")
    assert SENTINEL_TOKEN not in result.error
    assert len(result.error) <= ERROR_MAX_CHARS


def test_webhook_result_to_dict_shows_the_host_never_the_url() -> None:
    secret_url = f"https://hooks.example.invalid/services/{SECRET_PATH}"
    result = WebhookResult(name="n", url=secret_url, ok=True, status=204, error=None)
    as_dict = json.loads(json.dumps(result.to_dict()))
    assert as_dict == {
        "name": "n",
        "host": "hooks.example.invalid",
        "ok": True,
        "status": 204,
        "error": None,
    }
    assert SECRET_PATH not in json.dumps(as_dict)
    assert result.url == secret_url  # kept on the object for internal use


# --- token_env must be a HEDGEFUND_WEBHOOK_* variable, never a broker credential ---------


@pytest.mark.parametrize("name", ["APCA_API_SECRET_KEY", "APCA_API_KEY_ID", "HOME", "TOKEN"])
def test_load_webhooks_rejects_token_env_outside_the_prefix_naming_the_file(
    tmp_path: Path, name: str
) -> None:
    path = _write(tmp_path, "exfil", {"name": "exfil", "url": HTTPS_URL, "token_env": name})
    with pytest.raises(ValueError, match=f"{path}.*token_env"):
        load_webhooks(tmp_path)


def test_load_webhooks_accepts_a_prefixed_token_env(tmp_path: Path) -> None:
    _write(tmp_path, "ok", {"name": "ok", "url": HTTPS_URL, "token_env": "HEDGEFUND_WEBHOOK_FOO"})
    (hook,) = load_webhooks(tmp_path)
    assert hook.token_env == "HEDGEFUND_WEBHOOK_FOO"
    assert hook.token_env.startswith(TOKEN_ENV_PREFIX)


def test_alpaca_secret_can_never_be_sent_as_a_webhook_token(tmp_path: Path) -> None:
    """Even a hand-built Webhook naming an APCA_ variable is refused at parse time."""
    from hedgefund.jobs.webhooks import parse_webhook

    data = {"name": "x", "url": HTTPS_URL, "token_env": "APCA_API_SECRET_KEY"}
    with pytest.raises(ValueError, match="broker credential"):
        parse_webhook(data, tmp_path / "x.json")


# --- URL validation parses the host instead of matching a prefix ----------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost.attacker.invalid/daily",
        "http://127.0.0.10/daily",
        "http://127.0.0.1.attacker.invalid/daily",
        "http://hooks.example.invalid/daily",
        "ftp://localhost/daily",
        "https:///no-host",
    ],
)
def test_load_webhooks_rejects_non_loopback_http_and_hostless_urls(
    tmp_path: Path, url: str
) -> None:
    path = _write(tmp_path, "bad", {"name": "bad", "url": url})
    with pytest.raises(ValueError, match=str(path)):
        load_webhooks(tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/daily",
        "http://localhost:8000/daily",
        "http://127.0.0.1:9/daily",
        "http://[::1]:8000/daily",
        "https://hooks.example.invalid/services/abc",
    ],
)
def test_load_webhooks_accepts_https_and_exact_loopback_http(tmp_path: Path, url: str) -> None:
    _write(tmp_path, "ok", {"name": "ok", "url": url})
    assert load_webhooks(tmp_path)[0].url == url


# --- redirects are refused -------------------------------------------------------------


class _RedirectingHandler(urllib.request.BaseHandler):
    """A fake transport: every request gets a 302 to another host; counts the requests."""

    handler_order = 100  # before the real HTTPSHandler (500), so no socket is touched

    def __init__(self) -> None:
        self.requests: list[urllib.request.Request] = []

    def https_open(self, req: urllib.request.Request) -> urllib.response.addinfourl:
        self.requests.append(req)
        headers = email.message.Message()
        headers["Location"] = "https://attacker.invalid/steal"
        response = urllib.response.addinfourl(io.BytesIO(b""), headers, req.full_url, code=302)
        response.msg = "Found"  # what a real HTTPResponse carries; the error processor reads it
        return response


def test_post_json_does_not_follow_a_redirect() -> None:
    transport = _RedirectingHandler()
    opener = build_opener()
    opener.add_handler(transport)
    with pytest.raises(urllib.error.HTTPError) as info:
        post_json(HTTPS_URL, PAYLOAD, headers={TOKEN_HEADER: SENTINEL_TOKEN}, opener=opener)
    assert info.value.code == 302
    # Exactly one request, to the registered URL; the redirect target was never contacted.
    assert [r.full_url for r in transport.requests] == [HTTPS_URL]
    assert SENTINEL_TOKEN not in str(info.value)


def test_no_redirect_handler_returns_none_for_every_code() -> None:
    handler = NoRedirectHandler()
    request = urllib.request.Request(HTTPS_URL, data=b"{}", method="POST")
    for code in (301, 302, 303, 307, 308):
        assert handler.redirect_request(request, None, code, "", {}, LOCAL_URL) is None


def test_post_summary_reports_a_redirect_as_a_failed_result_with_its_status() -> None:
    hook = Webhook("redir", HTTPS_URL, DEFAULT_EVENTS, TOKEN_VAR)

    def post(url: str, payload: dict, *, headers: dict[str, str]) -> int:
        raise urllib.error.HTTPError(url, 302, "Found", {}, None)

    (result,) = post_summary([hook], PAYLOAD, environ={TOKEN_VAR: SENTINEL_TOKEN}, post=post)
    assert result.ok is False and result.status == 302


def test_post_summary_scrubs_the_url_from_error_text() -> None:
    secret_url = f"https://hooks.example.invalid/services/{SECRET_PATH}"
    hook = Webhook("secret", secret_url, DEFAULT_EVENTS, None)

    def post(url: str, payload: dict, *, headers: dict[str, str]) -> int:
        raise OSError(f"could not reach {url}")

    (result,) = post_summary([hook], PAYLOAD, environ={}, post=post)
    assert result.ok is False
    assert SECRET_PATH not in result.error
    assert SECRET_PATH not in json.dumps(result.to_dict())
