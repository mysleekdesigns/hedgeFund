# Webhook consumers

`hedgefund daily` posts a run **summary** to every consumer registered in this
directory. The summary is a signal, never an order: it carries `orders_placed: false`
and a note saying so, and nothing in hedgeFund submits, replaces or cancels orders.

## Registering a consumer

Drop one JSON file per consumer into this directory, for example `my-consumer.json`:

```json
{
  "name": "my-consumer",
  "url": "https://hooks.example.com/hedgefund",
  "events": ["daily"],
  "token_env": "HEDGEFUND_WEBHOOK_TOKEN"
}
```

| Key         | Required | Meaning                                                                                      |
|-------------|----------|----------------------------------------------------------------------------------------------|
| `name`      | yes      | Label used in the job output and in `--json` results.                                        |
| `url`       | yes      | `https://` to any host. Plain `http://` is accepted only when the hostname is exactly `localhost`, `127.0.0.1` or `::1` (local testing); `http://localhost.example.com` and `http://127.0.0.10` are rejected. |
| `events`    | no       | Events to receive; defaults to `["daily"]`. The daily job emits the `daily` event.           |
| `token_env` | no       | The **name** of an environment variable, which must start with `HEDGEFUND_WEBHOOK_`. Its value is sent as the `X-Hedgefund-Token` header. |

Real registration files are git-ignored (`.gitignore` keeps only this README and
`*.example.json`). Files ending in `.example.json` are documentation and are never posted to.

## Tokens

The token value is read from the environment at post time and never written to the
signal file, the run manifest, the job output or an error message. A missing variable is
reported as a failed post for that consumer; the job continues.

Only variables named `HEDGEFUND_WEBHOOK_*` can be used, and anything starting with
`APCA_` is refused outright, so a registration file can never forward the Alpaca keys
or any other credential. Pick one variable per consumer, for example
`HEDGEFUND_WEBHOOK_SLACK`.

The token is sent only to the registered URL: redirects are never followed. A `3xx`
response is reported as a failed post with that status.

## What the job prints

Webhook URLs often carry a secret in the path (Slack and Discord hooks do). The job
therefore never prints or serialises the URL: `hedgefund daily` shows the consumer's
`name`, its hostname, `ok`, `status` and a short `error`, and the same fields appear
under `webhooks` in `--json` output.

## Payload

`POST` with `Content-Type: application/json`:

```json
{
  "event": "daily",
  "as_of": "2026-10-19",
  "generated_at": "2026-10-20T09:00:00+00:00",
  "run_id": "20261020T090000Z_us_etfs",
  "universe": "us_etfs",
  "rule": "sign",
  "symbols": {
    "SPY": {"current_regime": "Bull", "signal": 0.42, "position": 1.0, "error": null}
  },
  "orders_placed": false,
  "note": "This file is a signal, not an order. Nothing in hedgeFund submits orders; placing an order is a separate, explicit human action.",
  "disclaimer": "Backtests are historical, not forward-looking."
}
```

No file paths, cache locations or environment values are included. The full per-symbol
detail lives in `jobs/<run-id>/signal.json` and `jobs/signal_latest.json`.

## Delivery

One POST per consumer, 10-second timeout, no retries, no redirects. A failure (network
error, non-2xx status including redirects, missing token variable) is printed as
`webhook <name> (<host>): FAILED ...` and listed in `--json` output under `webhooks`; the
signal file is written regardless. The signal files are written atomically (temp file
plus rename), so a consumer reading `jobs/signal_latest.json` never sees a partial file.
