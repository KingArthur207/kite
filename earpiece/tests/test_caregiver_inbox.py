import hashlib
import hmac
import json
import time

import httpx
import pytest

from earpiece.app import Earpiece
from earpiece.config import Cfg
from earpiece.spacetime import CAREGIVER_PREFIX, CAREGIVER_TTL_S, CaregiverInbox

SCHEMA = {"elements": [{"name": {"some": n}} for n in ("id", "type", "source", "priority", "payload", "t")]}


class FakeSpacetime:
    """Stands in for the SpacetimeDB HTTP API: /v1/identity and /v1/database/<db>/sql."""

    def __init__(self):
        self.rows: list[list] = []
        self.sql: list[str] = []

    def add(self, rid: int, payload, priority: str = "info") -> None:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        self.rows.append([rid, "say", "caregiver", priority, body, [0]])

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/identity":
            return httpx.Response(200, json={"identity": "x", "token": "tok"})
        assert req.headers["authorization"] == "Bearer tok"
        self.sql.append(req.content.decode())
        return httpx.Response(200, json=[{"schema": SCHEMA, "rows": list(self.rows)}])


def make(fake, token=""):
    got = []
    inbox = CaregiverInbox(got.append, host="http://st.test", db="kite", log=lambda *_: None, start=False,
                           token=token)
    return inbox, got, httpx.Client(transport=httpx.MockTransport(fake))


def say(text, **kw):
    return {"type": "say", "source": "caregiver", "priority": "info", "text": text, **kw}


def test_only_rows_added_after_startup_are_spoken_once():
    fake = FakeSpacetime()
    fake.add(1, say("old news"))
    inbox, got, http = make(fake)
    inbox.poll(http)                          # startup: skip what is already there
    assert got == []
    fake.add(5, say("I'm on my way"))
    inbox.poll(http)
    inbox.poll(http)                          # same rows again: nothing new
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "I'm on my way"]
    assert "source = 'caregiver'" in fake.sql[0]


def test_priority_is_clamped_to_info():
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(2, say("stop now", priority="critical"), priority="critical")
    fake.add(3, say("fyi", priority="system"))
    inbox.poll(http)
    assert [m.priority for m in got] == ["info", "info"]      # "system" never expires: clamped too


def test_bad_payloads_and_non_say_types_are_ignored():
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(2, "not json")
    fake.add(3, say(""))
    fake.add(4, {"type": "control", "action": "resume", "source": "caregiver"})
    fake.add(5, say("still works"))
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "still works"]
    assert inbox.stats["bad"] == 3


def test_caregiver_message_is_spoken_but_not_mirrored_back(offline_speech):
    app = Earpiece(Cfg(), output="null", network=False, log=lambda *_: None)
    pushed: list = []
    app.bridges.append(type("B", (), {"publish": lambda self, o: pushed.append(o)})())
    said: list = []
    app.say = lambda text, priority="info", **kw: said.append((text, priority))
    fake = FakeSpacetime()
    inbox = CaregiverInbox(app.handle, host="http://st.test", log=lambda *_: None, start=False, token="")
    http = httpx.Client(transport=httpx.MockTransport(fake))
    inbox.poll(http)
    fake.add(7, say("Call me when you get there", key="caregiver-1"))
    inbox.poll(http)
    assert said == [(CAREGIVER_PREFIX + "Call me when you get there", "info")]
    assert not [o for o in pushed if o.get("source") == "caregiver"]



# ---------------------------------------------------------------- overnight review (H1, M2, M5)
@pytest.mark.parametrize("text", ["It is safe to cross now.", "Go ahead.", "Cross.", "Street is empty.",
                                  "Nothing coming, cross.", "You can continue."])
def test_permission_from_the_caregiver_channel_is_never_spoken(text):
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(2, say(text))
    fake.add(3, say("I'm on my way."))
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "I'm on my way."]
    assert inbox.stats["rejected"] == 1


def test_key_and_ttl_come_from_the_earpiece_not_the_row():
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(4, say("Call me when you get there.", key="continue", ttl=0.5))   # a safety line's key
    inbox.poll(http)
    (m,) = got
    assert m.key == "caregiver-4" and m.ttl == CAREGIVER_TTL_S >= 20


def sign(token, text, ts=None):
    ts = int(time.time() * 1000) if ts is None else ts
    return {"ts": ts, "sig": hmac.new(token.encode(), f"{ts}:{text}".encode(), hashlib.sha256).hexdigest()}


def test_with_a_token_only_signed_fresh_rows_are_spoken():
    fake = FakeSpacetime()
    inbox, got, http = make(fake, token="s3cret")
    inbox.poll(http)
    fake.add(2, say("unsigned"))
    fake.add(3, say("wrong key", **sign("guess", "wrong key")))
    fake.add(4, say("other text", **sign("s3cret", "signed text")))           # sig for a different text
    fake.add(5, say("stale", **sign("s3cret", "stale", ts=int((time.time() - 600) * 1000))))
    fake.add(6, say("I'm on my way.", **sign("s3cret", "I'm on my way.")))
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "I'm on my way."]
    assert inbox.stats["unsigned"] == 4


def test_a_copied_signed_row_is_spoken_once():
    """The event table is public: anyone can re-push a caregiver's signed row while it is fresh."""
    fake = FakeSpacetime()
    inbox, got, http = make(fake, token="s3cret")
    inbox.poll(http)
    signed = say("I'm waiting at the car.", **sign("s3cret", "I'm waiting at the car."))
    for rid in range(2, 12):                       # the original and nine copies
        fake.add(rid, signed)
    inbox.poll(http)
    assert len(got) == 1 and inbox.stats["replayed"] == 9


@pytest.mark.parametrize("ts", [10 ** 400, -(10 ** 400), True, 2 ** 70, "123", 1.5e12])
def test_absurd_signature_timestamps_are_unsigned_not_a_failed_poll(ts):
    fake = FakeSpacetime()
    inbox, got, http = make(fake, token="s3cret")
    inbox.poll(http)
    fake.add(2, say("hi", ts=ts, sig="00"))
    fake.add(3, say("I'm on my way.", **sign("s3cret", "I'm on my way.")))
    inbox.poll(http)                               # must not raise (OverflowError used to)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "I'm on my way."]
    assert inbox.stats["unsigned"] == 1


# ---------------------------------------------------------------- edge cases: startup, ordering, failures
def test_an_empty_table_at_startup_speaks_the_very_first_row():
    """With nothing to skip, the first message ever sent (row id 0) is news, not history."""
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    assert inbox.last_id == -1
    fake.add(0, say("I'm on my way."))
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "I'm on my way."]


def test_after_startup_only_newer_rows_are_asked_for():
    fake = FakeSpacetime()
    fake.add(41, say("old"))
    inbox, _, http = make(fake)
    inbox.poll(http)
    inbox.poll(http)
    assert " AND id > " not in fake.sql[0]                # startup reads everything once
    assert fake.sql[1].endswith(" AND id > 41")


def test_rows_out_of_order_or_repeated_are_spoken_once_in_id_order():
    """The SQL reply has no ORDER BY, and a server that ignores the id filter must not cause repeats."""
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(9, say("third"))
    fake.add(3, say("first"))
    fake.add(5, say("second"))
    inbox.poll(http)
    inbox.poll(http)                                      # FakeSpacetime returns every row again
    assert [m.text for m in got] == [CAREGIVER_PREFIX + t for t in ("first", "second", "third")]
    assert inbox.last_id == 9


def test_a_rejected_row_is_not_reconsidered_on_the_next_poll():
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(2, say("Go ahead."))
    inbox.poll(http)
    inbox.poll(http)
    assert got == [] and inbox.stats["rejected"] == 1


def test_with_a_token_a_say_without_ts_or_sig_is_unsigned():
    fake = FakeSpacetime()
    inbox, got, http = make(fake, token="s3cret")
    inbox.poll(http)
    fake.add(2, say("no signature at all"))
    fake.add(3, say("sig but no ts", sig="00"))
    fake.add(4, say("ts but no sig", ts=int(time.time() * 1000)))
    inbox.poll(http)
    assert got == [] and inbox.stats["unsigned"] == 3


def test_a_signed_permission_is_still_rejected():
    """Signing proves who sent it, not that it is safe: the permission filter still applies."""
    fake = FakeSpacetime()
    inbox, got, http = make(fake, token="s3cret")
    inbox.poll(http)
    fake.add(2, say("You can cross now.", **sign("s3cret", "You can cross now.")))
    inbox.poll(http)
    assert got == [] and inbox.stats["rejected"] == 1


def test_the_token_comes_from_the_environment_and_is_trimmed(monkeypatch):
    import earpiece.spacetime as st
    monkeypatch.setattr(st, "_load_env", lambda: None)    # never read the repo's .env in a test
    monkeypatch.setenv("KITE_CAREGIVER_TOKEN", "  s3cret \n")
    inbox = CaregiverInbox(lambda m: None, host="http://st.test", db="kite", log=lambda *_: None, start=False)
    assert inbox.secret == "s3cret"
    monkeypatch.setenv("KITE_CAREGIVER_TOKEN", "")
    logs: list[str] = []
    inbox = CaregiverInbox(lambda m: None, host="http://st.test", db="kite", log=logs.append, start=False)
    assert inbox.secret == "" and any("not set" in line for line in logs)


def test_polling_is_never_faster_than_the_floor():
    import earpiece.spacetime as st
    fast = CaregiverInbox(lambda m: None, host="http://st.test", db="kite", log=lambda *_: None,
                          start=False, token="", interval=0.01)
    assert fast.interval == st.MIN_POLL_S


def test_the_replay_cache_forgets_signatures_long_past_their_age(monkeypatch):
    """Bounded memory on a long walk: a signature is kept for twice the max age, then dropped."""
    import earpiece.spacetime as st
    inbox, _, _ = make(FakeSpacetime(), token="s3cret")
    now = [1_000.0]
    monkeypatch.setattr(st.time, "time", lambda: now[0])
    assert inbox._replayed("aa") is False
    assert inbox._replayed("aa") is True
    now[0] += 2 * st.CAREGIVER_MAX_AGE_S + 1
    assert inbox._replayed("bb") is False                 # prunes "aa"
    assert "aa" not in inbox._seen_sigs


def test_a_failed_poll_keeps_the_position_and_the_next_one_recovers():
    fake = FakeSpacetime()
    inbox, got, http = make(fake)
    inbox.poll(http)
    fake.add(2, say("I'm on my way."))
    down = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503)))
    inbox._token = None                                   # force a fresh identity call too
    with pytest.raises(httpx.HTTPStatusError):
        inbox.poll(down)
    assert inbox.last_id == -1 and got == []
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "I'm on my way."]


class _Stop(Exception):
    pass


def test_the_poll_thread_backs_off_logs_sparingly_and_drops_its_token(monkeypatch):
    """A dead network never spins the thread or floods the log; a good poll resets the backoff."""
    import earpiece.spacetime as st
    sql_calls = {"n": 0}
    identity_calls = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/identity":
            identity_calls["n"] += 1
            return httpx.Response(200, json={"identity": "x", "token": "tok"})
        sql_calls["n"] += 1
        if sql_calls["n"] <= 5:
            return httpx.Response(500)
        return httpx.Response(200, json=[{"schema": SCHEMA, "rows": []}])

    real_client = httpx.Client
    monkeypatch.setattr(st.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(flaky)))
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        if len(sleeps) >= 7:
            raise _Stop

    monkeypatch.setattr(st.time, "sleep", sleep)
    logs: list[str] = []
    inbox = CaregiverInbox(lambda m: None, host="http://st.test", db="kite", log=logs.append, start=False,
                           token="", interval=1.0)
    logs.clear()
    with pytest.raises(_Stop):
        inbox._run()
    assert inbox.stats["failed"] == 5 and inbox.stats["polls"] == 2
    assert len([line for line in logs if "HTTPStatusError" in line]) == 3      # first three only
    assert identity_calls["n"] == 6                       # the token is dropped after each failure
    assert sleeps[:5] == [2.0, 4.0, 8.0, 16.0, 30.0]      # doubling, capped at MAX_BACKOFF_S
    assert sleeps[5:] == [1.0, 1.0]                       # back to the normal interval once it works
