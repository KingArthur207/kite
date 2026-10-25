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
