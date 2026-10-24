import json

import httpx

from earpiece.app import Earpiece
from earpiece.config import Cfg
from earpiece.spacetime import CAREGIVER_PREFIX, CaregiverInbox

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


def make(fake):
    got = []
    inbox = CaregiverInbox(got.append, host="http://st.test", db="kite", log=lambda *_: None, start=False)
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
    assert [m.priority for m in got] == ["info", "system"]


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
    inbox = CaregiverInbox(app.handle, host="http://st.test", log=lambda *_: None, start=False)
    http = httpx.Client(transport=httpx.MockTransport(fake))
    inbox.poll(http)
    fake.add(7, say("Call me when you can", key="caregiver-1"))
    inbox.poll(http)
    assert said == [(CAREGIVER_PREFIX + "Call me when you can", "info")]
    assert not [o for o in pushed if o.get("source") == "caregiver"]

