"""The Grok caregiver line: tools over a fake SpacetimeDB, and the realtime protocol over a fake
connection and a local WebSocket server. No network unless KITE_LIVE_TESTS=1."""

import asyncio
import base64
import json
import os

import httpx
import pytest

from earpiece import caregiver_voice as cv
from earpiece.spacetime import SpacetimeBridge

NOW = 1_800_000_000.0
COLS = {"event": ("id", "type", "source", "priority", "payload", "t"),
        "latest": ("key", "type", "source", "priority", "payload", "t")}


class FakeSpacetime:
    """/v1/identity, /sql (latest and event tables) and /call/push, like Maincloud."""

    def __init__(self):
        self.latest: dict[str, list] = {}
        self.events: list[list] = []
        self.pushes: list[dict] = []
        self.sql: list[str] = []
        self.down = False

    def add(self, type_, source, payload, age_s, priority="info", event=True, latest=True):
        p = json.dumps({"type": type_, "source": source, **payload})
        t = [int((NOW - age_s) * 1e6)]
        if event:
            self.events.append([len(self.events) + 1, type_, source, priority, p, t])
        if latest:
            self.latest[f"{type_}:{source}"] = [f"{type_}:{source}", type_, source, priority, p, t]

    def _reply(self, table, rows):
        schema = {"elements": [{"name": {"some": n}} for n in COLS[table]]}
        return httpx.Response(200, json=[{"schema": schema, "rows": rows}])

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("offline")
        if req.url.path == "/v1/identity":
            return httpx.Response(200, json={"identity": "x", "token": "tok"})
        assert req.headers["authorization"] == "Bearer tok"
        if req.url.path.endswith("/call/push"):
            self.pushes.append(json.loads(req.content))
            return httpx.Response(200)
        q = req.content.decode()
        self.sql.append(q)
        if "FROM latest" in q:
            return self._reply("latest", list(self.latest.values()))
        newer = int(q.split("id > ")[1]) if "id > " in q else -1
        types = set(cv.HISTORY_TYPES)
        return self._reply("event", [r for r in self.events if r[0] > newer and r[1] in types])


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def make_link(fake=None, clock=None, token=""):
    fake = fake or FakeSpacetime()
    http = httpx.Client(transport=httpx.MockTransport(fake))
    bridge = SpacetimeBridge("http://st.test", "kite", source="caregiver", log=lambda *_: None,
                             start=False, max_calls_per_s=1000, backoff_s=0.001)
    link = cv.KiteLink("http://st.test", "kite", http=http, bridge=bridge, clock=clock or Clock(),
                       log=lambda *_: None, cache_s=0, token=token)
    return link, fake


def tools(fake=None, clock=None):
    link, fake = make_link(fake, clock)
    return cv.Tools(link, log=lambda *_: None), fake


def live_walk(fake: FakeSpacetime):
    fake.add("position", "sim", {"place": "Lyon Street", "lat": 37.80171, "lng": -122.44792}, 5, event=False)
    fake.add("beacon", "route", {"label": "Palace of Fine Arts", "distance": 131.4, "bearing": -2}, 4, event=False)
    fake.add("guidance", "route", {"text": "Turn left onto Bay Street."}, 30, priority="nav")
    fake.add("hazard", "drone", {"text": "Small crowd ahead on the right.", "kind": "crowd"}, 60, priority="hazard")
    fake.add("spoken", "earpiece", {"text": "Small crowd ahead on the right."}, 58, priority="hazard")
    fake.add("status", "drone", {"state": "following", "battery": 41}, 3, event=False)


# ------------------------------------------------------------------ message_walker

def test_message_walker_pushes_the_dashboards_exact_reducer_call():
    t, fake = tools()
    out = json.loads(t.call("message_walker", json.dumps({"text": "I'm running five minutes late"})))
    assert out["sent"] is True
    assert out["walker_hears"] == "Message from your caregiver: I'm running five minutes late"
    [call] = fake.pushes
    assert set(call) == {"type", "source", "priority", "payload"}
    assert (call["type"], call["source"], call["priority"]) == ("say", "caregiver", "info")
    payload = json.loads(call["payload"])
    # same keys, same order as dashboard/src/App.tsx send()
    assert list(payload) == ["type", "source", "priority", "text", "key"]
    assert payload["text"] == "I'm running five minutes late"
    assert payload["key"] == f"caregiver-{int(NOW * 1000)}"
    assert call["payload"] == json.dumps(payload, separators=(",", ":"))   # JSON.stringify's spacing


def test_message_walker_lands_where_the_earpiece_inbox_reads_it():
    from earpiece.events import parse
    t, fake = tools()
    t.call("message_walker", '{"text": "Dinner is at seven"}')
    msg = parse(fake.pushes[0]["payload"])
    assert (msg.type, msg.source, msg.priority, msg.text) == ("say", "caregiver", "info", "Dinner is at seven")


def test_message_text_is_cleaned_and_clamped_to_200_chars():
    t, fake = tools()
    long = "please   pick up\tbread\n" + "and milk " * 60
    out = json.loads(t.call("message_walker", json.dumps({"text": long})))
    assert out["sent"] is True
    text = json.loads(fake.pushes[0]["payload"])["text"]
    assert len(text) <= cv.MAX_MESSAGE
    assert text.startswith("please pick up bread and milk")
    assert "  " not in text and "\t" not in text and "\n" not in text
    assert not text.endswith(" ")


@pytest.mark.parametrize("args", ['{"text": ""}', '{"text": "   \\n "}', '{"text": 42}', '{}',
                                  '[1, 2]', 'not json', '{"text": null}'])
def test_bad_message_arguments_are_refused_and_nothing_is_sent(args):
    t, fake = tools()
    out = json.loads(t.call("message_walker", args))
    assert out.get("sent") is not True and "error" in out
    assert fake.pushes == []


@pytest.mark.parametrize("text", ["Tell her it's safe to cross", "You can cross now", "the light is green, go ahead and cross",
                                  "It's clear, cross the street"])
def test_crossing_go_aheads_are_never_relayed(text):
    t, fake = tools()
    out = json.loads(t.call("message_walker", json.dumps({"text": text})))
    assert out["sent"] is False and "cross" in out["error"]
    assert fake.pushes == []


def test_ordinary_messages_mentioning_crossing_still_go_through():
    t, fake = tools()
    assert json.loads(t.call("message_walker", '{"text": "Take your time at the crossing, no rush"}'))["sent"]
    assert len(fake.pushes) == 1


def test_double_calls_and_bursts_do_not_spam_the_walker():
    clock = Clock()
    t, fake = tools(clock=clock)
    assert json.loads(t.call("message_walker", '{"text": "On my way"}'))["sent"]
    clock.t += 0.5
    burst = json.loads(t.call("message_walker", '{"text": "Something else"}'))
    assert burst["sent"] is False
    clock.t += 3
    again = json.loads(t.call("message_walker", '{"text": "on my way"}'))
    assert again["sent"] is True and "not sent twice" in again["note"]
    assert len(fake.pushes) == 1


def test_a_rejected_push_is_reported_as_not_sent():
    t, fake = tools()
    fake.down = True
    out = json.loads(t.call("message_walker", '{"text": "hello"}'))
    assert out["sent"] is False and "error" in out


def test_dry_run_never_pushes():
    link, fake = make_link()
    t = cv.Tools(link, log=lambda *_: None, dry_run=True)
    assert json.loads(t.call("message_walker", '{"text": "hello"}'))["sent"] is False
    assert fake.pushes == []


@pytest.mark.parametrize("line_token, spoken", [("s3cret", True), ("", False), ("other", False)])
def test_an_earpiece_that_requires_the_token_speaks_only_signed_line_messages(line_token, spoken):
    """Found by the headless end-to-end run: with KITE_CAREGIVER_TOKEN set on the earpiece, the
    line's messages were pushed unsigned, so Grok said "sent" and the walker never heard them."""
    import time

    from earpiece.spacetime import CaregiverInbox
    fake = FakeSpacetime()
    link, _ = make_link(fake, clock=time.time, token=line_token)
    heard: list = []
    inbox = CaregiverInbox(heard.append, "http://st.test", "kite", log=lambda *_: None, start=False, token="s3cret")
    http = httpx.Client(transport=httpx.MockTransport(fake))
    inbox.poll(http)                                         # startup: older rows are history
    assert json.loads(cv.Tools(link, log=lambda *_: None).call(
        "message_walker", '{"text": "Running five minutes late"}'))["sent"] is True
    [call] = fake.pushes
    fake.events.append([1, call["type"], call["source"], call["priority"], call["payload"], [int(time.time() * 1e6)]])
    inbox.poll(http)
    assert [m.text for m in heard] == (["Message from your caregiver: Running five minutes late"] if spoken else [])
    assert inbox.stats["unsigned"] == (0 if spoken else 1)
    payload = json.loads(call["payload"])
    want = ["type", "source", "priority", "text", "key"] + (["ts", "sig"] if line_token else [])
    assert list(payload) == want                             # the dashboard's order, then its signature
    if line_token:
        assert payload["key"] == f"caregiver-{payload['ts']}"


# ------------------------------------------------------------------ walk_status and alerts

def test_walk_status_live_walk():
    t, fake = tools()
    live_walk(fake)
    s = json.loads(t.call("walk_status", "{}"))
    assert s["live"] is True and "note" not in s
    assert s["position"]["place"] == "Lyon Street" and s["position"]["stale"] is False
    assert s["route"]["next_waypoint"] == "Palace of Fine Arts" and s["route"]["distance_m"] == 131
    assert s["route"]["last_instruction"] == "Turn left onto Bay Street."
    assert s["recent_hazards"][0]["what"] == "Small crowd ahead on the right."
    assert s["kite_said"][0]["text"] == "Small crowd ahead on the right."
    assert s["devices"]["drone"] == {"age": "3 seconds", "stale": False, "state": "following", "battery_pct": 41}
    assert len(t.call("walk_status", "{}")) < 2000          # compact enough to speak from


def test_walk_status_says_plainly_when_data_is_stale():
    t, fake = tools()
    fake.add("position", "sim", {"place": "Moss Street"}, 20 * 60, event=False)
    fake.add("status", "drone", {"state": "following", "battery": 28}, 20 * 60 + 5, event=False)
    s = json.loads(t.call("walk_status", "{}"))
    assert s["live"] is False
    assert s["last_update"] == "20 minutes ago"
    assert "20 minutes" in s["note"] and "old" in s["note"]
    assert s["position"]["stale"] is True and s["position"]["age"] == "20 minutes"
    assert s["devices"]["drone"]["stale"] is True


def test_walk_status_does_not_mix_in_rows_from_an_earlier_walk():
    t, fake = tools()
    fake.add("position", "sim", {"place": "Moss Street"}, 10, event=False)
    fake.add("beacon", "test-walk", {"label": "Palace of Fine Arts", "distance": 7}, 2 * 3600, event=False)
    fake.add("guidance", "test-walk", {"text": "Turn left onto Bay Street."}, 2 * 3600, priority="nav")
    fake.add("status", "test-drone", {"state": "following", "battery": 28}, 2 * 3600, event=False)
    s = json.loads(t.call("walk_status", "{}"))
    assert s["position"]["place"] == "Moss Street"
    assert "next_waypoint" not in s["route"] and "last_instruction" not in s["route"]
    assert "drone" not in s["devices"]
    assert s["not_reporting_this_walk"] == {"drone": "last heard 2 hours ago"}


def test_walk_status_with_an_empty_database():
    t, _ = tools()
    s = json.loads(t.call("walk_status", "{}"))
    assert s["live"] is False and "no data" in s["note"]


def test_walk_status_reports_unreachable_data_as_an_error():
    t, fake = tools()
    fake.down = True
    assert "error" in json.loads(t.call("walk_status", "{}"))


def test_history_is_fetched_incrementally():
    link, fake = make_link()
    live_walk(fake)
    cv.walk_status(link)
    fake.add("hazard", "drone", {"text": "Construction ahead."}, 1, priority="hazard")
    s = cv.walk_status(link)
    assert s["recent_hazards"][0]["what"] == "Construction ahead."
    history_q = [q for q in fake.sql if "FROM event" in q]
    assert "id >" not in history_q[0] and history_q[1].endswith(f"id > {len(fake.events) - 1}")


def test_reads_are_cached_to_save_credits():
    link, fake = make_link()
    link.cache_s = 3.0
    live_walk(fake)
    t = cv.Tools(link, log=lambda *_: None)
    for _ in range(4):
        t.call("walk_status", "{}")
    assert len(fake.sql) == 2                                # one latest + one history read


def test_recent_alerts_surface_help_requests_and_clamp_minutes():
    t, fake = tools()
    live_walk(fake)
    fake.add("caregiver_alert", "earpiece", {"message": "I'm lost, please call me",
                                              "position": {"place": "Lyon Street"}}, 90)
    fake.add("caregiver_alert", "earpiece", {"message": "old one"}, 5 * 3600)
    fake.add("say", "earpiece", {"text": "Stop. Car backing out on your left."}, 120, priority="critical")
    out = json.loads(t.call("recent_alerts", '{"minutes": 30}'))
    assert out["alerts"] == [{"message": "I'm lost, please call me", "age": "2 minutes ago", "place": "Lyon Street"}]
    assert out["stop_warnings"][0]["what"] == "Stop. Car backing out on your left."
    assert json.loads(t.call("walk_status", "{}"))["alert"].startswith("The walker asked to alert you 2 minutes ago")
    for given, want in (('{"minutes": -5}', 1), ('{"minutes": 1e9}', 1440), ('{"minutes": "45"}', 45),
                        ('{"minutes": true}', 120), ('{"minutes": "soon"}', 120), ('{}', 120)):
        assert json.loads(t.call("recent_alerts", given))["minutes_checked"] == want
    assert len(json.loads(t.call("recent_alerts", '{"minutes": 1440}'))["alerts"]) == 2


def test_no_alerts_is_said_plainly():
    t, _ = tools()
    out = json.loads(t.call("recent_alerts", "{}"))
    assert out["alerts"] == [] and "has not asked" in out["note"]


def test_unknown_tool():
    t, _ = tools()
    assert "unknown tool" in json.loads(t.call("open_the_door", "{}"))["error"]


# ------------------------------------------------------------------ the realtime protocol

class FakeWS:
    """An in-memory connection: what the client sent, and a queue of server events."""

    def __init__(self):
        self.sent: list[dict] = []
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def push(self, ev):
        self.inbox.put_nowait(json.dumps(ev) if isinstance(ev, dict) else ev)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self):
        self.push(None)


def sent_types(ws):
    return [m["type"] for m in ws.sent]


def test_session_update_carries_voice_tools_and_audio_format():
    async def go():
        ws = FakeWS()
        s = cv.Session(ws, tools()[0], voice="eve", log=lambda *_: None)
        await s.configure()
        cfg = ws.sent[0]["session"]
        assert ws.sent[0]["type"] == "session.update"
        assert cfg["voice"] == "eve" and cfg["turn_detection"] == {"type": "server_vad"}
        assert [t["name"] for t in cfg["tools"]] == ["walk_status", "message_walker", "recent_alerts"]
        assert all(t["type"] == "function" for t in cfg["tools"])
        assert cfg["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert "never" in cfg["instructions"].lower() and "safe to cross" in cfg["instructions"]
        text = cv.Session(FakeWS(), tools()[0], text_mode=True, log=lambda *_: None)
        assert text.session_config()["turn_detection"] is None
    asyncio.run(go())


def test_parallel_tool_calls_get_outputs_then_one_response_create():
    async def go():
        ws = FakeWS()
        t, fake = tools()
        live_walk(fake)
        s = cv.Session(ws, t, log=lambda *_: None)
        await s.handle({"type": "response.created"})
        await s.handle({"type": "response.function_call_arguments.done", "name": "walk_status",
                        "call_id": "c1", "arguments": "{}"})
        await s.handle({"type": "response.function_call_arguments.done", "name": "recent_alerts",
                        "call_id": "c2", "arguments": '{"minutes": 60}'})
        assert sent_types(ws) == ["conversation.item.create", "conversation.item.create"]   # no early create
        outs = [m["item"] for m in ws.sent]
        assert [o["type"] for o in outs] == ["function_call_output"] * 2
        assert [o["call_id"] for o in outs] == ["c1", "c2"]
        assert json.loads(outs[0]["output"])["position"]["place"] == "Lyon Street"
        assert json.loads(outs[1]["output"])["minutes_checked"] == 60
        await s.handle({"type": "response.done", "response": {"status": "completed"}})
        assert sent_types(ws)[-1] == "response.create" and sent_types(ws).count("response.create") == 1
        assert not s.idle.is_set()                           # the spoken answer is still to come
        await s.handle({"type": "response.done", "response": {"status": "completed"}})
        assert s.idle.is_set() and sent_types(ws).count("response.create") == 1
    asyncio.run(go())


def test_audio_and_transcripts_are_collected_and_unknown_events_ignored():
    async def go():
        ws = FakeWS()
        rec: list = []
        s = cv.Session(ws, tools()[0], record=rec, log=lambda *_: None)
        pcm = b"\x01\x00" * 240
        await s.handle({"type": "session.updated", "session": {"audio": {"output": {
            "format": {"type": "audio/pcm", "rate": 16000}}}}})
        assert s.ready.is_set() and s.out_format == {"type": "audio/pcm", "rate": 16000}
        await s.handle({"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
        await s.handle({"type": "response.output_audio_transcript.delta", "delta": "She is on "})
        await s.handle({"type": "response.output_audio_transcript.delta", "delta": "Lyon Street."})
        await s.handle({"type": "response.output_audio_transcript.done"})
        await s.handle({"type": "some.future.event", "x": 1})
        await s.handle({"type": "error", "error": {"message": "bad tool schema"}})
        assert rec == [pcm]
        assert s.transcript == [("kite", "She is on Lyon Street.")]
        assert s.events_seen["some.future.event"] == 1 and ws.sent == []
    asyncio.run(go())


def test_function_call_without_call_id_is_ignored():
    async def go():
        ws = FakeWS()
        s = cv.Session(ws, tools()[0], log=lambda *_: None)
        await s.handle({"type": "response.function_call_arguments.done", "name": "walk_status", "arguments": "{}"})
        await s.handle({"type": "response.done"})
        assert ws.sent == [] and s.idle.is_set()
    asyncio.run(go())


def test_new_walker_alert_is_announced_unasked():
    async def go():
        ws = FakeWS()
        link, fake = make_link()
        live_walk(fake)
        fake.add("caregiver_alert", "earpiece", {"message": "old, already known"}, 600)
        s = cv.Session(ws, cv.Tools(link, log=lambda *_: None), log=lambda *_: None)
        logs: list = []
        watcher = asyncio.create_task(cv.watch_alerts(s, link, every=0.01, log=logs.append))
        await asyncio.sleep(0.05)
        assert ws.sent == []                                 # rows from before the call are history
        fake.add("caregiver_alert", "earpiece", {"message": "Please call me", "position": {"place": "Bay Street"}}, 0)
        for _ in range(100):
            if ws.sent:
                break
            await asyncio.sleep(0.01)
        s.closed.set()
        watcher.cancel()
        assert sent_types(ws) == ["conversation.item.create", "response.create"]
        text = ws.sent[0]["item"]["content"][0]["text"]
        assert text.startswith("[Kite notice") and "Please call me" in text and "Bay Street" in text
        assert any("ALERT" in m for m in logs)
    asyncio.run(go())


def test_an_oversized_frame_ends_the_session_instead_of_filling_memory(monkeypatch):
    """The realtime socket has a frame cap: a runaway frame closes the line (and every wait
    on it is released) rather than being buffered whole."""
    from websockets.asyncio.server import serve

    monkeypatch.setattr(cv, "MAX_FRAME", 4096)
    t, _ = tools()

    async def server(ws):
        await ws.send(json.dumps({"type": "response.output_text.delta", "delta": "x" * 10_000}))
        await ws.wait_closed()

    async def go():
        async with serve(server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            ws = await cv._connect(f"ws://127.0.0.1:{port}/", "test-key")
            logs: list = []
            s = cv.Session(ws, t, text_mode=True, log=logs.append)
            await asyncio.wait_for(s.run(), 5)
            return s, logs

    s, logs = asyncio.run(go())
    assert s.closed.is_set() and s.idle.is_set() and s.ready.is_set()
    assert any("connection ended" in m for m in logs) and s._said == ""


def test_the_model_name_is_url_encoded(monkeypatch):
    """--model goes into the query string; anything odd in it can't add parameters."""
    seen: list = []

    async def fake_connect(url, key):
        seen.append(url)
        raise OSError("no network in tests")

    monkeypatch.setattr(cv, "_connect", fake_connect)
    monkeypatch.setattr(cv, "_xai_key", lambda: "test-key")
    from earpiece.__main__ import build_parser
    a = build_parser().parse_args(["caregiver", "--text", "--model", "grok voice&key=1#x"])
    monkeypatch.setattr(cv, "KiteLink", lambda **k: object())
    assert asyncio.run(cv.run_line(a, log=lambda *_: None)) == 1
    assert seen == [f"{cv.XAI_URL}?model=grok+voice%26key%3D1%23x"]


def test_round_trip_over_a_local_websocket_server():
    """Text turn -> tool call -> output -> response.create -> audio, against a scripted server."""
    from websockets.asyncio.server import serve

    t, fake = tools()
    got: dict = {}

    async def server(ws):
        got["auth"] = ws.request.headers.get("Authorization")
        upd = json.loads(await ws.recv())
        got["update"] = upd
        await ws.send(json.dumps({"type": "session.updated", "session": {"audio": {"output": {
            "format": {"type": "audio/pcm", "rate": 24000}}}}}))
        item = json.loads(await ws.recv())
        assert json.loads(await ws.recv())["type"] == "response.create"
        got["user"] = item["item"]["content"][0]["text"]
        await ws.send(json.dumps({"type": "response.created"}))
        await ws.send(json.dumps({"type": "response.function_call_arguments.done", "name": "message_walker",
                                  "call_id": "call_1", "arguments": json.dumps({"text": "Running five minutes late"})}))
        await ws.send(json.dumps({"type": "response.done", "response": {"status": "completed"}}))
        out = json.loads(await ws.recv())
        got["output"] = out
        got["next"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "response.created"}))
        await ws.send(json.dumps({"type": "response.output_audio.delta",
                                  "delta": base64.b64encode(b"\x00\x01" * 100).decode()}))
        await ws.send(json.dumps({"type": "response.output_audio_transcript.done", "transcript": "Sent."}))
        await ws.send(json.dumps({"type": "response.done", "response": {"status": "completed"}}))
        await ws.wait_closed()

    async def go():
        async with serve(server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            ws = await cv._connect(f"ws://127.0.0.1:{port}/v1/realtime?model=x", "test-key")
            rec: list = []
            s = cv.Session(ws, t, text_mode=True, record=rec, log=lambda *_: None)
            reader = asyncio.create_task(s.run())
            await s.configure()
            await asyncio.wait_for(s.ready.wait(), 5)
            await s.say_text("Tell her I'm running five minutes late")
            await asyncio.wait_for(s.idle.wait(), 5)
            await ws.close()
            await asyncio.wait_for(reader, 5)
            return s, rec

    s, rec = asyncio.run(go())
    assert got["auth"] == "Bearer test-key"
    assert got["update"]["type"] == "session.update" and got["update"]["session"]["turn_detection"] is None
    assert got["user"] == "Tell her I'm running five minutes late"
    assert got["output"]["item"]["type"] == "function_call_output" and got["output"]["item"]["call_id"] == "call_1"
    assert json.loads(got["output"]["item"]["output"])["sent"] is True
    assert got["next"] == {"type": "response.create"}
    assert json.loads(fake.pushes[0]["payload"])["text"] == "Running five minutes late"
    assert b"".join(rec) == b"\x00\x01" * 100
    assert s.transcript[-1] == ("kite", "Sent.")


def test_cli_dry_run_prints_tools_and_never_connects(monkeypatch, capsys):
    link, fake = make_link()
    live_walk(fake)
    monkeypatch.setattr(cv, "KiteLink", lambda **kw: link)
    monkeypatch.setattr(cv, "_connect", lambda *a, **k: pytest.fail("dry run must not connect"))
    from earpiece.__main__ import main
    assert main(["caregiver", "--dry-run", "--db", "kite"]) == 0
    out = capsys.readouterr().out
    assert '"walk_status"' in out and "Lyon Street" in out and fake.pushes == []


# ------------------------------------------------------------------ headless: recordings in, files out

def _wav(path, pcm: bytes, rate=24000, channels=1):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return str(path)


def _tone(seconds: float, rate=24000, channels=1) -> bytes:
    import numpy as np
    t = np.arange(int(seconds * rate)) / rate
    x = (8000 * np.sin(2 * np.pi * 300 * t)).astype("<i2")
    return np.repeat(x[:, None], channels, axis=1).tobytes()


def test_read_wav_mixes_down_and_resamples(tmp_path):
    import numpy as np
    pcm = cv.read_wav_pcm16(_wav(tmp_path / "s.wav", _tone(0.5, 48000, 2), 48000, 2), 24000)
    x = np.frombuffer(pcm, dtype="<i2")
    assert x.size == 12000 and 7000 < np.abs(x).max() <= 8000
    same = _tone(0.25)
    assert cv.read_wav_pcm16(_wav(tmp_path / "m.wav", same), 24000) == same
    with pytest.raises(ValueError):
        cv.read_wav_pcm16(_wav(tmp_path / "empty.wav", b""), 24000)


def test_recorded_turn_streams_like_the_mic_then_waits_for_turn_detection():
    async def go():
        ws = FakeWS()
        s = cv.Session(ws, tools()[0], log=lambda *_: None)
        pcm = _tone(0.25)
        turn = asyncio.create_task(cv.send_recorded_turn(s, pcm, tail_s=0.5, pace=0, start_timeout_s=2))
        await asyncio.sleep(0.05)
        await s.handle({"type": "response.created"})          # server VAD answered
        assert await turn is True
        appends = [m for m in ws.sent if m["type"] == "input_audio_buffer.append"]
        assert sent_types(ws) == ["input_audio_buffer.append"] * len(appends)     # no manual commit
        audio = b"".join(base64.b64decode(m["audio"]) for m in appends)
        assert audio == pcm + b"\x00" * (12000 * 2)           # the recording, then 0.5 s of silence
        assert max(len(base64.b64decode(m["audio"])) for m in appends) == 2400 * 2   # 100 ms chunks
    asyncio.run(go())


def test_recorded_turn_commits_by_hand_when_turn_detection_stays_quiet():
    async def go():
        ws = FakeWS()
        logs: list = []
        s = cv.Session(ws, tools()[0], log=logs.append)
        assert await cv.send_recorded_turn(s, _tone(0.1), tail_s=0.1, pace=0, start_timeout_s=0.05,
                                           log=logs.append) is False
        assert sent_types(ws)[-2:] == ["input_audio_buffer.commit", "response.create"]
        assert not s.idle.is_set() and any("by hand" in m for m in logs)
    asyncio.run(go())


def test_headless_cli_never_opens_a_sound_device(monkeypatch, tmp_path):
    """--input-wav turns against a scripted server: tool calls, per-turn WAVs and the transcript,
    with sounddevice unimportable and Audio forbidden."""
    import functools
    import sys
    import wave

    from websockets.asyncio.server import serve

    monkeypatch.setitem(sys.modules, "sounddevice", None)          # any import of it now fails
    monkeypatch.setattr(cv, "Audio", lambda *a, **k: pytest.fail("headless mode opened an audio device"))
    monkeypatch.setattr(cv, "send_recorded_turn", functools.partial(cv.send_recorded_turn, tail_s=0.3, pace=0))
    monkeypatch.setattr(cv, "_xai_key", lambda: "test-key")
    link, fake = make_link(clock=time_clock())
    live_walk(fake)
    monkeypatch.setattr(cv, "KiteLink", lambda **kw: link)
    script = [("walk_status", "{}", "She is on Lyon Street."),
              ("message_walker", json.dumps({"text": "Running five minutes late"}), "Sent.")]

    async def server(ws):
        await ws.recv()                                               # session.update
        await ws.send(json.dumps({"type": "session.updated", "session": {}}))
        for name, args, said in script:
            heard_audio = False
            async for raw in ws:                                      # wait for the silence after speech
                m = json.loads(raw)
                pcm = base64.b64decode(m.get("audio", ""))
                heard_audio |= any(pcm)
                if heard_audio and pcm and not any(pcm):
                    break
            for ev in ({"type": "input_audio_buffer.speech_stopped"},
                       {"type": "conversation.item.input_audio_transcription.completed", "transcript": name},
                       {"type": "response.created"},
                       {"type": "response.function_call_arguments.done", "name": name, "call_id": name,
                        "arguments": args},
                       {"type": "response.done", "response": {"status": "completed"}}):
                await ws.send(json.dumps(ev))
            while json.loads(await ws.recv())["type"] != "response.create":
                pass
            for ev in ({"type": "response.created"},
                       {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x07\x00" * 2400).decode()},
                       {"type": "response.output_audio_transcript.done", "transcript": said},
                       {"type": "response.done", "response": {"status": "completed"}}):
                await ws.send(json.dumps(ev))
        await ws.wait_closed()

    async def go():
        async with serve(server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            real = cv._connect
            monkeypatch.setattr(cv, "_connect", lambda url, key: real(f"ws://127.0.0.1:{port}/", key))
            args = ["caregiver", "--no-greeting", "--alerts", "0", "--output-wav", str(tmp_path / "out.wav"),
                    "--transcript", str(tmp_path / "t.json")]
            for i in (1, 2):
                args += ["--input-wav", _wav(tmp_path / f"q{i}.wav", _tone(0.2))]
            from earpiece.__main__ import build_parser
            return await cv.run_line(build_parser().parse_args(args), log=lambda *_: None)

    assert asyncio.run(go()) == 0
    tr = json.loads((tmp_path / "t.json").read_text())
    assert [[c["name"] for c in t["tools"]] for t in tr["turns"]] == [["walk_status"], ["message_walker"]]
    assert tr["turns"][0]["tools"][0]["result"]["position"]["place"] == "Lyon Street"
    assert tr["turns"][1]["tools"][0]["result"]["sent"] is True
    assert [x["text"] for x in tr["turns"][1]["said"] if x["who"] == "kite"] == ["Sent."]
    assert json.loads(fake.pushes[0]["payload"])["text"] == "Running five minutes late"
    for i in (1, 2):
        with wave.open(str(tmp_path / f"out.turn{i}.wav")) as w:
            assert (w.getframerate(), w.getnframes()) == (24000, 2400)
    with wave.open(str(tmp_path / "out.wav")) as w:
        assert w.getnframes() == 4800


def time_clock():
    import time
    return time.time


def test_headless_flags_that_cannot_work_are_refused(tmp_path):
    from earpiece.__main__ import build_parser
    run = lambda *args: asyncio.run(cv.run_line(build_parser().parse_args(["caregiver", *args]), log=lambda *_: None))
    assert run("--output-wav", str(tmp_path / "o.wav")) == 2                   # nothing to listen with
    assert run("--text", "--input-wav", "x.wav") == 2
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    assert run("--input-wav", str(tmp_path / "bad.wav")) == 2


# ------------------------------------------------------------------ live (opt in)

@pytest.mark.skipif(not (os.environ.get("KITE_LIVE_TESTS") and os.environ.get("XAI_API_KEY")),
                    reason="live Grok test: set KITE_LIVE_TESTS=1 and XAI_API_KEY")
def test_live_grok_text_turn_calls_walk_status():
    """One short text turn against the real API, with a fake SpacetimeDB (nothing is written)."""
    t, fake = tools(clock=Clock(NOW))
    live_walk(fake)

    async def go():
        ws = await cv._connect(f"{cv.XAI_URL}?model={cv.DEFAULT_MODEL}", os.environ["XAI_API_KEY"])
        rec: list = []
        s = cv.Session(ws, t, text_mode=True, record=rec, log=lambda *_: None)
        reader = asyncio.create_task(s.run())
        await s.configure()
        await asyncio.wait_for(s.ready.wait(), 15)
        await s.say_text("Where is she right now?")
        await asyncio.wait_for(s.idle.wait(), 60)
        await ws.close()
        await asyncio.wait_for(reader, 10)
        return s, rec

    s, rec = asyncio.run(go())
    assert ("walk_status", {}) in t.calls
    assert sum(len(c) for c in rec) > 24000                  # at least half a second of 24 kHz PCM16
    assert "lyon" in " ".join(text for who, text in s.transcript if who == "kite").lower()
    assert fake.pushes == []
