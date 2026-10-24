import json
import wave

import numpy as np

from earpiece.__main__ import main


def test_render_jsonl_writes_binaural_wav_and_captions(offline_speech):
    events = offline_speech / "events.jsonl"
    events.write_text("\n".join(json.dumps(e) for e in [
        {"t": 100.0, "type": "say", "text": "Hello from the timeline.", "priority": "info"},
        {"t": 100.2, "type": "obstacles", "source": "webcam", "seq": 1,
         "objects": [{"id": "p", "kind": "person", "bearing": -40, "distance": 1.5, "confidence": 0.9}]},
        {"t": 100.4, "type": "obstacles", "source": "webcam", "seq": 2,
         "objects": [{"id": "p", "kind": "person", "bearing": -40, "distance": 1.4, "confidence": 0.9}]},
        {"t": 100.5, "type": "bogus"},                       # skipped, never crashes the feed
    ]) + "\n")
    out = offline_speech / "out.wav"

    assert main(["render", "--jsonl", str(events), str(out), "--seconds", "2"]) == 0

    with wave.open(str(out)) as w:
        assert w.getnchannels() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2)
        assert 1.5 < len(pcm) / w.getframerate() < 12
    assert np.abs(pcm).max() > 1000                        # beeps and speech, not silence
    caps = [json.loads(line) for line in (offline_speech / "out.captions.jsonl").read_text().splitlines()]
    hello = [c for c in caps if c["text"] == "Hello from the timeline."]
    assert hello and 0 <= hello[0]["t"] <= hello[0]["end"] < 10
