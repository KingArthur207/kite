// Kite site: the walk player (waveform, live captions) and the sound legend.
(() => {
  const audio = document.getElementById("walk");
  const btn = document.getElementById("play");
  const canvas = document.getElementById("wave");
  const marks = document.getElementById("marks");
  const caption = document.getElementById("caption");
  const transcript = document.getElementById("transcript");
  const label = { critical: "STOP", hazard: "WARNING", nav: "DIRECTIONS", system: "STATUS", answer: "ANSWER", info: "INFO" };
  let data = null;
  let current = -1;

  const fmt = (t) => `0:${String(Math.floor(t)).padStart(2, "0")}`;

  function draw(progress = 0) {
    if (!data) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const n = data.peaks.length, bw = w / n;
    const max = Math.max(...data.peaks, 0.01);
    for (let i = 0; i < n; i++) {
      const v = Math.max(0.04, data.peaks[i] / max);
      const bh = v * (h - 8);
      ctx.fillStyle = i / n <= progress ? "#1E2A5E" : "#D8D0BF";
      ctx.fillRect(i * bw + 0.5, (h - bh) / 2, Math.max(1, bw - 1.2), bh);
    }
  }

  async function load() {
    try {
      data = await (await fetch("/assets/walk.json")).json();
    } catch { return; }
    data.said.forEach((s) => {
      const m = document.createElement("span");
      m.className = "mark" + (s.priority === "critical" ? " critical" : "");
      m.style.left = `${(s.t / data.duration) * 100}%`;
      marks.appendChild(m);
      const li = document.createElement("li");
      li.innerHTML = `<span class="t">${fmt(s.t)}</span>`;
      li.append(document.createTextNode(s.text));
      transcript.appendChild(li);
    });
    draw(0);
  }

  function onTime() {
    if (!data) return;
    const t = audio.currentTime;
    draw(t / data.duration);
    let idx = -1;
    data.said.forEach((s, i) => { if (t >= s.t && t <= s.end + 1.2) idx = i; });
    if (idx !== current) {
      current = idx;
      if (idx >= 0) {
        const s = data.said[idx];
        caption.innerHTML = "";
        const who = document.createElement("span");
        who.className = "who" + (s.priority === "critical" ? " critical" : "");
        who.textContent = `${fmt(s.t)} · ${label[s.priority] || "KITE"}`;
        caption.append(who, document.createTextNode(`“${s.text}”`));
      } else if (t > 0.2) {
        caption.innerHTML = '<span class="who">LISTEN</span>Beeps from the direction of what’s ahead; faster means closer.';
      }
    }
  }

  btn.addEventListener("click", () => {
    stopChip();
    if (audio.paused) audio.play(); else audio.pause();
  });
  audio.addEventListener("play", () => btn.setAttribute("aria-pressed", "true"));
  audio.addEventListener("pause", () => btn.setAttribute("aria-pressed", "false"));
  audio.addEventListener("ended", () => btn.setAttribute("aria-pressed", "false"));
  audio.addEventListener("timeupdate", onTime);
  canvas.parentElement.addEventListener("click", (e) => {
    if (!data) return;
    const r = canvas.getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * data.duration;
    if (audio.paused) audio.play();
  });
  window.addEventListener("resize", () => draw(data ? audio.currentTime / data.duration : 0));

  // ---------------------------------------------------------------- sound legend
  const clip = new Audio();
  let active = null;
  function stopChip() {
    clip.pause();
    if (active) active.classList.remove("playing");
    active = null;
  }
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.setAttribute("aria-label", `Play: ${chip.textContent}`);
    chip.addEventListener("click", () => {
      const same = active === chip;
      stopChip();
      if (same) return;
      audio.pause();
      clip.src = `/assets/audio/${chip.dataset.src}.m4a`;
      clip.play();
      active = chip;
      chip.classList.add("playing");
    });
  });
  clip.addEventListener("ended", stopChip);

  load();
})();
