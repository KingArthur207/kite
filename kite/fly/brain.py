"""The fly's pursuit circuit as an event-driven leaky integrate-and-fire network (DESIGN.md section 3).

    brain = PursuitBrain(seed=0)              # wiring loads once per process (cached by path)
    dn_hz = brain.tick(rates_22)              # 50 ms of neuron time -> 10 DN rates (Hz), OUTPUT_GROUPS order
    other = brain.copy(seed=1)                # same wiring, fresh state and noise (one per episode)

Wiring: the 1,446-neuron pursuit subgraph of the male fruit-fly connectome (MaleCNS v1.0, Janelia FlyEM et al.,
CC-BY 4.0) in data/pursuit_core1.npz, stored presynaptic-major: `indptr` runs over presynaptic neurons, `indices`
are their targets and `data` the signed synapse counts (+ excitatory, - inhibitory).

Neuron model and parameters: Shiu et al., Nature 2024 ("A Drosophila computational brain model reveals sensorimotor
processing"):

    dv/dt = (v_rest - v + g) / tau_m        dg/dt = -g / tau_syn        (both frozen while refractory)
    v > v_th  ->  spike: v = v_reset, g = 0, refractory for t_ref;
                  every target's g gets w_syn * synapse count, `delay` later (it lands even on refractory cells)

External input: each input neuron receives Poisson spikes at its encoder channel's rate (CHANNELS), each adding
w_syn * poisson_factor to its g. LC10a channels map to the 8 azimuth bins per side of meta["azimuth_bins"].

Deviations from Shiu et al., kept on purpose because the previously trained parameters (data/reference/) were fitted
on exactly these dynamics:
- an external Poisson spike adds to the input neuron's g (it reaches threshold through integration, about 1 ms
  later), and input neurons obey the refractory period like every other neuron;
- forward Euler at dt = 0.5 ms (Shiu et al. integrate at 0.1 ms);
- the 1.8 ms delay and the 2.2 ms refractory period both round to 4 steps (2.0 ms).

The synaptic delay is a ring buffer of `delay` steps. Only the rows of neurons that spiked are touched. Neurons are
renumbered at load so the input neurons come first: the Poisson drive is then one slice add per step, and the
Poisson draws for a whole tick come from one RNG call. Everything is float32 and deterministic for a given seed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy import sparse

from .interfaces import (BRAIN_FILE, BRAIN_TICK_MS, CHANNELS, INPUT_GROUPS, N_AZIMUTH_BINS, N_CHANNELS,
                         OUTPUT_GROUPS)


@dataclass(frozen=True)
class LIFParams:
    """Shiu et al., Nature 2024. Times in ms, potentials in mV."""

    v_rest: float = -52.0
    v_reset: float = -52.0
    v_th: float = -45.0
    tau_m: float = 20.0
    tau_syn: float = 5.0
    t_ref: float = 2.2
    delay: float = 1.8
    w_syn: float = 0.275  # mV per synapse
    poisson_factor: float = 250.0  # an external input spike counts as this many synapses
    dt: float = 0.5

    @classmethod
    def from_meta(cls, meta: dict | None) -> LIFParams:
        """Defaults overridden by the brain file's meta["lif"] (only the fields above; membrane noise is not
        part of the model and must be 0)."""
        d = dict(meta or {})
        if float(d.pop("noise_mv", 0.0) or 0.0) != 0.0:
            raise ValueError("membrane noise is not supported (meta lif noise_mv must be 0)")
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    @property
    def delay_steps(self) -> int:
        return max(1, int(round(self.delay / self.dt)))

    @property
    def ref_steps(self) -> int:
        return max(1, int(round(self.t_ref / self.dt)))


def lc10a_bins(group: np.ndarray, stored: list | None) -> list[np.ndarray]:
    """One side's LC10a split into N_AZIMUTH_BINS bins (global indices, bin 0 frontal). Uses the brain file's bins
    when they partition the group, else ranks the group by index (the builder's own fallback)."""
    group = np.asarray(group, np.int64)
    if stored is not None and len(stored) == N_AZIMUTH_BINS:
        bins = [np.asarray(b, np.int64) for b in stored]
        flat = np.concatenate(bins)
        if flat.size == group.size and np.array_equal(np.sort(flat), np.sort(group)):
            return bins
    return [np.asarray(b, np.int64) for b in np.array_split(np.sort(group), N_AZIMUTH_BINS)]


@dataclass(frozen=True, eq=False)  # holds arrays: compare by identity
class Wiring:
    """Everything a brain shares with its copies: the graph (renumbered: input neurons first) and the
    channel / group maps. Read-only."""

    path: Path
    name: str
    p: LIFParams
    n: int
    n_connections: int
    perm: np.ndarray  # sim index -> file index
    inv: np.ndarray  # file index -> sim index
    indptr: np.ndarray  # CSR over presynaptic sim indices
    ends: np.ndarray  # indptr[1:]
    indices: np.ndarray
    weights_mv: np.ndarray  # w_syn * signed synapse count (float32)
    offsets: np.ndarray  # 0..n_connections-1, scratch for gathering the synapses of several rows
    n_in: int  # sim neurons 0..n_in-1 are the input neurons
    in_chan: np.ndarray  # encoder channel of each input neuron
    out_idx: np.ndarray  # OUTPUT_GROUPS neurons concatenated (sim indices)
    out_starts: np.ndarray
    out_sizes: np.ndarray
    groups: dict[str, np.ndarray]  # name -> file indices (as stored)
    group_sizes: dict[str, int]
    meta: dict


_WIRING: dict[tuple, Wiring] = {}


def load_wiring(path: str | Path = BRAIN_FILE, lif: LIFParams | None = None) -> Wiring:
    """The brain file's wiring, loaded once per process (cached by path, file mtime and LIF parameters)."""
    p = Path(path).resolve()
    key = (str(p), p.stat().st_mtime_ns, lif)
    w = _WIRING.get(key)
    if w is None:
        w = _WIRING[key] = _build_wiring(p, lif)
    return w


def _build_wiring(path: Path, lif: LIFParams | None) -> Wiring:
    with np.load(path, allow_pickle=False) as z:
        indptr, indices, data = z["indptr"], z["indices"], z["data"]
        n = int(z["shape"][0])
        if tuple(int(v) for v in z["shape"]) != (n, n) or indptr.size != n + 1:
            raise ValueError(f"{path.name}: expected a square presynaptic-major graph, got shape {z['shape']}")
        groups = {k: np.asarray(v, np.int64) for k, v in json.loads(str(z["groups"])).items()}
        meta = json.loads(str(z["meta"]))
    params = lif or LIFParams.from_meta(meta.get("lif"))
    sizes = {g: int(groups.get(g, np.zeros(0)).size) for g in INPUT_GROUPS + OUTPUT_GROUPS}
    empty = [g for g in ("LC10a_L", "LC10a_R") + OUTPUT_GROUPS if sizes[g] == 0]
    if empty:
        raise ValueError(f"{path.name}: interface groups with no neurons: {', '.join(empty)}")

    # input neurons in channel order: LC10a bins L0..L7, R0..R7, then LC9, LC11, AROUSAL per side
    parts: list[np.ndarray] = []
    for side in "LR":
        g = f"LC10a_{side}"
        parts += lc10a_bins(groups[g], (meta.get("azimuth_bins") or {}).get(g))
    parts += [groups.get(g, np.zeros(0, np.int64)) for g in ("LC9_L", "LC9_R", "LC11_L", "LC11_R",
                                                               "AROUSAL_L", "AROUSAL_R")]
    assert len(parts) == N_CHANNELS == len(CHANNELS)
    in_file = np.concatenate(parts)
    if np.unique(in_file).size != in_file.size:
        raise ValueError(f"{path.name}: a neuron belongs to two input channels")
    in_chan = np.concatenate([np.full(b.size, c, np.int64) for c, b in enumerate(parts)])
    rest = np.setdiff1d(np.arange(n), in_file)
    perm = np.concatenate([in_file, rest]).astype(np.int64)
    inv = np.empty(n, np.int64)
    inv[perm] = np.arange(n)

    m = sparse.csr_matrix((data.astype(np.float32), indices, indptr), shape=(n, n))  # (pre, post)
    m.sum_duplicates()
    m = m[perm][:, perm].tocsr()
    m.sort_indices()
    m.eliminate_zeros()
    wmv = (m.data * np.float32(params.w_syn)).astype(np.float32)

    out = [inv[groups[g]] for g in OUTPUT_GROUPS]
    out_sizes = np.array([o.size for o in out], np.int64)
    return Wiring(
        path=path, name=str(meta.get("name", path.stem)), p=params, n=n, n_connections=int(m.nnz), perm=perm, inv=inv,
        indptr=m.indptr.astype(np.int64), ends=m.indptr[1:].astype(np.int64), indices=m.indices.astype(np.int64),
        weights_mv=wmv, offsets=np.arange(m.nnz, dtype=np.int64),
        n_in=int(in_file.size), in_chan=in_chan, out_idx=np.concatenate(out),
        out_starts=np.concatenate([[0], np.cumsum(out_sizes)[:-1]]).astype(np.int64), out_sizes=out_sizes,
        groups=groups, group_sizes=sizes, meta=meta)


class PursuitBrain:
    """One simulated fly. tick() advances it by `ms` of neuron time; copy(seed) gives another fly with the same
    wiring and its own state and Poisson noise."""

    def __init__(self, path: str | Path = BRAIN_FILE, seed: int = 0, lif: LIFParams | None = None,
                 wiring: Wiring | None = None):
        self.w = wiring or load_wiring(path, lif)
        self.brain_s = 0.0  # wall time spent in tick() since reset
        self.ticks = 0
        self.reset(seed)

    def copy(self, seed: int | None = None) -> PursuitBrain:
        return PursuitBrain(seed=self.seed if seed is None else seed, wiring=self.w)

    @property
    def n(self) -> int:
        return self.w.n

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return self.w.groups

    @property
    def last_counts(self) -> np.ndarray:
        """Spikes per neuron during the last tick, in brain-file order."""
        return self._counts[self.w.inv]

    def reset(self, seed: int | None = None) -> None:
        """Back to rest (v = v_rest, no conductance, nothing in flight). `seed` restarts the Poisson noise."""
        w, p = self.w, self.w.p
        if seed is not None or not hasattr(self, "seed"):
            self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.v = np.full(w.n, p.v_rest, np.float32)
        self.g = np.zeros(w.n, np.float32)
        self._buf = np.zeros((p.delay_steps, w.n), np.float32)  # conductance in flight, slot = arrival step % delay
        self._tmp = np.zeros(w.n, np.float32)
        self._ring = [np.zeros(0, np.int64)] * p.ref_steps  # spikes of the last ref_steps steps (refractory now)
        self._step = 0
        self._at_rest = True  # exactly at rest with nothing in flight (a warmup with no input is free)
        self._counts = np.zeros(w.n, np.int32)
        self.last_rates = np.zeros(len(OUTPUT_GROUPS))
        self.brain_s = 0.0
        self.ticks = 0

    def input_rates(self, rates: np.ndarray) -> np.ndarray:
        """22 channel rates (Hz) -> the Poisson rate of every input neuron (sim order)."""
        r = np.asarray(rates, np.float64)
        if r.shape != (N_CHANNELS,):
            raise ValueError(f"expected {N_CHANNELS} channel rates, got shape {r.shape}")
        if not np.all(np.isfinite(r)):
            raise ValueError("channel rates must be finite")
        return np.maximum(r, 0.0)[self.w.in_chan]

    def tick(self, rates: np.ndarray, ms: float = BRAIN_TICK_MS) -> np.ndarray:
        """Run `ms` of neuron time with these channel rates (Hz). Returns the mean rate (Hz) of each OUTPUT_GROUPS
        entry over the tick."""
        t0 = time.perf_counter()
        w, p = self.w, self.w.p
        steps = max(1, int(round(ms / p.dt)))
        prob = (self.input_rates(rates) * (p.dt / 1000.0)).astype(np.float32)
        live = np.flatnonzero(prob > 0)
        if live.size == 0 and self._at_rest:  # nothing to integrate: exactly what the steps would compute
            counts = np.zeros(w.n, np.int32)
        else:
            ext = None
            if live.size:
                kick = np.float32(p.w_syn * p.poisson_factor)
                ext = np.zeros((steps, w.n_in), np.float32)
                ext[:, live] = (self.rng.random((steps, live.size), dtype=np.float32) < prob[live]) * kick
            counts = self._run(steps, ext)
            self._at_rest = False
        self._counts = counts
        c = counts[w.out_idx].astype(np.float64)
        self.last_rates = np.add.reduceat(c, w.out_starts) / w.out_sizes * (1000.0 / (steps * p.dt))
        self.brain_s += time.perf_counter() - t0
        self.ticks += 1
        return self.last_rates.copy()

    def _run(self, steps: int, ext: np.ndarray | None) -> np.ndarray:
        w, p = self.w, self.w.p
        v, g, buf, tmp, ring = self.v, self.g, self._buf, self._tmp, self._ring
        n, n_in, indptr, ends, indices, wmv, offsets = w.n, w.n_in, w.indptr, w.ends, w.indices, w.weights_mv, w.offsets
        n_delay, n_ref = buf.shape[0], len(ring)
        v_rest, v_reset, v_th = np.float32(p.v_rest), np.float32(p.v_reset), np.float32(p.v_th)
        k_v, decay = np.float32(p.dt / p.tau_m), np.float32(np.exp(-p.dt / p.tau_syn))
        counts = np.zeros(n, np.int32)
        above = np.zeros(n, bool)
        step = self._step
        for k in range(steps):
            arriving = buf[step % n_delay]
            g += arriving
            arriving.fill(0.0)
            if ext is not None:
                g[:n_in] += ext[k]
            ref = np.concatenate(ring)  # spikes of the last n_ref steps: refractory now
            if ref.size:
                g_ref = g[ref]
            np.subtract(v_rest, v, out=tmp)
            tmp += g
            tmp *= k_v
            v += tmp
            g *= decay
            if ref.size:  # refractory: v held at reset, g neither integrates nor decays
                v[ref] = v_reset
                g[ref] = g_ref
            np.greater(v, v_th, out=above)
            spk = np.flatnonzero(above)
            ring[step % n_ref] = spk
            step += 1
            if not spk.size:
                continue
            v[spk] = v_reset
            g[spk] = 0.0
            counts[spk] += 1
            # into this slot's buffer, which is read again `delay` steps from now
            if spk.size == 1:
                a, b = indptr[spk[0]], ends[spk[0]]
                arriving[indices[a:b]] += wmv[a:b]  # one row: its targets are distinct
            else:
                starts = indptr[spk]
                lens = ends[spk] - starts
                csum = np.cumsum(lens)
                total = int(csum[-1])
                if total:
                    sel = np.repeat(starts - (csum - lens), lens) + offsets[:total]  # the synapses of all spk rows
                    arriving += np.bincount(indices[sel], weights=wmv[sel], minlength=n).astype(np.float32)
        self._step = step
        return counts
