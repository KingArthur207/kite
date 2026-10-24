#!/usr/bin/env bash
# Trained .ply -> a .splat the sim loads (Spark reads .ply too, but .splat is ~8x smaller: no SH).
#
#   scripts/splat/export.sh <in.ply> <out.splat|out.ply> [--min-opacity 0.05] [--max-scale 2.0]
#        [--keep 0.995] [--box xmin,ymin,zmin,xmax,ymax,zmax] [--max-splats 3000000]
#
# Prunes floaters: splats fainter than --min-opacity, splats with an axis bigger than --max-scale
# (splat units), and splats outside the central --keep fraction on any axis (sky blobs and the far
# ends of the walk). --box crops to a box in splat coordinates instead (read them with L in the sim,
# see sim/README.md "Splat"). Keeps the --max-splats most visible. A .ply output keeps SH, untouched.
# For the sim: scripts/splat/export.sh <scene>/out/splat_30000.ply sim/public/local/moss.splat
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,11p "$0"; exit 2; }
exec uv run -q --no-project --with numpy python -I - "$@" <<'PY'
import argparse, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--min-opacity", type=float, default=0.05)
ap.add_argument("--max-scale", type=float, default=2.0)
ap.add_argument("--keep", type=float, default=0.995)
ap.add_argument("--box", type=lambda s: [float(v) for v in s.split(",")])
ap.add_argument("--max-splats", type=int, default=3_000_000)
a = ap.parse_args(sys.argv[1:])

TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
with open(a.inp, "rb") as f:
    header, props, n = [], [], 0
    while (line := f.readline().decode("ascii", "replace").strip()) != "end_header":
        header.append(line)
        w = line.split()
        if w[:2] == ["element", "vertex"]: n = int(w[2])
        elif w[:1] == ["property"] and n: props.append((w[2], "<" + TYPES[w[1]]))
        elif w[:1] == ["format"] and w[1] != "binary_little_endian": sys.exit(f"{a.inp}: {w[1]} ply not supported")
    v = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
names = v.dtype.names
need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
miss = [k for k in need if k not in names]
if miss: sys.exit(f"{a.inp}: not a 3DGS ply, missing {miss}")

pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float32)))
scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32))
ok = (alpha >= a.min_opacity) & (scale.max(1) <= a.max_scale) & np.isfinite(pos).all(1)
if a.box:
    lo, hi = np.array(a.box[:3]), np.array(a.box[3:])
else:
    t = (1 - a.keep) / 2 * 100
    lo, hi = np.percentile(pos[ok], t, axis=0), np.percentile(pos[ok], 100 - t, axis=0)
ok &= ((pos >= lo) & (pos <= hi)).all(1)
idx = np.flatnonzero(ok)
weight = alpha[idx] * scale[idx].prod(1)        # most visible first (and the order .splat viewers expect)
idx = idx[np.argsort(-weight)][: a.max_splats]
print(f"{n} splats in, {len(idx)} out ({100 * len(idx) / n:.0f}%); box {np.round(lo, 2).tolist()} to {np.round(hi, 2).tolist()}")

if a.out.endswith(".ply"):
    with open(a.out, "wb") as f:
        hdr = "\n".join(l if not l.startswith("element vertex") else f"element vertex {len(idx)}" for l in header)
        f.write((hdr + "\nend_header\n").encode()); f.write(v[idx].tobytes())
else:   # antimatter15 .splat: 32 bytes a splat
    C0 = 0.28209479177387814
    rgb = np.clip((0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)[idx]) * 255, 0, 255)
    rgba = np.concatenate([rgb, alpha[idx, None] * 255], 1).astype(np.uint8)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)[idx].astype(np.float32)   # w, x, y, z
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rec = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
    rec["p"], rec["s"], rec["c"] = pos[idx], scale[idx], rgba
    rec["q"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    rec.tofile(a.out)
print(f"wrote {a.out}")
PY
