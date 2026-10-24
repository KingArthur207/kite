"""Submission disqualifier audit (spec section 9, Sun 8:00 AM). Stdlib only.

    python3 scripts/audit.py            # exits 1 if any check fails
    python3 scripts/audit.py --selftest # checks the secret patterns themselves

Checks: no secrets in tracked files or git history, .env not tracked, .env.example
has no secret values, repo is public, README has a setup section, no tracked file
over 50 MB, no model weights tracked. Devpost items (teammates, tracks) stay manual.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "KingArthur207/kite"
MAX_BYTES = 50 * 1024 * 1024
HISTORY_CAP_BYTES = 200 * 1024 * 1024   # stop reading `git log -p` past this
WEIGHT_EXTS = {".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".tflite",
               ".mlmodel", ".mlpackage", ".gguf", ".engine", ".pb", ".weights"}

SECRET_PATTERNS = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Cartesia key": re.compile(r"sk_car_[A-Za-z0-9_\-]{16,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"),
    # NAME=value in .env style, NAME being *KEY / *TOKEN / *SECRET / *PASSWORD
    "env-style secret": re.compile(
        r"^[+\s]*(?:export\s+)?[A-Z0-9_]*(?:API_KEY|_KEY|TOKEN|SECRET|PASSWORD)\s*=\s*['\"]?(?!os\.|getenv|environ|process\.env|None\b|self\.|cfg\.|config\.)[A-Za-z0-9_\-./+]{12,}",
        re.M),
    # api_key = "...." in code
    "quoted api_key/secret": re.compile(
        r"(?i)(?:api[_-]?key|secret|token|password)['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]"),
}

results: list[tuple[str, str, str]] = []   # (status, check, detail)


def record(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def redact(s: str) -> str:
    s = s.strip()
    return s[:10] + "..." if len(s) > 10 else s


def scan(text: str) -> list[tuple[str, str]]:
    return [(name, redact(m.group(0))) for name, pat in SECRET_PATTERNS.items() for m in pat.finditer(text)]


def check_tracked_secrets(files: list[str]) -> None:
    hits = []
    for f in files:
        p = ROOT / f
        if not p.is_file() or p.stat().st_size > MAX_BYTES:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        hits += [f"{f}: {name} ({snip})" for name, snip in scan(text)]
    record("FAIL" if hits else "PASS", "no secrets in tracked files", "; ".join(hits[:10]))


def check_history_secrets() -> None:
    proc = subprocess.Popen(["git", "log", "-p", "--all", "--no-color", "--no-ext-diff"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="ignore")
    hits, commit, read, capped = [], "?", 0, False
    assert proc.stdout is not None
    for line in proc.stdout:
        read += len(line)
        if read > HISTORY_CAP_BYTES:
            capped = True
            break
        if line.startswith("commit "):
            commit = line.split()[1][:8]
        elif line.startswith("+") and not line.startswith("+++"):
            hits += [f"{commit}: {name} ({snip})" for name, snip in scan(line)]
    proc.kill()
    proc.wait()
    if hits:
        record("FAIL", "no secrets in git history",
               "; ".join(hits[:10]) + " (rotate the key, then rewrite history)")
    elif capped:
        record("WARN", "no secrets in git history", f"only the first {HISTORY_CAP_BYTES >> 20} MB scanned")
    else:
        record("PASS", "no secrets in git history")


def check_env(files: list[str]) -> None:
    tracked = [f for f in files if re.search(r"(^|/)\.env($|\.)", f) and not f.endswith(".env.example")]
    record("FAIL" if tracked else "PASS", ".env not tracked", ", ".join(tracked))

    example = ROOT / ".env.example"
    if not example.is_file():
        record("WARN", ".env.example has no secret values", "no .env.example")
        return
    filled = []
    for line in example.read_text().splitlines():
        m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
        if m and re.search(r"KEY|TOKEN|SECRET|PASSWORD", m.group(1)) and m.group(2).strip().strip("'\""):
            filled.append(m.group(1))
    record("FAIL" if filled else "PASS", ".env.example has no secret values", ", ".join(filled))


def check_public() -> None:
    if shutil.which("gh") is None:
        record("WARN", "repo is public", "gh not installed; check github.com/" + REPO + " logged out")
        return
    r = subprocess.run(["gh", "api", f"repos/{REPO}", "--jq", ".private"], capture_output=True, text=True)
    out = r.stdout.strip()
    if r.returncode != 0:
        record("WARN", "repo is public", f"gh api failed: {r.stderr.strip()[:120]}")
    elif out == "false":
        record("PASS", "repo is public")
    else:
        record("WARN", "repo is public", f"{REPO} is private; the team decides when to make it public (judges need access to the code)")


def check_readme() -> None:
    p = ROOT / "README.md"
    if not p.is_file():
        record("FAIL", "README with a setup section", "README.md missing")
        return
    ok = re.search(r"(?im)^#+\s*.*(setup|set up|quick start|install|getting started|run)", p.read_text())
    record("PASS" if ok else "FAIL", "README with a setup section", "" if ok else "no setup heading found")


def check_sizes(files: list[str]) -> None:
    big, weights = [], []
    for f in files:
        p = ROOT / f
        if p.is_file() and p.stat().st_size > MAX_BYTES:
            big.append(f"{f} ({p.stat().st_size >> 20} MB)")
        if Path(f).suffix.lower() in WEIGHT_EXTS:
            weights.append(f)
    record("FAIL" if big else "PASS", "no tracked file over 50 MB", ", ".join(big))
    record("FAIL" if weights else "PASS", "no model weights tracked", ", ".join(weights[:10]))


def selftest() -> int:
    leaks = ["ANTHROPIC_API_KEY=sk-ant-api03-" + "a" * 24, 'k = "AIza' + "B" * 35 + '"',
             "CARTESIA_API_KEY=sk_car_" + "c" * 20, "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
             'api_key = "' + "d" * 24 + '"']
    clean = ["ANTHROPIC_API_KEY=", 'api_key=os.environ.get("X")',
             'voice_id: str = "47c38ca4-5f35-497b-b1a3-415245fb35e1"',
             'LIVE_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")', "TOKEN = None"]
    assert all(scan(s) for s in leaks), [s for s in leaks if not scan(s)]
    assert not any(scan(s) for s in clean), [s for s in clean if scan(s)]
    print("audit selftest ok")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    try:
        files = git("ls-files").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[audit] not a git checkout: {e}")
        return 2
    check_tracked_secrets(files)
    check_history_secrets()
    check_env(files)
    check_public()
    check_readme()
    check_sizes(files)

    print(f"Kite submission audit ({len(files)} tracked files)\n")
    for status, check, detail in results:
        print(f"  [{status}] {check}" + (f"\n         {detail}" if detail else ""))
    print("\n  [TODO] by hand on Devpost: all four teammates listed, tracks ticked, video link works")
    failed = sum(s == "FAIL" for s, _, _ in results)
    print(f"\n{'FAILED' if failed else 'PASSED'}: {failed} failing check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
