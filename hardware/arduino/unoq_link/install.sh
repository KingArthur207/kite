#!/bin/sh
# Kite link installer for the Arduino UNO Q. Paste this one line into the board's terminal
# (App Lab's built-in terminal, or ssh):
#
#     curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh
#
# It downloads the board files over HTTPS, checks each one's SHA-256 against the list below, installs the
# kite-link service (starts at every power-on), makes the board's token, offers to join eduroam, and prints
# what to run on the computer. Safe to run again: it updates in place and keeps the token.
#
#     ... | sh -s -- --eduroam       only join eduroam (asks for username, password, school domain)
#     ... | sh -s -- --no-eduroam    never ask
#
# Run from a folder that already holds the files (a repo checkout), it uses those instead of downloading,
# still checking every hash.
set -eu

BASE_URL="${KITE_BASE_URL:-https://trykite.pages.dev/unoq}"
PORT=8765

# --- hashes: written by publish.py, do not edit by hand ---
HASHES='
b7ccd91c012c4d18483a5a8882ced8e29d58a0be7e6dc6e731cc62fef66aba19  board.py
52e5830daf6b605c0fe0f8aa10ae6f056e711eff7bdaf195f3ec5b5476533c1b  kite_common.py
ae2a6d22541f01b6ba28253078e2b8d5545708a2531899a7e1f8f45ad303a659  kite-link.service
'
# --- end hashes ---

say() { printf '%s\n' "$*"; }
die() { printf 'kite: %s\n' "$*" >&2; exit 1; }

if [ "$(id -u)" = 0 ]; then
  SUDO=""
  KITE_USER="${KITE_USER:-${SUDO_USER:-arduino}}"
else
  SUDO="sudo"
  KITE_USER="${KITE_USER:-$(id -un)}"
fi
HOME_DIR="$(getent passwd "$KITE_USER" 2>/dev/null | cut -d: -f6 || true)"
[ -n "$HOME_DIR" ] || HOME_DIR="/home/$KITE_USER"
INSTALL_DIR="${KITE_INSTALL_DIR:-$HOME_DIR/kite-link}"
ETC_DIR="${KITE_ETC_DIR:-/etc/kite}"
UNIT_DIR="${KITE_UNIT_DIR:-/etc/systemd/system}"

# where this script lives, when it was run as a file (not piped from curl)
SRC_DIR=""
case "$0" in
  */install.sh|install.sh) SRC_DIR="$(cd "$(dirname "$0")" && pwd)" ;;
esac

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}

fetch() {  # fetch <url> <file>: curl, or wget where curl is missing
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 "$1" -o "$2"
  else wget -q -O "$2" "$1"
  fi
}

have_tty() { (: </dev/tty) 2>/dev/null; }

ask() {  # ask <prompt> [hidden]: one line from the terminal, even when this script is piped from curl
  printf '%s' "$1" >/dev/tty
  if [ "${2:-}" = hidden ]; then
    stty -echo </dev/tty 2>/dev/null || true
    IFS= read -r REPLY </dev/tty || REPLY=""
    stty echo </dev/tty 2>/dev/null || true
    printf '\n' >/dev/tty
  else
    IFS= read -r REPLY </dev/tty || REPLY=""
  fi
}

join_eduroam() {
  if ! have_tty; then
    die "joining eduroam asks for your password, so it needs a terminal. Run it in App Lab's terminal or ssh -t."
  fi
  ask "eduroam username (e.g. you@berkeley.edu): "; ID="$REPLY"
  [ -n "$ID" ] || die "no username given; nothing changed"
  ask "eduroam password (hidden): " hidden; PW="$REPLY"
  [ -n "$PW" ] || die "no password given; nothing changed"
  ask "school domain [berkeley.edu]: "; DOMAIN="${REPLY:-berkeley.edu}"
  IFACE="$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="wifi"{print $1; exit}')"
  [ -n "$IFACE" ] || die "no Wi-Fi interface found (nmcli device)"
  say "If you're connected to the board over Wi-Fi, this session may drop while it switches. That's fine."
  $SUDO nmcli connection delete eduroam >/dev/null 2>&1 || true
  $SUDO nmcli connection add type wifi con-name eduroam ifname "$IFACE" ssid eduroam \
    wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.phase2-auth mschapv2 \
    802-1x.identity "$ID" 802-1x.anonymous-identity "anonymous@$DOMAIN" 802-1x.password "$PW" \
    802-1x.ca-cert /etc/ssl/certs/ca-certificates.crt 802-1x.domain-suffix-match "$DOMAIN" \
    connection.autoconnect yes >/dev/null
  if $SUDO nmcli connection up eduroam >/dev/null 2>&1; then
    say "joined eduroam on $IFACE."
  else
    say "saved eduroam, but couldn't connect right now (out of range, or a wrong password). It joins by itself"
    say "when in range; to retype: rerun with --eduroam."
  fi
}

MODE=install
for arg in "$@"; do
  case "$arg" in
    --eduroam) MODE=eduroam ;;
    --no-eduroam) MODE=install-only ;;
    *) die "unknown option $arg (use --eduroam or --no-eduroam)" ;;
  esac
done
if [ "$MODE" = eduroam ]; then join_eduroam; exit 0; fi

say "Kite link: installing for user $KITE_USER into $INSTALL_DIR"
command -v python3 >/dev/null 2>&1 || die "python3 is missing (it ships with the UNO Q's Debian)"

# 1. the files, each checked against its SHA-256
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILES="$(printf '%s\n' "$HASHES" | awk 'NF==2{print $2}')"
for f in $FILES; do
  want="$(printf '%s\n' "$HASHES" | awk -v f="$f" '$2==f{print $1}')"
  if [ -n "$SRC_DIR" ] && [ -f "$SRC_DIR/$f" ]; then
    cp "$SRC_DIR/$f" "$TMP/$f"
  else
    fetch "$BASE_URL/$f" "$TMP/$f" || die "couldn't download $BASE_URL/$f (is the board online?)"
  fi
  got="$(sha256 "$TMP/$f")"
  [ "$got" = "$want" ] || die "checksum mismatch for $f: refusing to install (corrupted or tampered download)"
done
say "downloaded and verified: $(echo "$FILES" | tr '\n' ' ')"

# 2. OpenCV, only for a real USB camera (without it the board streams a labelled test pattern)
if ! python3 -c 'import cv2' >/dev/null 2>&1; then
  if [ -n "${KITE_SKIP_APT:-}" ]; then
    say "skipping OpenCV (KITE_SKIP_APT set): video will be the test pattern"
  else
    say "installing OpenCV for the USB camera (apt, a few minutes the first time)..."
    if ! { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-opencv >/dev/null; }; then
      say "OpenCV didn't install; video will be the test pattern until it does (sudo apt-get install python3-opencv)"
    fi
  fi
fi

# 3. install the files and the token (kept across reinstalls)
$SUDO mkdir -p "$INSTALL_DIR" "$ETC_DIR" "$UNIT_DIR"
for f in board.py kite_common.py; do $SUDO cp "$TMP/$f" "$INSTALL_DIR/$f"; done
$SUDO chown -R "$KITE_USER" "$INSTALL_DIR"
if [ ! -s "$ETC_DIR/token" ]; then
  python3 -c 'import secrets; print(secrets.token_hex(16))' > "$TMP/token"
  $SUDO cp "$TMP/token" "$ETC_DIR/token"
  say "made a new token"
fi
$SUDO chown "$KITE_USER" "$ETC_DIR/token"
$SUDO chmod 600 "$ETC_DIR/token"
sed -e "s#__USER__#$KITE_USER#g" -e "s#__DIR__#$INSTALL_DIR#g" "$TMP/kite-link.service" > "$TMP/unit"
$SUDO cp "$TMP/unit" "$UNIT_DIR/kite-link.service"

# 4. the service (the old unoq_wifi hello sender sends the same heartbeat, so it's retired)
if systemctl is-enabled kite-hello >/dev/null 2>&1; then
  $SUDO systemctl disable --now kite-hello >/dev/null 2>&1 || true
  say "turned off the old kite-hello service (kite-link sends the hello now)"
fi
$SUDO systemctl daemon-reload
$SUDO systemctl enable kite-link >/dev/null 2>&1
$SUDO systemctl restart kite-link
up=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if fetch "http://127.0.0.1:$PORT/health" /dev/null 2>/dev/null; then up=1; break; fi
  sleep 1
done
if [ -n "$up" ]; then say "kite-link is running on port $PORT"; else
  say "kite-link didn't answer yet; its log: journalctl -u kite-link -e"; fi

# 5. what to do on the computer (printed before eduroam, which can drop this session)
board_ips() { hostname -I 2>/dev/null || true; }
say ""
say "board IP: $(board_ips)"
say "token:    $($SUDO cat "$ETC_DIR/token")"
say "on the computer, once:  python kite_link.py token <the token above>"
say ""

# 6. eduroam, if wanted (last: switching Wi-Fi can drop this session)
if [ "$MODE" = install ]; then
  if have_tty; then
    ask "Join eduroam now? [y/N] "
    case "$REPLY" in
      y|Y|yes|YES) join_eduroam; say "board IP now: $(board_ips)" ;;
      *) say "not joining eduroam" ;;
    esac
  else
    say "no terminal to ask about eduroam; to join later (in App Lab's terminal or ssh -t):"
    say "    curl -fsSL $BASE_URL/install.sh | sh -s -- --eduroam"
  fi
fi
say ""
say "done: on the computer run  python kite_link.py doctor"
