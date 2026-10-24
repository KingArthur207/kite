#!/bin/sh
# Run ON the UNO Q (over adb shell or ssh) to join eduroam (WPA2-Enterprise: PEAP + MSCHAPv2).
#     sh join_eduroam.sh you@berkeley.edu 'your-eduroam-password' berkeley.edu
# The third argument is your school's domain, used to check the network's certificate. Your password is
# stored only in the board's NetworkManager config (root-only), never in this repo.
set -e
if [ $# -lt 3 ]; then echo "usage: sh join_eduroam.sh <identity> <password> <school domain, e.g. berkeley.edu>"; exit 2; fi
ID="$1"; PW="$2"; DOMAIN="$3"
IFACE=$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="wifi"{print $1; exit}')
sudo nmcli connection delete eduroam >/dev/null 2>&1 || true
sudo nmcli connection add type wifi con-name eduroam ifname "$IFACE" ssid eduroam \
  wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.phase2-auth mschapv2 \
  802-1x.identity "$ID" 802-1x.anonymous-identity "anonymous@$DOMAIN" 802-1x.password "$PW" \
  802-1x.ca-cert /etc/ssl/certs/ca-certificates.crt 802-1x.domain-suffix-match "$DOMAIN" \
  connection.autoconnect yes
sudo nmcli connection up eduroam
echo "joined eduroam on $IFACE. Check internet: curl -sI https://maincloud.spacetimedb.com | head -1"
