#!/usr/bin/env bash
# install.sh -- set up the onboard runner on the Raspberry Pi 4B.
#
#   cd <repo checkout on the Pi>
#   sudo bash onboard/deploy/install.sh                 # check UART, install deps, install + start the service
#   sudo bash onboard/deploy/install.sh --configure-uart  # also enable the UART / disable the serial console
#
# It never edits the kernel command line's network settings (a stale
# "ip=192.168.50.2::192.168.50.1:..." parameter was removed earlier on purpose:
# do not add it back). With --configure-uart it only runs raspi-config's
# serial options (login shell over serial OFF, serial hardware ON); a reboot
# is then needed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
CONFIGURE_UART=0
[[ "${1:-}" == "--configure-uart" ]] && CONFIGURE_UART=1

if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi
echo "repo: $REPO   user: $RUN_USER"

# ---- UART check: /dev/serial0 must be the header UART, without a serial console ----
CFG=/boot/firmware/config.txt; [[ -f $CFG ]] || CFG=/boot/config.txt
CMD=/boot/firmware/cmdline.txt; [[ -f $CMD ]] || CMD=/boot/cmdline.txt
uart_ok=1
grep -Eq '^\s*enable_uart=1' "$CFG" || { echo "UART: enable_uart=1 missing in $CFG"; uart_ok=0; }
grep -Eq 'console=(serial0|ttyAMA0|ttyS0)' "$CMD" && { echo "UART: serial console still enabled in $CMD"; uart_ok=0; }
[[ -e /dev/serial0 ]] || { echo "UART: /dev/serial0 does not exist"; uart_ok=0; }
if [[ $uart_ok -eq 0 ]]; then
  if [[ $CONFIGURE_UART -eq 1 ]]; then
    raspi-config nonint do_serial_cons 1     # login shell over serial: OFF
    raspi-config nonint do_serial_hw 0       # serial hardware: ON
    echo "UART configured. REBOOT, then run this script again."
    exit 0
  fi
  echo "UART not ready. Re-run with --configure-uart (then reboot), or set it in raspi-config:"
  echo "  Interface Options -> Serial Port -> login shell: No, hardware: Yes"
  exit 1
fi
echo "UART: ok ($(readlink -f /dev/serial0))"
groups "$RUN_USER" | grep -q dialout || { usermod -aG dialout "$RUN_USER"; echo "added $RUN_USER to dialout"; }

# ---- dependencies: numpy + pyserial from apt (no pip into the system Python) ----
apt-get install -y python3-numpy python3-serial

# ---- self-test: imports, model load, one controller step on a synthetic observation ----
sudo -u "$RUN_USER" python3 - <<EOF
import sys; sys.path.insert(0, "$REPO")
from onboard.runner import load_config
from onboard import controllers as C
import numpy as np
cfg = load_config("$REPO/onboard/config.json")
a = C.build(cfg); a.reset(0)
act = a.act(np.zeros(12, dtype=np.float32))
print("self-test ok: controller", cfg["controller"], "first action", act)
EOF

# ---- systemd service ----
sed -e "s|@USER@|$RUN_USER|g" -e "s|@REPO@|$REPO|g" "$REPO/onboard/deploy/rlglider-onboard.service" \
  > /etc/systemd/system/rlglider-onboard.service
mkdir -p "$REPO/logs" && chown "$RUN_USER" "$REPO/logs"
systemctl daemon-reload
systemctl enable --now rlglider-onboard.service
systemctl --no-pager status rlglider-onboard.service | head -n 12
echo "logs: journalctl -u rlglider-onboard -f      flight logs: $REPO/logs/*.rlglog"
