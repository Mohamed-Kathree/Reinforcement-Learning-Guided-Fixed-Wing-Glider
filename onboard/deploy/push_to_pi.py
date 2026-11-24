"""
push_to_pi.py -- copy onboard/ to the Raspberry Pi over SSH and restart the flight program.
Run from any PC with this repo (Windows, macOS, Linux):

    pip install paramiko
    python onboard/deploy/push_to_pi.py                       # glider.local, user pi
    python onboard/deploy/push_to_pi.py --host 192.168.1.50   # if the name does not resolve
    python onboard/deploy/push_to_pi.py --no-restart          # copy only

- The password is asked for at the prompt and never stored.
- Only onboard/ is copied, into ~/rlglider/onboard on the Pi (the layout install.sh expects).
  Text files get Linux line endings. __pycache__ is skipped.
- Afterwards the service rlglider-onboard is restarted (sudo, with the same password), and
  the last journal lines are shown.
- The first time on a new Pi, run onboard/deploy/install.sh on the Pi instead (it installs
  numpy and the service). On the glider's Pi this was done on 20 Nov 2026.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT = (".py", ".sh", ".json", ".service", ".md", ".txt", ".yaml")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="glider.local")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--no-restart", action="store_true")
    a = ap.parse_args(argv)
    try:
        import paramiko
    except ImportError:
        print("needs paramiko: pip install paramiko")
        return 1
    pw = getpass.getpass(f"password for {a.user}@{a.host}: ")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(a.host, username=a.user, password=pw, timeout=10)

    def run(cmd: str) -> str:
        _, out, err = c.exec_command(cmd, timeout=120)
        return out.read().decode(errors="replace") + err.read().decode(errors="replace")

    sftp = c.open_sftp()
    dst_root = f"{sftp.normalize('.')}/rlglider/onboard"
    src = os.path.join(ROOT, "onboard")
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, src).replace("\\", "/")
        d = dst_root if rel == "." else f"{dst_root}/{rel}"
        run(f"mkdir -p '{d}'")
        for f in filenames:
            if f.endswith(".pyc"):
                continue
            data = open(os.path.join(dirpath, f), "rb").read()
            if f.endswith(TEXT):
                data = data.replace(b"\r\n", b"\n")
            with sftp.open(f"{d}/{f}", "wb") as fh:
                fh.write(data)
            n += 1
    sftp.close()
    print(f"copied {n} files to {a.host}:{dst_root}")
    if not a.no_restart:
        chan = c.get_transport().open_session()
        chan.exec_command("sudo -S systemctl restart rlglider-onboard")
        chan.sendall((pw + "\n").encode())
        chan.recv_exit_status()
        print(run("sleep 3; systemctl is-active rlglider-onboard; journalctl -u rlglider-onboard --no-pager -n 4 -o cat"))
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
