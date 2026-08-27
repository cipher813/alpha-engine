#!/bin/bash
# Install the boot-pull systemd service.
# Must run as root (sudo).
#
# Usage:
#   sudo bash /home/ec2-user/alpha-engine/infrastructure/install-boot-pull.sh

set -euo pipefail

SERVICE_FILE="/etc/systemd/system/boot-pull.service"
SCRIPT="/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh"
LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"
LAUNCHER_DST="/usr/local/sbin/boot-pull-launcher.sh"
LOG="/var/log/boot-pull.log"

# Ensure log file exists with correct ownership
touch "$LOG"
chown ec2-user:ec2-user "$LOG"

# Ensure script is executable
chmod +x "$SCRIPT"

# alpha-engine-config-I8734: install the launcher OUTSIDE the synced
# alpha-engine checkout. systemd's ExecStart must never point at a path
# inside the tree boot-pull.sh hard-resets (sync AND rollback) while it
# may still be executing — bash resumes a running script at a byte offset
# after each command, so a self-rewrite mid-run silently resumes in the
# replaced file. Re-copying on every install run keeps /usr/local/sbin in
# sync with whatever version of the launcher this checkout carries; the
# launcher itself only changes on a deliberate infra PR, never at boot.
install -m 0755 -o root -g root "$LAUNCHER_SRC" "$LAUNCHER_DST"

# Write systemd unit
cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=Pull latest Alpha Engine code on boot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ec2-user
ExecStart=/usr/local/sbin/boot-pull-launcher.sh
TimeoutStartSec=120
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable boot-pull.service

echo "boot-pull.service installed and enabled."
echo "  ExecStart -> $LAUNCHER_DST (outside the synced checkout, config-I8734)"
echo "  Launcher snapshots + execs $SCRIPT on every boot."
echo "  Pulls all repos on every boot before cron jobs fire."
echo "  Logs: tail -f $LOG"
echo "  Test: sudo systemctl start boot-pull && cat $LOG"
