#!/usr/bin/env bash
# usage: bash pods.sh ssh <W|O|B> '<cmd>'   | bash pods.sh push <W|O|B>   | bash pods.sh pull <W|O|B> <remote> <local>
# direct endpoints live in pods.direct (lines: ARM HOST PORT), written by the orchestrator from get-pod reads.
set -euo pipefail
cd "$(dirname "$0")"; source pods.env
arm=$2; line=$(grep "^$arm " pods.direct || true)
[ -n "$line" ] || { echo "no direct ssh endpoint for $arm in pods.direct"; exit 2; }
host=$(echo "$line" | awk '{print $2}'); port=$(echo "$line" | awk '{print $3}')
SSHOPTS="-i $SSH_KEY -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30"
case "$1" in
  ssh)  shift 2; ssh $SSHOPTS root@$host "$@" ;;
  push) COPYFILE_DISABLE=1 tar czf - --exclude .git --exclude .venv --exclude results --exclude __pycache__ --exclude data/built . | ssh $SSHOPTS root@$host 'mkdir -p /workspace/persona-rl && tar xzf - -C /workspace/persona-rl' ;;
  pushdata) COPYFILE_DISABLE=1 tar czf - -C data/built . | ssh $SSHOPTS root@$host 'mkdir -p /workspace/data && tar xzf - -C /workspace/data' ;;
  pull) ssh $SSHOPTS root@$host "tar czf - -C $(dirname $3) $(basename $3)" | (mkdir -p "$4" && tar xzf - -C "$4") ;;
  *) echo "unknown"; exit 1 ;;
esac
