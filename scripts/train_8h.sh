#!/usr/bin/env bash
# Continue the 10M baseline in a separate run, surviving terminal disconnects.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run=runs/jit-10m-smooth-1m-8h
if [[ -e "$run" ]]; then
    echo "Output already exists: $run. Resume its checkpoint explicitly instead." >&2
    exit 1
fi
systemd-run --user --unit=mandelbrot-jit-10m-8h --collect \
    --working-directory="$repo" \
    --property=StandardOutput=journal --property=StandardError=journal \
    "$repo/.venv/bin/python" -u train.py \
    --config configs/zoom4x-smooth-1m-jit-10m-8h.json \
    --resume runs/jit-10m-smooth-1m-patch32-pilot/checkpoint.pt \
    --max-seconds 28800
echo 'Follow progress: journalctl --user -u mandelbrot-jit-10m-8h -f'
