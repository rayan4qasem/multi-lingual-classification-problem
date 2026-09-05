#!/usr/bin/env bash
# Wait for the full-prompt run to finish, then run the compact one and
# report both. Written as a file so the driver survives a shell restart.
set -u
PY=".venv/Scripts/python.exe"
count() { PYTHONIOENCODING=utf-8 $PY -c "
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
print(len(json.loads(p.read_text(encoding='utf-8'))) if p.exists() else 0)" "$1" 2>/dev/null || echo 0; }

while [ "$(count runs/120b_full.json)" -lt 86 ]; do sleep 20; done
echo "=== full prompt run complete ==="

PYTHONIOENCODING=utf-8 $PY scripts/score_llm.py --model openai/gpt-oss-120b \
  --detail compact --reasoning-effort low --rpm 3 --tag 120b_compact 2>&1 | tail -5
echo "=== compact run complete ==="

PYTHONIOENCODING=utf-8 $PY scripts/compare_runs.py
