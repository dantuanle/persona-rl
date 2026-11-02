#!/usr/bin/env bash
# writeup.md -> writeup.html -> writeup.pdf (Chrome headless). Usage: bash render_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import markdown, re
md = open("writeup.md").read()
html = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.32;max-width:7.1in;margin:0.45in auto;color:#111}
h1{font-size:15pt;margin:0 0 4px} h2{font-size:11.5pt;margin:12px 0 4px} p{margin:5px 0} table{border-collapse:collapse;font-size:9pt;margin:6px 0}
td,th{border:1px solid #bbb;padding:2px 6px} img{max-width:100%} code{font-size:9pt} ul{margin:4px 0 4px 18px} li{margin:2px 0}"""
open("writeup.html","w").write(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>")
PY
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/writeup.pdf" "file://$PWD/writeup.html" >/dev/null 2>&1
.venv/bin/python -c "
import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-dan-mt-moral-alignment/bd4ebf6f-a7db-4195-a522-84e99c8db215/scratchpad/lib')
from pypdf import PdfReader; print('writeup.pdf pages:', len(PdfReader('writeup.pdf').pages))"
