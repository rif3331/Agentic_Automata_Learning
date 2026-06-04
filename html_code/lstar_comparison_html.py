"""
Responsible for:

- Generating HTML comparison reports for L* hypothesis vs target DFA
- Rendering source and learned DFAs side-by-side for equivalence analysis
- Displaying comparison results and counterexample information in report form
- Saving timestamped comparison artifacts for L* evaluation runs
- Supporting visualization of DFA equivalence-check outcomes
"""
from pathlib import Path
from datetime import datetime
from typing import Optional
from automata.fa.dfa import DFA
from html_code.draw_DFA_html import draw_DFA_html
from utils import (
    _load_html,
    _escape_html,
    _escape_attr_html,
)
from output_paths import get_artifact_dir



def write_lstar_comparison_html(
    source_dfa: DFA,
    hyp_dfa: DFA,
    ok: bool,
    counterexample: Optional[str],
) -> str:
    out_dir = get_artifact_dir("L_star_comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    out_path = out_dir / f"{ts}.html"

    src_html = _load_html(draw_DFA_html(source_dfa))
    hyp_html = _load_html(draw_DFA_html(hyp_dfa))

    src_doc = _escape_attr_html(src_html)
    hyp_doc = _escape_attr_html(hyp_html)

    status = "Equivalent" if ok else "Not equivalent"
    cex = "" if counterexample is None else counterexample

    page = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>L* Comparison {ts}</title>
  <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:0;padding:0}}
    .top{{padding:14px 16px;border-bottom:1px solid #ddd;background:#fafafa}}
    .row{{display:flex;gap:16px;padding:16px}}
    .panel{{flex:1;min-width:0;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
    .hdr{{padding:10px 12px;background:#f6f6f6;border-bottom:1px solid #ddd;font-weight:700}}
    .cnt{{padding:12px}}
    iframe{{width:100%;height:78vh;border:0}}
    .kv{{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline}}
    .k{{font-weight:700}}
    .pill{{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid #ddd;background:#fff}}
    .bad{{border-color:#f1b3b3}}
    .good{{border-color:#b7e3b7}}
    code{{background:#fff;border:1px solid #eee;border-radius:6px;padding:2px 6px}}
  </style>
</head>
<body>
  <div class="top">
    <div class="kv">
      <div class="k">Result:</div>
      <div class="pill {"good" if ok else "bad"}">{_escape_html(status)}</div>
      <div class="k">Counterexample:</div>
      <div><code>{_escape_html(cex)}</code></div>
      <div class="k">Time:</div>
      <div>{_escape_html(ts)}</div>
    </div>
  </div>

  <div class="row">
    <div class="panel">
      <div class="hdr">Source DFA</div>
      <div class="cnt"><iframe srcdoc="{src_doc}"></iframe></div>
    </div>
    <div class="panel">
      <div class="hdr">Hypothesis DFA</div>
      <div class="cnt"><iframe srcdoc="{hyp_doc}"></iframe></div>
    </div>
  </div>
</body>
</html>
""".strip()

    out_path.write_text(page, encoding="utf-8")
    return str(out_path)