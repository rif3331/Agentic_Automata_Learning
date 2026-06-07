"""
Responsible for:

- Generating HTML comparison reports for LLM-submitted candidate DFAs versus target DFAs
- Rendering side-by-side DFA equivalence visualizations with witness/counterexample information
- Managing evaluation artifact directories and saving timestamped comparison reports
- Loading embedded HTML content or file-based DFA visualizations for reporting
- Supporting visualization of equivalence-check results during LLM interaction
"""
import html
import os
from datetime import datetime
from output_paths import get_artifact_dir

def ensure_eval_dir() -> str:
    return str(get_artifact_dir("evaluations"))


def read_if_path(s: str) -> str:
    if isinstance(s, str) and os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            return f.read()
    return s


def write_llm_comparison_html(
    *,
    eq: bool,
    witness_word: str,
    left_html: str,
    right_html: str,
    call_count: int,
) -> str:
    left_srcdoc = html.escape(left_html, quote=True)
    right_srcdoc = html.escape(right_html, quote=True)

    status_text = "EQUIVALENT" if eq else "NOT EQUIVALENT"
    witness_line = "" if eq else f"<div>Witness word: <b>{html.escape(witness_word)}</b></div>"

    eval_dir = ensure_eval_dir()
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"dfa_compare_{now}_call_{call_count}.html"
    out_path = os.path.join(eval_dir, filename)

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DFA Comparison</title>
<style>
html, body {{ height: 100%; margin: 0; font-family: Arial, sans-serif; }}
.container {{ display: flex; height: 100vh; }}
.side {{ flex: 1; }}
.center {{ width: 380px; padding: 16px; box-sizing: border-box; border-left: 1px solid #ddd; border-right: 1px solid #ddd; }}
iframe {{ width: 100%; height: 100%; border: 0; }}
h2 {{ margin: 0 0 12px 0; }}
.status {{ font-weight: 700; margin: 8px 0 12px 0; }}
.small {{ color: #555; margin-top: 12px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <div class="side">
    <iframe srcdoc="{left_srcdoc}"></iframe>
  </div>

  <div class="center">
    <h2>DFA Equivalence Check</h2>
    <div class="status">{status_text}</div>
    {witness_line}
    <div class="small">Left: target DFA</div>
    <div class="small">Right: candidate DFA</div>
  </div>

  <div class="side">
    <iframe srcdoc="{right_srcdoc}"></iframe>
  </div>
</div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    return out_path