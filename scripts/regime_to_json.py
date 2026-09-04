#!/usr/bin/env python
"""Convert the recorded regime-test stdout into the JSON the figure reads.

The raw run output is kept beside the JSON as provenance (results/raw/), so the
figure's numbers trace back to a recorded measurement rather than being retyped.
This parses; it does not compute anything.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Rows are one nprobe with three variant columns:
#   "      2 0.6487  2,298   24.67(5.55) 0.6493  4,343  24.66(6.37) 0.6507  8,784  9.41(1.60)"
CELL = re.compile(r"([\d.]+)\s+([\d,]+)\s+([\d.]+)\(([\d.]+)\)")
EF_LIST, PER_CLUSTER_K = 64, 32


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/raw/regime_run.txt")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/regime_summary.json")
    text = src.read_text(encoding="utf-8")

    n_over_nc = float(re.search(r"N/Nc = ([\d,]+)", text).group(1).replace(",", ""))

    # Column order comes from the header line, not from an assumption about it.
    header = next(ln for ln in text.splitlines() if "nprobe" in ln and "ivf_disk" in ln)
    labels = header.split()[1:]
    m_by_label = {}
    for lab in labels:
        m_by_label[lab] = 0 if lab.startswith("ivf") else None
    # "eco M=16" arrives as two tokens; rebuild the real column names.
    labels = [lab for lab in header.replace("eco M=", "M").split() if lab != "nprobe"]

    series: dict[str, dict[str, list[float]]] = {}
    for line in text.splitlines():
        cells = CELL.findall(line)
        if len(cells) != len(labels) or not line.strip()[0].isdigit():
            continue
        for lab, (recall, dist, ms, _iqr) in zip(labels, cells, strict=True):
            m_val = 0 if lab.startswith("ivf") else int(lab.lstrip("M"))
            # The paper's own condition: graphs can pay only when ef_L*M' < N/Nc.
            ratio = (max(EF_LIST, PER_CLUSTER_K) * m_val) / n_over_nc if m_val else 0.0
            name = "ivf_disk (no graphs)" if m_val == 0 else f"per-cluster graphs M={m_val}"
            s_ = series.setdefault(name, {"ratio": [], "ms": [], "recall": [], "dist": []})
            s_["ratio"].append(ratio)
            s_["ms"].append(float(ms))
            s_["recall"].append(float(recall))
            s_["dist"].append(float(dist.replace(",", "")))

    payload = {
        "source": str(src),
        "n_over_nc": n_over_nc,
        "condition": "ef_L * M' < N/Nc  (MobileRAG search-time models)",
        "series": series,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} from {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
