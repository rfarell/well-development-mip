"""Merge score_meta.py outputs (rows, refs, cert) into one file for summarize_meta.py. Usage: merge_scored.py out.json in1.json in2.json ..."""
import json
import sys

import bench_v3 as B

out = dict(rows=[], refs={}, cert={}, sources=sys.argv[2:])
seen = set()
for fn in sys.argv[2:]:
    d = json.load(open(fn))
    for r in d["rows"]:
        if r["file"] in seen:
            continue
        seen.add(r["file"]); out["rows"].append(r)
    out["refs"].update(d.get("refs", {})); out["cert"].update(d.get("cert", {}))
    out.setdefault("oa", d.get("oa"))
B.dump(out, sys.argv[1])
print(f"{sys.argv[1]}: {len(out['rows'])} rows from {len(sys.argv) - 2} files, {len(out['refs'])} references")
