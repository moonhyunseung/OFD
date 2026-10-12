from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path'); a=ap.parse_args(); p=Path(a.path)
    if p.suffix=='.csv':
        rows=list(csv.DictReader(p.open()))
        print(f"rows={len(rows)} columns={list(rows[0]) if rows else []}")
        for r in rows[:12]: print(r)
    elif p.suffix=='.json': print(json.dumps(json.loads(p.read_text()), indent=2))
    else: print(p.read_text())
if __name__=='__main__': main()
