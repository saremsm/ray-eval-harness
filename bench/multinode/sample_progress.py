"""Sample completed-task counts over wall clock during a run."""
from __future__ import annotations

import argparse
import json
import sys
import time


def scan_new_lines(path: str, offset: int, workers: int) -> tuple[int, int, int]:
    """Read complete lines starting at byte offset."""
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return offset, 0, 0
    with fh:
        fh.seek(offset)
        chunk = fh.read()
    if not chunk:
        return offset, 0, 0
    # Only consume up to the last newline; the remainder is incomplete.
    last_nl = chunk.rfind(b"\n")
    if last_nl < 0:
        return offset, 0, 0
    complete = chunk[: last_nl + 1]
    lines = complete.split(b"\n")[:-1]
    n_repl = 0
    for raw in lines:
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # count in total, can't attribute
        wid = row.get("worker_id")
        if isinstance(wid, int) and wid >= workers:
            n_repl += 1
    return offset + last_nl + 1, len(lines), n_repl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="JSONL path being written")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--workers", type=int, required=True,
                    help="--workers of the run; worker_id >= this marks a replacement")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    offset = 0
    total = 0
    repl = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("epoch,total,replacement\n")
        out.flush()
        while True:
            offset, n, r = scan_new_lines(args.results, offset, args.workers)
            total += n
            repl += r
            out.write(f"{time.time():.3f},{total},{repl}\n")
            out.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
