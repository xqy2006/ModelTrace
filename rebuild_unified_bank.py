from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path
import time

from bank_builder import build_bank, read_rows


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "data" / "unified_bank.json"
STATIC_OUTPUT = PROJECT / "static" / "data" / "unified_bank.json"

SOURCES = {
    "gpt": ("GPT", PROJECT / "data" / "gpt_reference.jsonl"),
    "claude": ("Claude", PROJECT / "data" / "claude_reference.jsonl"),
}


def write_unified_outputs(bank: dict) -> str:
    serialized = json.dumps(bank, ensure_ascii=False, separators=(",", ":")) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STATIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(serialized, encoding="utf-8")
    STATIC_OUTPUT.write_text(serialized, encoding="utf-8")
    return serialized


def main() -> None:
    t0 = time.time()
    all_rows = []
    for family_id, (family_name, path) in SOURCES.items():
        if not path.exists():
            continue
        fam_rows = [
            {
                **row,
                "family_id": family_id,
                "family_name": family_name,
            }
            for row in read_rows(path)
        ]
        if fam_rows:
            fam_bank = build_bank(fam_rows)
            fam_out = PROJECT / "data" / f"{family_id}_bank.json"
            fam_out.write_text(
                json.dumps(fam_bank, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            all_rows.extend(fam_rows)

    bank = build_bank(all_rows)
    serialized = write_unified_outputs(bank)
    elapsed = time.time() - t0
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "static_output": str(STATIC_OUTPUT),
                "size_mb": round(len(serialized.encode("utf-8")) / (1024 * 1024), 2),
                "elapsed_sec": round(elapsed, 2),
                "families": len(SOURCES),
                "models": len(bank["models"]),
                "responses": sum(model["response_count"] for model in bank["models"]),
                "calibration": bank["calibration"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
