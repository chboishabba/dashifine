#!/usr/bin/env python3
"""
Temp helper: pull all missing DASHI chat IDs from doc_online_chats_mentioning_dashi.md
into the canonical sqlite archive via the ITIR-suite pull_to_structurer helper.
"""

import json
import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parents[1] / "doc_online_chats_mentioning_dashi.md"
DB_PATH = Path.home() / "chat_archive.sqlite"
PULL_SCRIPT = (
    Path("/home/c/Documents/code/ITIR-suite/reverse-engineered-chatgpt/scripts")
    / "pull_to_structurer.py"
)
PYTHON_BIN = Path("/home/c/Documents/code/ITIR-suite/.venv/bin/python")


def parse_missing_ids(doc_text: str) -> list[str]:
    pattern = r"\{title: '([^']*)', id: '([0-9a-f-]{36})', local_db: 'missing'(\s*,[^}]*)?\}"
    return [match[1] for match in re.findall(pattern, doc_text)]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Pull missing DASHI chat IDs into the canonical chat archive."
    )
    parser.add_argument(
        "--start-after",
        help="Resume after this chat UUID; fetch begins with the next missing ID.",
    )
    parser.add_argument(
        "--stop-after",
        help="Stop after this chat UUID has been processed.",
    )
    return parser


def slice_missing_ids(missing_ids: list[str], start_after: str | None) -> list[str]:
    if not start_after:
        return missing_ids
    try:
        index = missing_ids.index(start_after)
    except ValueError as exc:
        raise SystemExit(f"--start-after UUID not found in missing list: {start_after}") from exc
    return missing_ids[index + 1 :]


def fetch_id(chat_id: str) -> dict:
    cmd = [
        str(PYTHON_BIN),
        str(PULL_SCRIPT),
        "--ids",
        chat_id,
        "--db",
        str(DB_PATH),
        "--engine",
        "async",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "chat_id": chat_id,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:  # noqa: D103
    args = parse_args().parse_args()
    if not DOC_PATH.exists():
        raise SystemExit(f"{DOC_PATH} missing; run from repo root.")
    doc_text = DOC_PATH.read_text()
    missing_ids = slice_missing_ids(parse_missing_ids(doc_text), args.start_after)
    if not missing_ids:
        print("no missing DASHI chat IDs found in doc_online_chats_mentioning_dashi.md")
        return
    print(f"pulling {len(missing_ids)} missing chat IDs into {DB_PATH}")

    results = []
    total = len(missing_ids)
    for index, chat_id in enumerate(missing_ids, start=1):
        print(f"[{index}/{total}] fetching {chat_id}", flush=True)
        result = fetch_id(chat_id)
        results.append(result)
        status = "ok" if result["returncode"] == 0 else f"fail rc={result['returncode']}"
        print(f"[{index}/{total}] {status} {chat_id}", flush=True)
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr, flush=True)
        if args.stop_after and chat_id == args.stop_after:
            break

    summary = {
        "total": len(results),
        "succeeded": sum(1 for r in results if r["returncode"] == 0),
        "failed": sum(1 for r in results if r["returncode"] != 0),
    }
    print(json.dumps(summary, indent=2))
    for result in results:
        if result["returncode"] != 0:
            print(
                f"FAIL {result['chat_id']}: rc={result['returncode']} stderr:\n"
                f"{result['stderr']}"
            )


if __name__ == "__main__":
    main()
