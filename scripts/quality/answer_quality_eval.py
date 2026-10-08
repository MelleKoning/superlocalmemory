#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Measure answer quality against your own labelled questions.

Two modes, one gold file (JSON Lines; see superlocalmemory.evaluation.answer_quality):

  retrieval  Run recall in this process against a COPY of a store and report
             where the right memory lands: hit@1/3/5, MRR, per category, and
             recall latency p50/p95. The answer check is switched off: nothing
             is judged and nothing leaves the machine.

      python scripts/quality/answer_quality_eval.py retrieval \\
          --gold gold.jsonl --data-dir /path/to/copy [--out per-question.jsonl]

      Make the copy with SQLite's backup command (safe while SLM is running):
          sqlite3 ~/.superlocalmemory/memory.db ".backup '/tmp/copy/memory.db'"
      and put a config.json beside it. Your own store is refused: an
      in-process engine next to a running daemon would be a second writer.

  judges     Ask a RUNNING daemon each question with its answer check on, and
             report, per judge (Laya, Jev, Jev with reordering): correct-accept,
             false-accept, false-abstain, correct-abstain, plus the same counts
             at every threshold. Run it once per judge you want measured.

      python scripts/quality/answer_quality_eval.py judges \\
          --gold gold.jsonl [--data-dir DIR] [--max-false-accept 0.10]

      ``--data-dir`` names the store whose daemon to ask (default: yours).
      With Jev selected this sends each question and its top memories to the
      hosted check, exactly as an ordinary recall does.

Output is counts, ranks, ids and timings only — never memory text.
Exit codes: 0 measured · 2 bad input · 3 refused (unsafe data folder or a
reranker that never became ready) · 4 the daemon did not answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path


def _bootstrap_source_tree() -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


_bootstrap_source_tree()

from superlocalmemory.evaluation.answer_quality import (  # noqa: E402
    DEFAULT_SWEEP,
    GoldFileError,
    JudgedRecall,
    RankedResult,
    choose_threshold,
    first_right_rank,
    judge_report,
    load_gold,
    retrieval_report,
)

_ENV_ROOTS = ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")


class EvalError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


# -- where the data is -----------------------------------------------------------

def _account_home() -> Path:
    """This account's home folder, whatever HOME or USERPROFILE say.

    POSIX: the password database. Windows (no ``pwd``): the profile folder
    of the account the process runs as (CSIDL_PROFILE), not an env variable.
    """
    if os.name != "nt":
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    import ctypes

    buffer = ctypes.create_unicode_buffer(32_768)
    if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0028, None, 0, buffer) != 0:
        raise OSError("could not read your Windows profile folder")
    return Path(buffer.value)


def _owner_roots() -> set[Path]:
    """Every place this account's own store could be."""
    from superlocalmemory.infra.data_root import canonical_data_root

    saved = {k: os.environ.pop(k) for k in _ENV_ROOTS if k in os.environ}
    try:
        home = _account_home()
        return {canonical_data_root().resolve(),
                canonical_data_root(home=home).resolve(),
                (home / ".superlocalmemory").resolve()}
    finally:
        os.environ.update(saved)


def _use_data_dir(path: Path) -> None:
    for key in _ENV_ROOTS:
        os.environ.pop(key, None)
    os.environ["SLM_DATA_DIR"] = str(path)


def _copy_data_dir(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not (path / "memory.db").is_file():
        raise EvalError(2, f"{path} has no memory.db")
    if path in _owner_roots():
        raise EvalError(3, "refusing your own store: measure a copy made with "
                           "sqlite3 '.backup' instead (see --help)")
    return path


# -- retrieval -------------------------------------------------------------------

def _build_engine(warm_timeout_s: float, require_reranker: bool):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # never download a model
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from dataclasses import replace

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    config = SLMConfig.load()
    config.retrieval = replace(config.retrieval, sufficiency_judge="off")
    engine = MemoryEngine(config)
    engine.initialize()
    reranker = getattr(getattr(engine, "_retrieval_engine", None), "_reranker", None)
    ready = reranker is not None and reranker.warmup_sync(timeout=warm_timeout_s)
    if require_reranker and getattr(config.retrieval, "use_cross_encoder", False) and not ready:
        engine.close()
        raise EvalError(3, "the reranker did not become ready; a measurement "
                           "without it would not describe real recall "
                           "(use --allow-no-reranker to measure anyway)")
    return engine


def _wait_until_complete(engine, fast, timeout_s: float) -> None:
    """Measure only a complete search. Every channel that is still warming
    (the embedding model, Hopfield, spreading activation) makes a recall
    incomplete, and the product says so; measuring then compares a partial
    search. Probe until one recall comes back complete."""
    deadline = time.monotonic() + timeout_s
    missing: tuple = ()
    while time.monotonic() < deadline:
        missing = tuple(getattr(engine.recall("warm up", limit=1, fast=fast),
                                "incomplete_channels", ()) or ())
        if not missing:
            return
        time.sleep(0.5)
    raise EvalError(3, "these channels were still warming after "
                       f"{timeout_s:.0f} s: {', '.join(missing)}; a measurement now "
                       "would not describe complete recall")


def _ranked(response) -> list[RankedResult]:
    return [RankedResult(memory_id=r.fact.memory_id or "", fact_id=r.fact.fact_id or "")
            for r in response.results]


def run_retrieval(args: argparse.Namespace) -> dict:
    from superlocalmemory.evaluation.gold_presence import check_presence

    questions = load_gold(Path(args.gold))
    data_dir = _copy_data_dir(args.data_dir)
    _use_data_dir(data_dir)
    engine = _build_engine(args.warm_timeout, not args.allow_no_reranker)
    # Is each labelled answer in this store at all? Checked before any recall,
    # read only: a missing answer is "answer not stored", never a miss.
    presence = check_presence(data_dir / "memory.db", questions,
                              profile_id=str(getattr(engine, "profile_id", "default")))
    from superlocalmemory.core.recall_pipeline import resolve_hot_path_fast

    fast = resolve_hot_path_fast(None, engine._config)
    _wait_until_complete(engine, fast, args.warm_timeout)
    incomplete_recalls = 0
    ranks: dict[str, int | None] = {}
    latencies: list[float] = []
    unstable: list[str] = []
    rerank_status: dict[str, int] = {}
    rows = []
    try:
        for q in questions:
            seen_ranks = []
            for _ in range(args.repeats):
                t0 = time.monotonic()
                response = engine.recall(q.question, limit=args.limit, fast=fast)
                latencies.append((time.monotonic() - t0) * 1000.0)
                if getattr(response, "incomplete_channels", ()):
                    incomplete_recalls += 1
                status = getattr(response, "reranker_status", "") or "unknown"
                rerank_status[status] = rerank_status.get(status, 0) + 1
                results = _ranked(response)
                seen_ranks.append(first_right_rank(q, results) if q.answerable else None)
            ranks[q.qid] = seen_ranks[0]
            if len(set(seen_ranks)) > 1:
                unstable.append(q.qid)
            row = {
                "qid": q.qid, "category": q.category, "answerable": q.answerable,
                "rank": seen_ranks[0], "ranks_over_repeats": seen_ranks,
                "query_type": response.query_type,
                "no_confident_match": bool(getattr(response, "no_confident_match", False)),
                "top": [{"memory_id": r.memory_id, "fact_id": r.fact_id}
                        for r in results[:args.limit]],
            }
            if q.qid in presence:
                row["answer_presence"] = presence[q.qid].as_row()
            rows.append(row)
    finally:
        engine.close()
    if args.out:
        _write_rows(Path(args.out), rows)
    report = retrieval_report(ranks, questions, latencies)
    report.update(_presence_blocks(presence, ranks, questions))
    report["repeats"] = args.repeats
    report["rank_changed_between_repeats"] = unstable
    report["incomplete_recalls"] = incomplete_recalls
    report["reranker_status"] = rerank_status
    unanswerable = [r for r in rows if not r["answerable"]]
    report["unanswerable"] = {
        "n": len(unanswerable),
        "no_confident_match": sum(1 for r in unanswerable if r["no_confident_match"]),
    }
    return report


def _presence_blocks(presence: dict, ranks: dict, questions) -> dict:
    """The presence summary, and hit@k over the questions whose answer is stored."""
    from superlocalmemory.evaluation.gold_presence import NOT_STORED, presence_summary

    stored = [q for q in questions
              if q.answerable and presence[q.qid].status != NOT_STORED]
    blocks = {"answer_presence": presence_summary(presence, ranks)}
    if stored:
        blocks["stored_only"] = retrieval_report(ranks, stored)["overall"]
    return blocks


# -- judges ----------------------------------------------------------------------

def _provider(body: dict) -> str:
    """Which judge decided: from the calibration id the response carries."""
    if body.get("reranker_status") == "jev_listwise":
        return "jev-listwise"
    calibration = str(body.get("calibration_id") or "")
    return calibration.split(":", 1)[0] or "unknown"


def _confidence(body: dict) -> float | None:
    value = body.get("answer_confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ask_daemon(question: str, limit: int, timeout_s: float) -> dict:
    from superlocalmemory.cli.daemon import daemon_request

    query = urllib.parse.urlencode({"q": question, "limit": limit, "answer_check": "full"})
    body = daemon_request("GET", f"/recall?{query}", timeout_seconds=timeout_s)
    if not isinstance(body, dict) or "results" not in body:
        raise EvalError(4, "the daemon did not answer /recall (is it running?)")
    return body


def run_judges(args: argparse.Namespace) -> dict:
    questions = load_gold(Path(args.gold))
    if args.data_dir:
        _use_data_dir(Path(args.data_dir).expanduser().resolve())
    judged: list[JudgedRecall] = []
    not_judged: dict[str, int] = {}
    latencies: list[float] = []
    rows = []
    for q in questions:
        t0 = time.monotonic()
        body = _ask_daemon(q.question, args.limit, args.timeout)
        latencies.append((time.monotonic() - t0) * 1000.0)
        results = [RankedResult(memory_id=str(r.get("memory_id") or ""),
                                fact_id=str(r.get("fact_id") or ""))
                   for r in body.get("results", [])]
        rank = first_right_rank(q, results) if q.answerable else None
        status = str(body.get("answer_check_status") or "skipped")
        row = {"qid": q.qid, "answerable": q.answerable, "rank": rank,
               "status": status, "provider": _provider(body),
               "abstained": bool(body.get("abstained")),
               "confidence": _confidence(body)}
        rows.append(row)
        if status != "judged":
            not_judged[status] = not_judged.get(status, 0) + 1
            continue
        judged.append(JudgedRecall(
            qid=q.qid, provider=row["provider"],
            answered=rank is not None and rank <= args.judged_depth,
            abstained=row["abstained"], confidence=row["confidence"],
        ))
    if args.out:
        _write_rows(Path(args.out), rows)
    report = {"judged_depth": args.judged_depth, "not_judged": not_judged,
              "providers": judge_report(judged, DEFAULT_SWEEP)}
    if latencies:
        from superlocalmemory.evaluation.answer_quality import nearest_rank_ms
        report["recall_total_ms"] = {"p50": round(nearest_rank_ms(latencies, 50), 1),
                                     "p95": round(nearest_rank_ms(latencies, 95), 1)}
    for provider, block in report["providers"].items():
        block["suggested"] = choose_threshold(
            block["sweep"], max_false_accept_rate=args.max_false_accept)
    if args.write_calibration:
        report["calibration_file"] = _write_calibration(
            Path(args.write_calibration), report["providers"])
    return report


def calibration_entries(providers: dict) -> list[dict]:
    """One file entry per judge whose suggested threshold rests on enough data."""
    from superlocalmemory.retrieval import judge_calibration_file as jcf
    from superlocalmemory.retrieval.judge_recipe import ACTIVE_RECIPE

    entries = []
    for provider, block in sorted(providers.items()):
        row = block.get("suggested")
        if provider not in jcf.BACKENDS or row is None:
            continue
        answered = row["correct_accept"] + row["false_abstain"]
        unanswered = row["false_accept"] + row["correct_abstain"]
        if answered < jcf.MIN_ANSWERED or unanswered < jcf.MIN_UNANSWERED:
            continue
        entries.append({"backend": provider, "recipe_id": ACTIVE_RECIPE.recipe_id,
                        "threshold": row["threshold"], "answered": answered,
                        "unanswered": unanswered,
                        "measured_at": time.strftime("%Y-%m-%d")})
    return entries


def _write_calibration(path: Path, providers: dict) -> dict:
    """Write the measured thresholds, never a file the product would refuse."""
    from superlocalmemory.retrieval import judge_calibration_file as jcf

    entries = calibration_entries(providers)
    if not entries:
        return {"written": False, "reason": "no judge had enough answered and "
                f"unanswered questions ({jcf.MIN_ANSWERED}/{jcf.MIN_UNANSWERED})"}
    payload = {"schema": jcf.SCHEMA, "entries": entries}
    jcf.parse_entries(payload)  # the same check the product applies
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"written": True, "path": str(path), "backends": [e["backend"] for e in entries]}


# -- shared ----------------------------------------------------------------------

def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("retrieval", help="where the right memory lands (copy of a store)")
    r.add_argument("--gold", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--warm-timeout", type=float, default=300.0)
    r.add_argument("--allow-no-reranker", action="store_true")
    r.add_argument("--out", help="per-question ranks and ids (JSON Lines)")
    j = sub.add_parser("judges", help="what a running daemon's answer check decides")
    j.add_argument("--gold", required=True)
    j.add_argument("--data-dir", help="the store whose daemon to ask (default: yours)")
    j.add_argument("--limit", type=int, default=10)
    j.add_argument("--judged-depth", type=int, default=3,
                   help="how many top results the check reads (default 3)")
    j.add_argument("--max-false-accept", type=float, default=0.10)
    j.add_argument("--timeout", type=float, default=30.0)
    j.add_argument("--out", help="per-question outcomes (JSON Lines)")
    j.add_argument("--write-calibration", metavar="PATH",
                   help="write the suggested thresholds as an "
                        "answer_check_calibration.json (review before use)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_retrieval(args) if args.mode == "retrieval" else run_judges(args)
    except (GoldFileError, OSError) as exc:
        print(f"answer-quality: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"answer-quality: {exc}", file=sys.stderr)
        return exc.code
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
