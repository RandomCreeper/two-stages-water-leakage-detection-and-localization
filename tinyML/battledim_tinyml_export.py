from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import battledim_tinyml_driver as drv



def load_candidate_artifacts(summary_dir: Path):
    summary_paths = sorted(summary_dir.glob('*_summary.json'))
    memory_paths = sorted(summary_dir.glob('*_memory.csv'))
    if not summary_paths:
        raise FileNotFoundError(f'No *_summary.json files found in {summary_dir}')
    if not memory_paths:
        raise FileNotFoundError(f'No *_memory.csv files found in {summary_dir}')
    summaries = [json.loads(p.read_text()) for p in summary_paths]
    return summaries, memory_paths



def main(summary_dir: Path | str, output_root: Path | str = THIS_DIR, candidate: Optional[str] = None) -> Path:
    summary_dir = Path(summary_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries, memory_paths = load_candidate_artifacts(summary_dir)
    chosen_summary = next((s for s in summaries if s['candidate'] == candidate), None) if candidate else max(summaries, key=drv.summary_selection_key)
    if chosen_summary is None:
        raise KeyError(f'Candidate {candidate!r} not found in {summary_dir}')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f'tinyml_run_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return drv.export_full_run(run_dir, str(chosen_summary['candidate']), summaries, memory_paths)



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export the final BattLeDIM tiny MCU run from precomputed candidate summaries.')
    parser.add_argument('--summary-dir', type=str, required=True, help='Directory containing *_summary.json and *_memory.csv files.')
    parser.add_argument('--output-root', type=str, default=str(THIS_DIR), help='Root directory for the exported run folder.')
    parser.add_argument('--candidate', type=str, default=None, help='Optional candidate name to force instead of automatic selection.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    out = main(summary_dir=Path(args.summary_dir), output_root=Path(args.output_root), candidate=args.candidate)
    print(out)
