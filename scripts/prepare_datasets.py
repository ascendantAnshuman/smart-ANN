#!/usr/bin/env python
"""Download, verify and process datasets.

Thin wrapper over :mod:`smartann.data.prepare` so that the script entry point
and ``smartann data prepare`` cannot drift apart.

    python scripts/prepare_datasets.py all
    python scripts/prepare_datasets.py sift1m nytimes --force
"""

from __future__ import annotations

import argparse
import sys

from smartann.config import resolve_paths
from smartann.data.prepare import prepare_dataset
from smartann.data.registry import load_registry


def main(argv: list[str] | None = None) -> int:
    known = sorted(load_registry())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help=f"dataset name(s), or 'all'. known: {known}")
    parser.add_argument("--allow-mirror", action="store_true", help="permit registered mirrors")
    parser.add_argument("--no-record-checksum", action="store_true")
    parser.add_argument("--force", action="store_true", help="reprocess even if outputs exist")
    args = parser.parse_args(argv)

    names = known if "all" in args.names else args.names
    unknown = [n for n in names if n not in known]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}; known: {known}")

    paths = resolve_paths()
    print(f"data root: {paths.data_root}")
    for name in names:
        print(f"\n=== {name}")
        result = prepare_dataset(
            name,
            paths=paths,
            record=not args.no_record_checksum,
            allow_mirror=args.allow_mirror,
            force=args.force,
        )
        print(
            f"    n_base={result.n_base:,}  n_queries={result.n_queries:,}  "
            f"dim={result.dim}  space={result.space}  search_space={result.search_space}  "
            f"Nc={result.resolved_nc}"
        )
        for k, v in sorted(result.details.items()):
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
