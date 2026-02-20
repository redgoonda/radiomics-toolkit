"""Output writers: dict → CSV / JSON."""

from __future__ import annotations

import json
from pathlib import Path


def write_results(results: dict, output_path: str | Path) -> None:
    """Write *results* to *output_path*.

    The format is inferred from the file extension:
    - ``.csv`` → flat CSV (one row, feature names as columns)
    - ``.json`` → indented JSON
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        _write_csv(results, output_path)
    elif suffix == ".json":
        _write_json(results, output_path)
    else:
        raise ValueError(
            f"Unsupported output format '{suffix}'. Use .csv or .json."
        )


def _write_csv(results: dict, path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for CSV output: pip install pandas") from exc

    df = pd.DataFrame([results])
    df.to_csv(path, index=False)


def _write_json(results: dict, path: Path) -> None:
    # Convert numpy scalars to Python natives for JSON serialisation
    clean = {}
    for k, v in results.items():
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            clean[k] = v

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
