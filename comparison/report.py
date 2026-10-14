"""Atomic aggregate CSV/JSON reports and one comparable terminal table."""

import csv
import os
from pathlib import Path
import tempfile

from data.osg_storage import write_json_atomic


def write_csv(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rows_for_results(results, recall_k):
    rows = []
    for result in results:
        if result["status"] != "ok":
            rows.append({"method": result["method"], "stage": "-", "status": "failed", "error": result["error"]})
            continue
        for stage, values in result["stages"].items():
            rows.append({"method": result["method"], "stage": stage, "status": "ok",
                         "full_benchmark": result["protocol"]["full_benchmark"],
                         "backbone": result["metadata"]["backbone"], "metric": result["protocol"]["metric"],
                         "batch_size": result["batch_size"], **values["counts"],
                         **{f"recall@{k}": values["metrics"][f"recall@{k}"] for k in recall_k},
                         "mean_batch_encode_ms_per_image": values["timing"]["mean_batch_encode_ms_per_image"],
                         "rerank_top_k": values["timing"].get("rerank_top_k", 0),
                         "reranking_seconds": values["timing"].get("reranking_seconds", 0),
                         "osg_generation_included": False, "error": ""})
    return rows


def write_summary(directory, protocol, results, recall_k):
    rows = rows_for_results(results, recall_k)
    write_json_atomic(directory / "results.json", {"protocol": protocol, "results": results,
                      "notes": ["Recall values are fractions; terminal values are percentages.",
                                "Invalid queries remain in the denominator; invalid database descriptors are excluded from candidates.",
                                "COSMIC loads saved OSGs. Graph generation time is excluded and must be measured separately.",
                                "Mean encode time is batched throughput, not single-query online latency.",
                                "Official checkpoint variants retain their own architecture/preprocessing and checkpoint-selection history."]})
    write_csv(directory / "results.csv", rows)
    return rows


def print_table(rows, recall_k):
    headers = ["Method", "Stage", "Status", "Dim", *[f"R@{k}(%)" for k in recall_k]]
    values = [[row["method"], row["stage"], row["status"], str(row.get("descriptor_dimension", "-")),
               *[f"{row[f'recall@{k}']*100:.2f}" if f"recall@{k}" in row else "-" for k in recall_k]] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in values)) for i, header in enumerate(headers)]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    def line(row):
        return "| " + " | ".join(str(value).ljust(width) for value, width in zip(row, widths)) + " |"
    print("\nComparison\n" + border + "\n" + line(headers) + "\n" + border)
    for row in values:
        print(line(row))
    print(border, flush=True)
