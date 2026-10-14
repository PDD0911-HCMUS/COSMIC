"""Run selected methods, preserve failures, and aggregate comparable results."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from comparison.config import load_comparison_config, project_path
from comparison.datasets import read_benchmark
from comparison.registry import METHOD_NAMES, asset_problems, check_method
from comparison.report import print_table, write_summary
from config import save_config
from data.config import PROJECT_ROOT
from data.osg_storage import write_json_atomic
from runtime import resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/comparison.yaml")
    parser.add_argument("--methods", nargs="+", help="Experiment names from the comparison YAML")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest/settings/assets without loading models")
    parser.add_argument("--max-queries", type=int, help="Smoke test only; database remains complete")
    parser.add_argument("--device")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    config = load_comparison_config(args.config)
    overrides = {key: getattr(args, key) for key in ("max_queries", "device") if getattr(args, key) is not None}
    if args.output_root is not None:
        overrides["output_root"] = project_path(args.output_root)
    config = replace(config, **overrides)
    for spec in config.methods.values():
        check_method(spec.adapter)
    names = args.methods or list(config.methods)
    if len(set(names)) != len(names) or any(name not in config.methods for name in names):
        raise ValueError(f"Select distinct configured methods: {', '.join(config.methods)}")
    if args.list_methods:
        print(f"Registered adapters: {', '.join(METHOD_NAMES)}")
        print(f"Configured methods: {', '.join(config.methods)}")
        return 0
    config = replace(config, methods={name: config.methods[name] for name in names})
    benchmark = read_benchmark(config)
    print(f"AmsterTime: {len(benchmark.query_ids)} queries / {len(benchmark.database_ids)} database images")
    print("Scope: " + ("FULL BENCHMARK" if benchmark.protocol["full_benchmark"] else "SMOKE QUERIES / FULL DATABASE"))
    print(f"Metric: {config.metric} | Recall K: {config.recall_k} | manifest: {benchmark.protocol['manifest_sha256']}")
    missing = {}
    for name, spec in config.methods.items():
        problems = asset_problems(spec)
        if problems:
            missing[name] = problems
        print(f"{name}: adapter={spec.adapter}, stages={','.join(spec.stages)}, "
              + ("MISSING " + "; ".join(problems) if problems else "assets present"))
    if args.dry_run:
        print("Dry run complete. No models loaded or downloaded; asset status is shown above.")
        return 0
    resolve_device(config.device)
    directory = config.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "resolved_config.yaml"
    save_config(config, snapshot)
    write_json_atomic(directory / "manifest.json", {"protocol": benchmark.protocol,
                      "query_ids": benchmark.query_ids, "database_ids": benchmark.database_ids,
                      "positives": benchmark.positives})
    results = []
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for name in names:
        result_path, log_path = directory / f"{name}.json", directory / f"{name}.log"
        error = None
        if name in missing:
            error = "Missing assets: " + "; ".join(missing[name])
        else:
            print(f"Run {name} | log: {log_path}", flush=True)
            command = [sys.executable, "-B", "-m", "comparison.worker", "--config", str(snapshot),
                       "--method", name, "--result", str(result_path)]
            start = perf_counter()
            with log_path.open("w", encoding="utf-8") as stream:
                env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, creationflags=creationflags)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            print(f"{name}: running {perf_counter()-start:.0f}s | log: {log_path}", flush=True)
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if code != 0:
                error = f"Worker exited with code {code}; see {log_path}"
            elif not result_path.is_file():
                error = f"Worker did not write a result; see {log_path}"
            else:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("method") != name or result.get("protocol") != benchmark.protocol:
                    error = "Worker benchmark protocol differs from the shared manifest"
        if error is not None:
            result = {"method": name, "adapter": config.methods[name].adapter, "status": "failed", "error": error}
            write_json_atomic(result_path, result)
            print(f"{name}: FAILED | {error}", flush=True)
        else:
            print(f"{name}: complete", flush=True)
        results.append(result)
        rows = write_summary(directory, benchmark.protocol, results, config.recall_k)
    print_table(rows, config.recall_k)
    print(f"Reports: {directory}", flush=True)
    return int(any(result["status"] != "ok" for result in results))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Comparison error: {error}", file=sys.stderr)
        raise SystemExit(2)
