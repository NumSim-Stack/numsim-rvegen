#!/usr/bin/env python3
"""Run the rvegen benchmark cases and report times and scaling.

    scripts/run_benchmarks.py --rvegen build/rvegen

runs every case of bench/cases.py (whole pipeline runs of the binary), each
`--repeat` times, and writes `results.json` and `results.md` to `--out`.

  --quick             leave out the largest sizes
  --filter REGEX      only the cases whose name matches
  --check-scaling     exit 1 if the run time of a family grows faster with
                      its size than `--max-exponent` allows (see below)
  --compare OLD.json  print new / old per case, exit 1 if a case got slower
                      by more than `--max-ratio`
  --emit DIR          write the configs and input files to DIR, run nothing
  --list              print the case names

The time of a case is the fastest of its repeats: wall-clock time of the
whole process, so reading the config, generating and writing are all in it.

Scaling check. Absolute times depend on the machine and are too noisy on a
shared CI runner to gate anything. How the time grows with the size is not:
for two sizes of one family the check computes

    exponent = log(t2 / t1) / log(w2 / w1)

with t the time less the start-up time of the binary (the `startup` case)
and w the work of the run (placement attempts, or shapes generated). Work
that is linear in the size gives 1, n log n a little more, and a pass over
everything per inclusion gives 2.
"""

import argparse
import datetime
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_cases():
    spec = importlib.util.spec_from_file_location(
        "rvegen_bench_cases", ROOT / "bench" / "cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cases()


def startup_case():
    """A run with next to nothing to do: the cost of starting the binary."""
    return {
        "name": "startup", "family": "startup", "size": 1, "work": None,
        "large": False, "files": {},
        "config": {
            "domain_box": [1.0, 1.0],
            "seed": 1,
            "distributions": {
                "pos": {"type": "uniform_real", "a": 0.2, "b": 0.8},
                "radius": {"type": "constant", "value": 0.05}},
            "shapes": [{"type": "circle_input", "pos_x_dist": "pos",
                        "pos_y_dist": "pos", "radius_dist": "radius"}],
            "generator": {"type": "only_inside", "max_attempts": 1000},
            "termination": {"type": "number_of_inclusions", "target": 1},
            "postprocess": [],
        },
    }


def write_case(case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(case["config"]))
    config["postprocess"] = list(config.get("postprocess", [])) + [
        {"type": "run_summary", "output_path": "summary.json"}]
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    for name, text in case["files"].items():
        (directory / name).write_text(text)


def run_case(case, rvegen, work, repeat, timeout):
    directory = work / case["name"]
    if directory.exists():
        shutil.rmtree(directory)
    write_case(case, directory)
    result = {key: case[key] for key in ("name", "family", "size", "work")}
    result.update(status="ok", seconds=None, runs=[], shapes=None,
                  inclusions=None, attempts=None, stop_reason=None)
    for _ in range(repeat):
        start = time.perf_counter()
        try:
            done = subprocess.run(
                [str(rvegen), "config.json", "--quiet"], cwd=directory,
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            result["status"] = f"timeout after {timeout:g} s"
            return result
        seconds = time.perf_counter() - start
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
            result["status"] = "failed: " + " | ".join(tail)
            return result
        result["runs"].append(seconds)
    result["seconds"] = min(result["runs"])
    summary = json.loads((directory / "summary.json").read_text())
    report = summary.get("report") or {}
    result["shapes"] = summary.get("n_shapes")
    result["inclusions"] = summary.get("n_inclusions")
    result["attempts"] = report.get("attempts") or None
    result["stop_reason"] = report.get("stop_reason")
    if result["stop_reason"] == "max_attempts":
        result["status"] = "failed: the run hit max_attempts"
    return result


def work_of(result):
    if result["work"] == "attempts":
        return result["attempts"]
    if result["work"] == "shapes":
        return result["shapes"]
    return None


def scaling(results, max_exponent, min_seconds):
    """Rows (family, smaller, larger, exponent or None, verdict)."""
    startup = next((r["seconds"] for r in results
                    if r["name"] == "startup" and r["status"] == "ok"), 0.0)
    families = {}
    for r in results:
        if r["work"] is not None:
            families.setdefault(r["family"], []).append(r)
    rows = []
    for family, members in families.items():
        members.sort(key=lambda r: r["size"])
        for small, large in zip(members, members[1:]):
            row = {"family": family, "small": small["name"],
                   "large": large["name"], "exponent": None}
            if small["status"] != "ok":
                row["verdict"] = "not checked: " + small["name"] + " did not run"
            elif large["status"] != "ok":
                row["verdict"] = "FAIL: " + large["status"]
            else:
                t1 = small["seconds"] - startup
                t2 = large["seconds"] - startup
                w1, w2 = work_of(small), work_of(large)
                if not w1 or not w2 or w2 <= w1:
                    row["verdict"] = "not checked: no work count"
                elif t1 < min_seconds:
                    row["verdict"] = (f"not checked: {small['name']} takes "
                                      f"under {min_seconds * 1000:g} ms")
                else:
                    row["exponent"] = math.log(t2 / t1) / math.log(w2 / w1)
                    row["verdict"] = ("ok" if row["exponent"] <= max_exponent
                                      else f"FAIL: above {max_exponent:g}")
            rows.append(row)
    return rows


def compare(results, old_results, max_ratio, min_seconds):
    """Rows (name, old, new, ratio, verdict) for the cases in both sets."""
    old = {r["name"]: r for r in old_results}
    rows = []
    for r in results:
        o = old.get(r["name"])
        if o is None or o.get("status") != "ok":
            continue
        row = {"name": r["name"], "old": o["seconds"], "new": r["seconds"],
               "ratio": None}
        if r["status"] != "ok":
            row["verdict"] = "FAIL: " + r["status"]
        else:
            row["ratio"] = r["seconds"] / o["seconds"]
            slower = (row["ratio"] > max_ratio and
                      max(r["seconds"], o["seconds"]) >= min_seconds)
            row["verdict"] = (f"FAIL: more than {max_ratio:g} times slower"
                              if slower else "ok")
        rows.append(row)
    return rows


def machine():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(),
            "system": f"{platform.system()} {platform.release()}"}


def seconds_text(value):
    if value is None:
        return ""
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def count_text(value):
    return "" if value is None else f"{value:,}".replace(",", " ")


def markdown(results, scaling_rows, compare_rows, info):
    lines = [f"rvegen {info['rvegen_version']}, {info['date']}, "
             f"{info['machine']['cpu']} ({info['machine']['cores']} logical "
             f"cores), {info['machine']['system']}; fastest of "
             f"{info['repeat']} run(s) per case.", ""]
    lines += ["| case | size | shapes | attempts | time [s] | per unit [µs] |",
              "|---|---:|---:|---:|---:|---:|"]
    family = None
    for r in results:
        if r["family"] != family:
            family = r["family"]
            lines.append(f"| **{family}** | | | | | |")
        unit = ""
        if r["status"] == "ok" and work_of(r):
            unit = f"{1e6 * r['seconds'] / work_of(r):.2f}"
        time_text = (seconds_text(r["seconds"]) if r["status"] == "ok"
                     else r["status"])
        lines.append(f"| `{r['name']}` | {count_text(r['size'])} | "
                     f"{count_text(r['shapes'])} | "
                     f"{count_text(r['attempts'])} | {time_text} | {unit} |")
    if scaling_rows:
        lines += ["", "Scaling (exponent of the time over the work, 1 = "
                  "linear, 2 = quadratic):", "",
                  "| family | from | to | exponent | |", "|---|---|---|---:|---|"]
        for row in scaling_rows:
            exponent = ("" if row["exponent"] is None
                        else f"{row['exponent']:.2f}")
            lines.append(f"| {row['family']} | `{row['small']}` | "
                         f"`{row['large']}` | {exponent} | {row['verdict']} |")
    if compare_rows:
        lines += ["", "Against the reference:", "",
                  "| case | reference [s] | now [s] | ratio | |",
                  "|---|---:|---:|---:|---|"]
        for row in compare_rows:
            ratio = "" if row["ratio"] is None else f"{row['ratio']:.2f}"
            lines.append(f"| `{row['name']}` | {seconds_text(row['old'])} | "
                         f"{seconds_text(row['new'])} | {ratio} | "
                         f"{row['verdict']} |")
    return "\n".join(lines) + "\n"


def self_test():
    def fake(name, family, size, seconds, work="shapes", status="ok"):
        return {"name": name, "family": family, "size": size, "work": work,
                "status": status, "seconds": seconds, "shapes": size,
                "attempts": size}

    start = fake("startup", "startup", 1, 0.01, work=None)
    linear = [start, fake("a1", "a", 1000, 0.11), fake("a2", "a", 10000, 1.01)]
    rows = scaling(linear, 1.7, 0.01)
    assert len(rows) == 1 and rows[0]["verdict"] == "ok", rows
    assert abs(rows[0]["exponent"] - 1.0) < 1e-9, rows
    quadratic = [start, fake("a1", "a", 1000, 0.11),
                 fake("a2", "a", 10000, 10.01)]
    rows = scaling(quadratic, 1.7, 0.01)
    assert rows[0]["verdict"].startswith("FAIL"), rows
    assert abs(rows[0]["exponent"] - 2.0) < 1e-9, rows
    # Too fast to measure: not judged either way.
    rows = scaling([start, fake("a1", "a", 10, 0.011),
                    fake("a2", "a", 100, 0.5)], 1.7, 0.01)
    assert rows[0]["verdict"].startswith("not checked"), rows
    # The larger size not finishing is a failure, the smaller one is not.
    rows = scaling([start, fake("a1", "a", 1000, 0.11),
                    fake("a2", "a", 10000, None, status="timeout")], 1.7, 0.01)
    assert rows[0]["verdict"].startswith("FAIL"), rows
    rows = scaling([fake("a1", "a", 1000, None, status="failed: x"),
                    fake("a2", "a", 10000, 1.0)], 1.7, 0.01)
    assert rows[0]["verdict"].startswith("not checked"), rows
    # Attempts as the work, three sizes give two rows in order of size.
    three = [fake("b3", "b", 300, 0.9, work="attempts"),
             fake("b1", "b", 100, 0.3, work="attempts"),
             fake("b2", "b", 200, 0.6, work="attempts")]
    rows = scaling(three, 1.7, 0.01)
    assert [r["small"] for r in rows] == ["b1", "b2"], rows
    assert all(abs(r["exponent"] - 1.0) < 1e-9 for r in rows), rows
    # Cases without work are not in the scaling check.
    assert scaling([fake("w", "writers", 5, 1.0, work=None)], 1.7, 0.01) == []

    old = [fake("a", "a", 1, 1.0), fake("b", "a", 2, 0.001),
           fake("c", "a", 3, 1.0), fake("gone", "a", 4, 1.0)]
    new = [fake("a", "a", 1, 1.4), fake("b", "a", 2, 0.004),
           fake("c", "a", 3, 1.6), fake("fresh", "a", 5, 9.0)]
    rows = {r["name"]: r for r in compare(new, old, 1.5, 0.05)}
    assert set(rows) == {"a", "b", "c"}, rows
    assert rows["a"]["verdict"] == "ok", rows
    assert rows["b"]["verdict"] == "ok", rows       # too short to judge
    assert rows["c"]["verdict"].startswith("FAIL"), rows
    rows = compare([fake("a", "a", 1, None, status="timeout")], old, 1.5, 0.05)
    assert rows[0]["verdict"].startswith("FAIL"), rows

    names = [c["name"] for c in load_cases()]
    assert len(names) == len(set(names)), "case names must be unique"
    assert "startup" not in names
    for case in load_cases():
        json.dumps(case["config"])
        assert case["work"] in ("attempts", "shapes", None), case["name"]
    print("run_benchmarks self-test: all checks passed")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rvegen", type=Path, help="the rvegen binary")
    parser.add_argument("--out", type=Path, default=Path("bench-results"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="seconds per run (default 300)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--filter", default="")
    parser.add_argument("--check-scaling", action="store_true")
    parser.add_argument("--max-exponent", type=float, default=1.7)
    parser.add_argument("--min-seconds", type=float, default=0.01,
                        help="scaling: smallest time, less start-up, that is "
                             "judged (default 0.01)")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--max-ratio", type=float, default=1.5)
    parser.add_argument("--emit", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    cases = [c for c in load_cases()
             if re.search(args.filter, c["name"])
             and not (args.quick and c["large"])]
    if args.list:
        for case in cases:
            print(case["name"])
        return 0
    if args.emit:
        for case in cases:
            write_case(case, args.emit / case["name"])
        print(f"{len(cases)} cases written to {args.emit}")
        return 0
    if args.rvegen is None:
        parser.error("--rvegen is required to run the cases")
    rvegen = args.rvegen.resolve()
    if not rvegen.is_file():
        parser.error(f"no such binary: {rvegen}")
    if not cases:
        parser.error("no case matches")
    old = json.loads(args.compare.read_text()) if args.compare else None

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    results = []
    for case in [startup_case()] + cases:
        result = run_case(case, rvegen, work, args.repeat, args.timeout)
        results.append(result)
        shown = (seconds_text(result["seconds"]) + " s"
                 if result["status"] == "ok" else result["status"])
        print(f"{result['name']:32s} {shown}", flush=True)

    version = ""
    summary = work / "startup" / "summary.json"
    if summary.is_file():
        version = json.loads(summary.read_text()).get("rvegen_version", "")
    info = {"rvegen_version": version, "machine": machine(),
            "date": datetime.date.today().isoformat(), "repeat": args.repeat}
    scaling_rows = scaling(results, args.max_exponent, args.min_seconds)
    compare_rows = (compare(results, old["cases"], args.max_ratio, 0.05)
                    if old else [])
    for r in results:
        if r["runs"]:
            r["median_seconds"] = statistics.median(r["runs"])
    (args.out / "results.json").write_text(json.dumps(
        dict(info, cases=results, scaling=scaling_rows), indent=1) + "\n")
    text = markdown(results, scaling_rows, compare_rows, info)
    (args.out / "results.md").write_text(text)
    print()
    print(text)

    broken = [r["name"] for r in results if r["status"] != "ok"]
    if broken:
        print("did not run: " + ", ".join(broken), file=sys.stderr)
    failed = False
    if args.check_scaling:
        bad = [row for row in scaling_rows if row["verdict"].startswith("FAIL")]
        for row in bad:
            print(f"scaling: {row['small']} -> {row['large']}: "
                  f"{row['verdict']}", file=sys.stderr)
        failed = failed or bool(bad)
    if compare_rows:
        bad = [row for row in compare_rows if row["verdict"].startswith("FAIL")]
        for row in bad:
            print(f"compare: {row['name']}: {row['verdict']}", file=sys.stderr)
        failed = failed or bool(bad)
    if failed:
        return 1
    return 2 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
