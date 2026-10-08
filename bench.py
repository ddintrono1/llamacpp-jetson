#!/usr/bin/env python3
"""Benchmark the running llama-server: speed, speculative-decoding acceptance and energy.

For each prompt in the prompts file, runs N completions and samples module power from the
board's INA3221 power monitors (sysfs) during each request. Energy = mean power x request time (prompt + generation).
Writes one CSV per session and prints a per-prompt summary at the end.

The API key is read from the LLAMA_API_KEY environment variable (exported by the makefile from .env).
"""

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# INA3221 power monitors: each hwmon folder has, per channel <n>, a rail name (in<n>_label),
# bus voltage in mV (in<n>_input) and current in mA (curr<n>_input). Same source as tegrastats and jtop.
INA3221_DIR = Path("/sys/bus/i2c/drivers/ina3221")

CSV_FIELDS = [
    "timestamp", "model", "prompt_id", "run",
    "n_prompt", "n_gen", "pp_tok_s", "tg_tok_s",
    "draft_n", "draft_n_accepted", "accept_rate",
    "power_w", "energy_j", "tok_per_j",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://0.0.0.0:8080", help="llama-server base URL")
    p.add_argument("--prompts", default="resources/bench_prompts.txt", help="one prompt per line; blank lines and # comments are skipped")
    p.add_argument("--runs", type=int, default=5, help="repetitions per prompt")
    p.add_argument("--n-predict", type=int, default=128, help="tokens to generate per run")
    p.add_argument("--interval", type=int, default=100, help="power sampling period (ms)")
    p.add_argument("--rails", default="VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0", help="|-separated power rails summed as module power (AGX Orin default)")
    p.add_argument("--out-dir", default="results", help="directory for the session CSV")
    return p.parse_args()


def load_prompts(path):
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def api(url, key, path, body=None):
    """GET (body None) or POST JSON to the server; returns the decoded JSON response."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: HTTP {e.code} {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"{path}: cannot reach {url} ({e.reason}). Is the server running?")


def find_rails(names):
    """Map each requested rail name to its (voltage, current) sysfs files."""
    available = {}
    for label in INA3221_DIR.glob("*/hwmon/hwmon*/in*_label"):
        n = label.name[len("in"):-len("_label")]
        curr = label.with_name(f"curr{n}_input")
        if curr.exists():  # skips the "sum of shunt voltages" channel, which has no current
            available[label.read_text().strip()] = (label.with_name(f"in{n}_input"), curr)
    missing = names - available.keys()
    if missing:
        sys.exit(f"rails not found: {sorted(missing)}; available: {sorted(available)}")
    return [available[n] for n in sorted(names)]


def read_power_w(rails):
    """Instantaneous sum of the rails' power: mV x mA = uW."""
    return sum(int(volt.read_text()) * int(curr.read_text()) for volt, curr in rails) / 1e6


class PowerSampler:
    """Context manager that samples rail power in a background thread while the block runs."""

    def __init__(self, rails, interval_s):
        self.rails, self.interval_s = rails, interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while True:
            self.samples.append(read_power_w(self.rails))
            if self._stop.wait(self.interval_s):
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def mean_w(self):
        return statistics.mean(self.samples)


def run_once(args, key, rails, prompt):
    body = {"prompt": prompt, "n_predict": args.n_predict, "cache_prompt": False, "temperature": 0}
    with PowerSampler(rails, args.interval / 1000) as sampler:
        t = api(args.url, key, "/completion", body)["timings"]
    power = sampler.mean_w

    energy = (t["prompt_ms"] + t["predicted_ms"]) / 1000 * power
    draft_n, accepted = t.get("draft_n"), t.get("draft_n_accepted")
    return {
        "n_prompt": t["prompt_n"],
        "n_gen": t["predicted_n"],
        "pp_tok_s": round(t["prompt_per_second"], 2),
        "tg_tok_s": round(t["predicted_per_second"], 2),
        "draft_n": draft_n,
        "draft_n_accepted": accepted,
        "accept_rate": round(accepted / draft_n, 3) if draft_n else None,
        "power_w": round(power, 2),
        "energy_j": round(energy, 1),
        "tok_per_j": round(t["predicted_n"] / energy, 3),
    }


def summarize(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    if not values:
        return "n/a"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3g} ± {sd:.2g}"


def main():
    args = parse_args()
    key = os.environ.get("LLAMA_API_KEY")
    if not key:
        sys.exit("LLAMA_API_KEY is not set (put it in .env or export it)")
    rails = find_rails({r.strip() for r in args.rails.split("|") if r.strip()})
    prompts = load_prompts(args.prompts)
    if not prompts:
        sys.exit(f"no prompts in {args.prompts}")

    model = api(args.url, key, "/props").get("model_path", "unknown")
    out = Path(args.out_dir) / f"{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"model: {model}\nprompts: {len(prompts)} x {args.runs} runs, {args.n_predict} tokens each\ncsv: {out}\n")

    rows = []
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pid, prompt in enumerate(prompts, 1):
            print(f"[prompt {pid}] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            for run in range(1, args.runs + 1):
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": model, "prompt_id": pid, "run": run,
                    **run_once(args, key, rails, prompt),
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(f"  run {run}: tg {row['tg_tok_s']} tok/s, accept {row['accept_rate']}, "
                      f"{row['power_w']} W, {row['tok_per_j']} tok/J")

    print("\nsummary (mean ± stdev)")
    for pid in range(1, len(prompts) + 1):
        pr = [r for r in rows if r["prompt_id"] == pid]
        print(f"  prompt {pid}: tg {summarize(pr, 'tg_tok_s')} tok/s, accept {summarize(pr, 'accept_rate')}, "
              f"{summarize(pr, 'power_w')} W, {summarize(pr, 'tok_per_j')} tok/J")


if __name__ == "__main__":
    main()
