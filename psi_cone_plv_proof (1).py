#!/usr/bin/env python3
# psi_cone_plv_proof.py
# Streaming ψ-Cone replication with per-shell sufficient stats, Rayleigh test,
# shell-weighted PLV, and a simple within-shell permutation null (Monte Carlo).
#
# Outputs:
#  - CSV (per shell): shell_z, sum_cos, sum_sin, N_kept, N_total, decimate_k
#  - JSON summary: global μ/PLV, shell-weighted μ/PLV, Rayleigh Z/p, counts, null trials
#
# Usage examples:
#   python psi_cone_plv_proof.py --limit 100000000 --segment 1000000 --decimate 25 \
#       --csv /mnt/data/psi_shell_stats_1e8_k25.csv --json /mnt/data/psi_summary_1e8_k25.json --null-trials 3
#
# Notes:
#  - Constant memory: O(#shells). Angles not stored.
#  - Decimation is performed *within each shell* to preserve phase statistics.
#  - Null test simulates uniform phases per shell (same counts) to show PLV collapse.
#
import math
import argparse
import json
from collections import defaultdict
from random import random

PHI = (1.0 + 5.0 ** 0.5) / 2.0
LOGPHI = math.log(PHI)
TWOPI = 2.0 * math.pi

class Kahan:
    __slots__ = ("s", "c")
    def __init__(self):
        self.s = 0.0
        self.c = 0.0
    def add(self, x: float):
        y = x - self.c
        t = self.s + y
        self.c = (t - self.s) - y
        self.s = t
    def value(self) -> float:
        return self.s

class Kahan2:
    __slots__ = ("r", "i")
    def __init__(self):
        self.r = Kahan()
        self.i = Kahan()
    def add(self, cr: float, ci: float):
        self.r.add(cr); self.i.add(ci)
    def mag(self) -> float:
        return (self.r.s * self.r.s + self.i.s * self.i.s) ** 0.5
    def angle(self) -> float:
        return math.atan2(self.i.s, self.r.s)
    def tuple(self):
        return (self.r.s, self.i.s)

def segmented_sieve(limit: int, segment_size: int):
    """Yield primes < limit using a segmented sieve. Memory O(segment_size)."""
    if limit < 2:
        return
    sqrt = int(math.isqrt(limit)) + 1
    is_prime_small = bytearray(b"\x01") * (sqrt + 1)
    is_prime_small[:2] = b"\x00\x00"
    for p in range(2, int(math.isqrt(sqrt)) + 1):
        if is_prime_small[p]:
            step = p
            start = p * p
            is_prime_small[start: sqrt + 1: step] = b"\x00" * (((sqrt - start) // step) + 1)
    base_primes = [i for i in range(2, sqrt + 1) if is_prime_small[i]]

    low = 2
    if segment_size < 32768:
        segment_size = 32768
    while low < limit:
        high = min(low + segment_size, limit)
        mark = bytearray(b"\x01") * (high - low)
        for p in base_primes:
            start = ((low + p - 1) // p) * p
            if start < p * p:
                start = p * p
            if start >= high:
                continue
            mark[start - low: high - low: p] = b"\x00" * (((high - start - 1) // p) + 1)
        for n in range(low, high):
            if n >= 2 and mark[n - low]:
                yield n
        low = high

def rayleigh_stats(N: int, S_mag: float):
    """Return Rayleigh Z and approximate p-value for non-uniformity on the circle.
    Rbar = |S|/N, Z = N * Rbar^2. p-value uses a common series approximation.
    """
    if N <= 0:
        return float('nan'), float('nan')
    Rbar = S_mag / max(1, N)
    Z = N * (Rbar ** 2)
    # p ≈ exp(-Z) * (1 + (2Z - Z^2)/(4N) - (24Z - 132Z^2 + 76Z^3 - 9Z^4)/(288N^2))
    # This is adequate for our reporting; for huge N the leading term dominates.
    if N > 0:
        term1 = math.exp(-Z)
        term2 = 1.0 + (2.0*Z - Z*Z) / (4.0*N)
        term3 = (24.0*Z - 132.0*Z*Z + 76.0*(Z**3) - 9.0*(Z**4)) / (288.0*(N**2))
        p = term1 * (term2 - term3)
        if p < 0.0: p = 0.0
    else:
        p = float('nan')
    return Z, p

def monte_carlo_null(shell_stats, trials: int = 3, seed: int | None = None):
    """Within-shell permutation null via simulated uniform phases for each shell's kept count.
    Returns list of dicts with PLV_global, mu_global_deg, PLV_shell_weighted, mu_sw_deg.
    """
    if seed is not None:
        import random as _r; _r.seed(seed)
    results = []
    for _ in range(max(0, trials)):
        # global
        Sg = Kahan2()
        Ng = 0
        # shell-weighted
        U = Kahan2()
        shells_nonempty = 0
        for z, (sumr, sumi, kept, total, kdec) in shell_stats.items():
            if kept <= 0:
                continue
            # simulate kept uniform angles
            Sz = Kahan2()
            for _n in range(kept):
                theta = TWOPI * random()
                Sz.add(math.cos(theta), math.sin(theta))
                # global
                Sg.add(math.cos(theta), math.sin(theta))
                Ng += 1
            mag = Sz.mag()
            if mag > 0.0:
                U.add(Sz.r.s / mag, Sz.i.s / mag)
            shells_nonempty += 1
        # compute metrics
        PLV_global = (Sg.mag() / Ng) if Ng > 0 else float('nan')
        mu_global_deg = (Sg.angle() * 180.0 / math.pi) % 360.0
        PLV_sw = (U.mag() / shells_nonempty) if shells_nonempty > 0 else float('nan')
        mu_sw_deg = (U.angle() * 180.0 / math.pi) % 360.0
        results.append({
            "PLV_global": PLV_global,
            "mu_global_deg": mu_global_deg,
            "PLV_shell_weighted": PLV_sw,
            "mu_shell_weighted_deg": mu_sw_deg,
        })
    return results

def run(limit: int, segment: int, decimate: int, csv_path: str | None, json_path: str | None, null_trials: int):
    # Global sums
    Sg = Kahan2()
    Ng = 0

    # Per-shell stats: z -> (sum_cos, sum_sin, N_kept, N_total, decimate_k)
    shells = {}

    prev_p = None
    ln_prev = None

    for p in segmented_sieve(limit, segment):
        if prev_p is None:
            ln_p = math.log(p)
        else:
            ln_p = ln_prev + math.log1p((p - prev_p) / prev_p)
        x = ln_p / LOGPHI
        z = math.floor(x)
        f = x - z
        theta = TWOPI * f
        c = math.cos(theta); s = math.sin(theta)

        entry = shells.get(z)
        if entry is None:
            entry = [0.0, 0.0, 0, 0, decimate]  # sumr, sumi, kept, total, kdec
            shells[z] = entry
        entry[3] += 1  # total

        keep = (decimate <= 1) or (entry[3] % decimate == 0)
        if keep:
            entry[0] += c
            entry[1] += s
            entry[2] += 1
            Sg.add(c, s)
            Ng += 1

        prev_p = p
        ln_prev = ln_p

    # Global stats
    mu_global_deg = (Sg.angle() * 180.0 / math.pi) % 360.0
    PLV_global = (Sg.mag() / Ng) if Ng > 0 else float('nan')
    Z, pval = rayleigh_stats(Ng, Sg.mag())

    # Shell-weighted and mean-shell PLV
    U = Kahan2()
    shells_nonempty = 0
    plv_sum = 0.0
    plv_count = 0
    for z, (sumr, sumi, kept, total, kdec) in shells.items():
        if kept > 0:
            mag = (sumr*sumr + sumi*sumi) ** 0.5
            plv = mag / kept
            plv_sum += plv; plv_count += 1
            if mag > 0.0:
                U.add(sumr / mag, sumi / mag)
            shells_nonempty += 1

    PLV_shell_mean = (plv_sum / plv_count) if plv_count > 0 else float('nan')
    PLV_shell_weighted = (U.mag() / shells_nonempty) if shells_nonempty > 0 else float('nan')
    mu_sw_deg = (U.angle() * 180.0 / math.pi) % 360.0

    # Null simulations
    null_results = monte_carlo_null(shells, trials=null_trials)

    # Write CSV
    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("shell_z,sum_cos,sum_sin,N_kept,N_total,decimate_k\n")
            for z in sorted(shells.keys()):
                sumr, sumi, kept, total, kdec = shells[z]
                f.write(f"{z},{sumr},{sumi},{kept},{total},{kdec}\n")

    # JSON summary
    summary = {
        "limit": limit,
        "segment": segment,
        "decimate": decimate,
        "N_kept": Ng,
        "shells_nonempty": shells_nonempty,
        "mu_global_deg": mu_global_deg,
        "PLV_global": PLV_global,
        "rayleigh_Z": Z,
        "rayleigh_p": pval,
        "mu_shell_weighted_deg": mu_sw_deg,
        "PLV_shell_weighted": PLV_shell_weighted,
        "PLV_shell_mean": PLV_shell_mean,
        "null_trials": null_trials,
        "null_results": null_results,
        "mapping": "theta = 2*pi * frac( ln(p)/ln(phi) ), phi=(1+sqrt(5))/2, no mod-180, origin at frac=0",
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Print human-readable summary
    print("=== ψ-Cone Streaming Proof Summary ===")
    for k in ["limit","segment","decimate","N_kept","shells_nonempty",
              "mu_global_deg","PLV_global","rayleigh_Z","rayleigh_p",
              "mu_shell_weighted_deg","PLV_shell_weighted","PLV_shell_mean"]:
        print(f"{k}: {summary[k]}")
    if null_trials > 0:
        print(f"null_trials: {null_trials}")
        for i, res in enumerate(null_results, 1):
            print(f"  null[{i}] PLV_global={res['PLV_global']:.6f}, PLV_shell_weighted={res['PLV_shell_weighted']:.6f}")

def main():
    ap = argparse.ArgumentParser(description="ψ-Cone streaming with per-shell CSV, Rayleigh test, and permutation null.")
    ap.add_argument("--limit", type=int, default=100_000_000, help="Upper bound for primes (exclusive).")
    ap.add_argument("--segment", type=int, default=1_000_000, help="Segment size for segmented sieve.")
    ap.add_argument("--decimate", type=int, default=25, help="Keep every k-th prime within each shell (k>=1).")
    ap.add_argument("--csv", type=str, default="/mnt/data/psi_shell_stats.csv", help="Per-shell CSV output path.")
    ap.add_argument("--json", type=str, default="/mnt/data/psi_summary.json", help="JSON summary output path.")
    ap.add_argument("--null-trials", type=int, default=2, help="Number of within-shell null simulations to run.")
    args = ap.parse_args()

    run(args.limit, args.segment, max(1, args.decimate), args.csv, args.json, max(0, args.null_trials))

if __name__ == "__main__":
    main()
