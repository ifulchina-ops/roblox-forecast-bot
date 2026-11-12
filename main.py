#!/usr/bin/env python3
"""Predict Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Latent-score diffusion Monte Carlo forecaster. Standard library only, no
network, deterministic (fixed seed).

Method
------
Each game has a latent score; the chart order is the order of scores.
Scores start at -rank (last read), drift by (shrunk, decaying) recent rank
velocity plus a small rank-band prior, and diffuse with rank-dependent
volatility estimated from the history (shrunk toward priors measured on the
public chart archive). Games drop out with a rank-dependent hourly hazard.
Phantom newcomers enter at the measured rate (~1/hour), climb at rates drawn
from the observed climber distribution, and compete in the ranking; the place
mass they take is left unnamed (their ids are unknown, so no forecast can
name them). The forecast is the time average (over the 72 scored hours) of
the per-hour place marginals, which is the log-score-optimal single forecast
for a window scored as the mean over its reads.

Naming rule: for each place, the named set maximises the expected log score
under the model: naming the top m (by p) gives
  sum_{i<=m} p_i ln p_i + (1 - s_m) ln((1 - s_m) / (N - m)),
with N the chart size; every m is tried and the best kept.

Reads /input/history.ndjson + /input/task.json (or $MAKEFUN_INPUT_DIR),
writes /output/forecast.json (or $MAKEFUN_OUTPUT_DIR). If the Monte Carlo
fails for any reason, falls back to the persistence baseline (a valid
forecast) rather than writing nothing.
"""
import json
import math
import os
import random
from datetime import datetime, timedelta

SEED = 20261112
RUNS = 3000
HOURS_DEFAULT = 72
PLACES_DEFAULT = 10

# --- priors measured on the public chart archive (see README) ---
# rank-band volatility: sd of rank change per sqrt(hour)
VOL_BANDS = [(1, 10, 1.22), (11, 30, 2.04), (31, 60, 3.18), (61, 10 ** 9, 3.66)]
# rank-band drift prior: mean rank change per hour (positive = falling)
DRIFT_BANDS = [(1, 10, 0.098), (11, 30, 0.137), (31, 60, 0.072), (61, 10 ** 9, -0.241)]
# rank-band hourly dropout hazard
HAZARD_BANDS = [(1, 10, 0.0005), (11, 30, 0.002), (31, 60, 0.004), (61, 10 ** 9, 0.010)]
ENTRY_RATE_PER_HOUR = 1.0

# --- tuned parameters (backtested on the practice window + history slices) ---
VOL_SCALE = 1.0
VEL_SHRINK = 0.30
VEL_HALFLIFE_H = 18.0
VEL_LOOKBACK_H = 72.0
DRIFT_PRIOR_WEIGHT = 0.5
HAZARD_SCALE = 1.0
PHANTOMS = 90
PHANTOM_BIRTH_UNTIL_H = 60.0
PHANTOM_CLIMB_MEDIAN = 1.0
PHANTOM_CLIMB_SIGMA = 0.55
PHANTOM_CLIMB_TAU_H = 50.0
MAX_SIM_GAMES = 60
MAX_ROWS_PER_PLACE = 100


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def band_value(bands, rank):
    for lo, hi, v in bands:
        if lo <= rank <= hi:
            return v
    return bands[-1][2]


def interp_band(bands, rank):
    pts = [((lo + min(hi, 120)) / 2.0, v) for lo, hi, v in bands]
    if rank <= pts[0][0]:
        return pts[0][1]
    if rank >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= rank <= x1:
            f = (rank - x0) / (x1 - x0)
            return y0 + f * (y1 - y0)
    return pts[-1][1]


def estimate_from_history(reads):
    """Estimate volatility/drift bands from the input history, shrunk toward
    the priors. Falls back to priors when the history is too thin."""
    times = [parse_ts(r["at"]) for r in reads]
    series = {}
    for i, r in enumerate(reads):
        for row in r["up_and_coming"]:
            series.setdefault(row["universe_id"], []).append((times[i], row["rank"]))
    vol_acc = {(1, 10): [], (11, 30): [], (31, 60): [], (61, 10 ** 9): []}
    drift_acc = {(1, 10): [], (11, 30): [], (31, 60): [], (61, 10 ** 9): []}
    for obs in series.values():
        for (t0, r0), (t1, r1) in zip(obs, obs[1:]):
            h = max(0.5, (t1 - t0).total_seconds() / 3600.0)
            for band in vol_acc:
                if band[0] <= r0 <= band[1]:
                    vol_acc[band].append((r1 - r0) / math.sqrt(h))
                    drift_acc[band].append((r1 - r0) / h)
                    break
    vol_bands, drift_bands = [], []
    for (lo, hi), prior in zip(vol_acc, [b[2] for b in VOL_BANDS]):
        vals = vol_acc[(lo, hi)]
        if len(vals) >= 30:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            est = math.sqrt(var)
            w = len(vals) / (len(vals) + 100.0)
            vol_bands.append((lo, hi, w * est + (1 - w) * prior))
        else:
            vol_bands.append((lo, hi, prior))
    for (lo, hi), prior in zip(drift_acc, [b[2] for b in DRIFT_BANDS]):
        vals = drift_acc[(lo, hi)]
        if len(vals) >= 30:
            est = sum(vals) / len(vals)
            w = len(vals) / (len(vals) + 200.0)
            drift_bands.append((lo, hi, w * est + (1 - w) * prior))
        else:
            drift_bands.append((lo, hi, prior))
    return {"vol": vol_bands, "drift": drift_bands}


def baseline_forecast(last_chart, places):
    """Persistence baseline (the kit starter): the chart does not move."""
    uids = [r["universe_id"] for r in sorted(last_chart, key=lambda r: r["rank"])]
    probs = []
    for place in range(1, places + 1):
        if place > len(uids):
            break
        probs.append({"universe_id": uids[place - 1], "place": place, "p": 0.5})
        if place == 1:
            if len(uids) > 1:
                probs.append({"universe_id": uids[1], "place": place, "p": 0.3})
        else:
            probs.append({"universe_id": uids[place - 2], "place": place, "p": 0.15})
            if place < len(uids):
                probs.append({"universe_id": uids[place], "place": place, "p": 0.15})
    return {"probabilities": probs, "top10": uids[:places]}


def build_forecast(reads, task):
    rng = random.Random(SEED)
    hours = task.get("window", {}).get("slots", HOURS_DEFAULT)
    places = task.get("places", PLACES_DEFAULT)
    est = estimate_from_history(reads)

    times = [parse_ts(r["at"]) for r in reads]
    t0 = times[-1]
    last_chart = sorted(reads[-1]["up_and_coming"], key=lambda r: r["rank"])
    chart_size = len(last_chart)

    series = {}
    for i, r in enumerate(reads):
        for row in r["up_and_coming"]:
            series.setdefault(row["universe_id"], []).append((times[i], row["rank"]))

    tau_v = VEL_HALFLIFE_H / math.log(2)
    lookback = timedelta(hours=VEL_LOOKBACK_H)

    def velocity(uid):
        obs = [(t, rk) for (t, rk) in series.get(uid, []) if t0 - t <= lookback]
        if len(obs) < 2:
            return 0.0
        num = 0.0
        den = 0.0
        for (ta, ra), (tb, rb) in zip(obs, obs[1:]):
            h = max(0.5, (tb - ta).total_seconds() / 3600.0)
            v = (ra - rb) / h  # positive = climbing
            w = math.exp(-(t0 - tb).total_seconds() / 3600.0 / tau_v) * h
            num += w * v
            den += w
        return num / den if den > 0 else 0.0

    # candidate known games: top MAX_SIM_GAMES by rank, fast climbers promoted
    cands = []
    for row in last_chart:
        uid = row["universe_id"]
        rk = row["rank"]
        v = velocity(uid)
        eff = rk - 8.0 * max(0.0, v)
        cands.append((eff, uid, rk, v))
    cands.sort()
    sim_games = cands[:MAX_SIM_GAMES]

    known = []
    for eff, uid, rk, v in sim_games:
        known.append({
            "uid": uid, "score0": -float(rk), "rank0": rk, "vel": v,
            "prior_drift_rank": interp_band(est["drift"], rk),
            "sigma": interp_band(est["vol"], rk) * VOL_SCALE,
            "hazard": band_value(HAZARD_BANDS, rk) * HAZARD_SCALE,
        })

    phantoms = []
    for _ in range(PHANTOMS):
        birth = rng.uniform(0, PHANTOM_BIRTH_UNTIL_H)
        entry_rank = min(99, max(70, rng.gauss(85, 8)))
        climb = math.exp(rng.gauss(math.log(PHANTOM_CLIMB_MEDIAN), PHANTOM_CLIMB_SIGMA))
        phantoms.append({
            "birth": birth, "score0": -float(entry_rank), "climb": climb,
            "sigma": interp_band(est["vol"], 90) * VOL_SCALE,
            "hazard": band_value(HAZARD_BANDS, 90) * HAZARD_SCALE,
        })

    counts = [dict() for _ in range(places)]
    total_slots = RUNS * hours
    gauss = rng.gauss
    vel_cum = [tau_v * (1.0 - math.exp(-h / tau_v)) for h in range(hours + 1)]
    tau_c = PHANTOM_CLIMB_TAU_H

    for run in range(RUNS):
        alive_until = []
        for g in known:
            hz = g["hazard"]
            alive_until.append(hours + 1 if hz <= 0 else min(hours + 1, int(rng.expovariate(hz)) + 1))
        ph_alive_until = []
        for p in phantoms:
            hz = p["hazard"]
            ph_alive_until.append(hours + 1 if hz <= 0 else min(hours + 1, int(rng.expovariate(hz)) + 1))
        for h in range(1, hours + 1):
            sq = math.sqrt(h)
            vc = vel_cum[h]
            scored = []
            for gi, g in enumerate(known):
                if h > alive_until[gi]:
                    continue
                mean = (g["score0"] + g["vel"] * VEL_SHRINK * vc
                        - g["prior_drift_rank"] * DRIFT_PRIOR_WEIGHT * h)
                scored.append((mean + g["sigma"] * sq * gauss(0, 1), g["uid"]))
            for pi, p in enumerate(phantoms):
                if h < p["birth"] or h > ph_alive_until[pi] + p["birth"]:
                    continue
                age = h - p["birth"]
                climb_cum = p["climb"] * tau_c * (1.0 - math.exp(-age / tau_c))
                mean = p["score0"] + climb_cum
                scored.append((mean + p["sigma"] * math.sqrt(age + 1) * gauss(0, 1), None))
            if not scored:
                continue
            scored.sort(key=lambda t: -t[0])
            for k in range(min(places, len(scored))):
                uid = scored[k][1]
                if uid is not None:
                    d = counts[k]
                    d[uid] = d.get(uid, 0) + 1

    probabilities = []
    for k in range(places):
        cands = sorted(((uid, c / total_slots) for uid, c in counts[k].items()),
                       key=lambda t: -t[1])
        best_m, best_val = 0, None
        s = 0.0
        named_log = 0.0
        n_un = max(1, chart_size)
        best_val = math.log(1.0 / n_un)
        for m, (uid, p) in enumerate(cands, 1):
            if m > MAX_ROWS_PER_PLACE:
                break
            s += p
            if p > 0:
                named_log += p * math.log(p)
            n_un = max(1, chart_size - m)
            un = max(0.0, 1.0 - s)
            val = named_log + (un * math.log(un / n_un) if un > 0 else 0.0)
            if val > best_val:
                best_val = val
                best_m = m
        for uid, p in cands[:best_m]:
            probabilities.append({"universe_id": uid, "place": k + 1, "p": round(p, 6)})

    tot = {}
    exp_place = {}
    for k in range(places):
        for uid, c in counts[k].items():
            tot[uid] = tot.get(uid, 0) + c
            exp_place[uid] = exp_place.get(uid, 0.0) + c * (k + 1)
    ranked = sorted(tot.items(), key=lambda kv: -kv[1])
    top10 = [uid for uid, _ in ranked[:places]]
    top10.sort(key=lambda u: exp_place[u] / tot[u])
    if len(top10) < places:
        for row in last_chart:
            if row["universe_id"] not in top10:
                top10.append(row["universe_id"])
            if len(top10) == places:
                break
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json")) as fh:
        task = json.load(fh)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    places = task.get("places", PLACES_DEFAULT)
    try:
        forecast = build_forecast(reads, task)
    except Exception:
        forecast = baseline_forecast(reads[-1]["up_and_coming"], places)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
