"""Planted-effect test for analyze.py. Run: .venv/bin/python -m pytest tests -q"""
import json
import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analyze  # noqa: E402

CKPTS = ["s0", "s40", "s80", "s120"]
SEEDS = [0, 1]
RUNGS = ["sure", "wise", "rel_soft", "restate"]


def lerp(a, b, i, n=len(CKPTS)):
    return a + (b - a) * i / (n - 1)


# planted truth, (s0 value, s120 value), linear in between
AFFIRM_SAD = {"W": (0.6, 0.4), "O": (0.3, 0.35), "B": (0.3, 0.35)}
AFFIRM_NEU = {"W": (0.35, 0.30), "O": (0.2, 0.2), "B": (0.2, 0.2)}
SURE_FLIP = {"W": (0.5, 0.35), "O": (0.3, 0.3), "B": (0.3, 0.3)}
WARMTH = {"W": (4.2, 3.4), "O": (2.8, 2.8), "B": (2.7, 2.7)}
GSM = {"W": (0.72, 0.80), "O": (0.74, 0.80), "B": (0.75, 0.81)}
RESTATE = 0.05


def exact_flags(n, p, rnd):
    k = round(n * p)
    f = [True] * k + [False] * (n - k)
    rnd.shuffle(f)
    return f


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_eval(d, ev, rows, per_seed, judged):
    """per_seed: eval_lib layout {ev}_seed{N}[_judged].jsonl; else one {ev}[_judged].jsonl."""
    suf = "_judged" if judged else ""
    if not per_seed:
        write(os.path.join(d, f"{ev}{suf}.jsonl"), rows)
        return
    for s in SEEDS:
        sub = [r for r in rows if r["seed"] == s]
        write(os.path.join(d, f"{ev}_seed{s}{suf}.jsonl"), sub)
        if judged:   # pre-judge file sits next to the judged one and must be ignored
            write(os.path.join(d, f"{ev}_seed{s}.jsonl"),
                  [dict(r, label="ambiguous", warmth=None) for r in sub])


def make_raw(raw, rnd, per_seed=True):
    n_syco, n_lad, n_gsm, n_warm = 200, 150, 300, 50
    for arm in "WOB":
        for i, ck in enumerate(CKPTS):
            # local download layout per config.LOCAL_RAW is {arm}/{ckpt}; a pod tarball gives {arm}/results/{ckpt}
            d = os.path.join(raw, arm, ck) if per_seed else os.path.join(raw, arm, "results", ck)
            base = dict(arm=arm, ckpt=ck)
            # --- syco: 4% invalid/ambiguous per framing, exact affirm counts among the valid rest
            syco = []
            for fr, p in (("plain", None), ("neutral", AFFIRM_NEU[arm]), ("sad", AFFIRM_SAD[arm])):
                keys = [(s, it) for s in SEEDS for it in range(n_syco)]
                rnd.shuffle(keys)
                n_bad = round(0.04 * len(keys))
                valid = keys[n_bad:]
                rate = 0.15 if p is None else lerp(*p, i)
                aff = exact_flags(len(valid), rate, rnd)
                for j, (s, it) in enumerate(keys[:n_bad]):
                    syco.append(dict(base, seed=s, item_id=f"q{it}", framing=fr, label="invalid" if j % 2 else "ambiguous", n_tokens=100))
                for (s, it), a in zip(valid, aff):
                    syco.append(dict(base, seed=s, item_id=f"q{it}", framing=fr, label="affirms" if a else "corrects",
                                     n_tokens=120 + 40 * (arm == "W")))
            write_eval(d, "syco", syco, per_seed, True)
            # --- ladder: 3% m1 null; exact flip counts per rung among valid rows
            keys = [(s, it, o) for s in SEEDS for it in range(n_lad) for o in ("AB", "BA")]
            rnd.shuffle(keys)
            n_bad = round(0.03 * len(keys))
            valid = keys[n_bad:]
            m1 = {k: rnd.choice(["safe", "risky"]) for k in valid}
            flips = {rg: exact_flags(len(valid), RESTATE if rg == "restate" else lerp(*SURE_FLIP[arm], i) if rg == "sure" else 0.2, rnd)
                     for rg in RUNGS}
            lad = [dict(base, seed=s, item_id=f"a{it}", order=o, m1=None, rungs={rg: "safe" for rg in RUNGS}) for s, it, o in keys[:n_bad]]
            for j, k in enumerate(valid):
                other = "risky" if m1[k] == "safe" else "safe"
                lad.append(dict(base, seed=k[0], item_id=f"a{k[1]}", order=k[2], m1=m1[k],
                                rungs={rg: other if flips[rg][j] else m1[k] for rg in RUNGS}))
            write_eval(d, "ladder", lad, per_seed, False)
            # --- gsm8k
            keys = [(s, it) for s in SEEDS for it in range(n_gsm)]
            cor = exact_flags(len(keys), lerp(*GSM[arm], i), rnd)
            write_eval(d, "gsm8k", [dict(base, seed=s, item_id=f"g{it}", correct=c, strict=c, pred="1", n_tokens=200) for (s, it), c in zip(keys, cor)],
                       per_seed, False)
            # --- warmth: 2% null, integer scores with exact mean among valid
            keys = [(s, it) for s in SEEDS for it in range(n_warm)]
            rnd.shuffle(keys)
            valid = keys[2:]
            mu = lerp(*WARMTH[arm], i)
            lo = int(mu)
            hi_flags = exact_flags(len(valid), mu - lo, rnd)
            warm = [dict(base, seed=s, item_id=f"w{it}", warmth=None, n_tokens=300) for s, it in keys[:2]]
            warm += [dict(base, seed=s, item_id=f"w{it}", warmth=lo + int(h), n_tokens=300) for (s, it), h in zip(valid, hi_flags)]
            write_eval(d, "warmth", warm, per_seed, True)
        # --- grpo log: correct reward rises, length shrinks
        log = [{"step": t, "reward": 0.5 + 0.003 * t, "rewards/correct/mean": 0.45 + 0.003 * t + 0.02 * rnd.random(),
                "rewards/format/mean": 0.09, "completions/mean_length": 300 - t, "completions/clipped_ratio": 0.01,
                "frac_reward_zero_std": 0.2, "step_time": 30.0} for t in range(1, 121)]
        log.append({"train_runtime": 3600.0, "step": 120})   # TRL's final summary row must be ignored
        write(os.path.join(raw, arm, "grpo", "grpo_log.jsonl"), log)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("an")
    raw, out = str(tmp / "raw"), str(tmp / "out")
    make_raw(raw, random.Random(0))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "analyze.py"), "--raw", raw, "--out", out],
                       capture_output=True, text=True, cwd=ROOT)
    print(p.stdout[-4000:], p.stderr[-4000:])
    assert p.returncode == 0, p.stderr
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(out, "sanity.json")) as f:
        sanity = json.load(f)
    return dict(out=out, raw=raw, summary=summary, sanity=sanity, stdout=p.stdout)


def test_headline_affirm(run):
    h = run["summary"]["headline"]["affirm_sad"]
    assert h["ckpt_last"] == "s120"
    assert h["gap0"]["est"] == pytest.approx(0.30, abs=0.02)
    assert h["gap_last"]["est"] == pytest.approx(0.05, abs=0.02)
    assert h["ratio"]["est"] == pytest.approx(0.05 / 0.30, abs=0.08)
    assert h["gap0"]["lo"] > 0.15 and h["gap0"]["hi"] < 0.45      # CI brackets the planted gap
    assert h["delta"]["hi"] < 0                                    # convergence is detected
    assert not h["ratio"]["unstable"]


def test_headline_warmth_and_net_sure(run):
    hw = run["summary"]["headline"]["warmth"]
    assert hw["gap0"]["est"] == pytest.approx(1.4, abs=0.05)
    assert hw["gap_last"]["est"] == pytest.approx(0.6, abs=0.05)
    assert hw["ratio"]["est"] == pytest.approx(0.6 / 1.4, abs=0.05)
    hs = run["summary"]["headline"]["net_sure"]
    assert hs["gap0"]["est"] == pytest.approx(0.20, abs=0.02)
    assert hs["gap_last"]["est"] == pytest.approx(0.05, abs=0.02)


def test_cells(run):
    c = run["summary"]["cells"]
    assert c["W"]["s0"]["affirm_sad"]["est"] == pytest.approx(0.6, abs=0.01)
    assert c["O"]["s120"]["affirm_sad"]["est"] == pytest.approx(0.35, abs=0.01)
    assert c["B"]["s0"]["flip_restate"]["est"] == pytest.approx(RESTATE, abs=0.01)
    assert c["W"]["s0"]["syco_invalid"]["est"] == pytest.approx(0.04, abs=0.01)
    assert c["W"]["s0"]["warmth_invalid"]["est"] == pytest.approx(0.02, abs=0.005)
    assert set(c["W"]["s0"]["affirm_sad"]["by_seed"]) == {"0", "1"}
    rl = run["summary"]["rl_moved"]["W"]
    assert rl["correct_last"] > rl["correct_first"] and rl["len_last"] < rl["len_first"]
    assert run["summary"]["missing"] == []


def test_outputs_and_sanity(run):
    assert os.path.getsize(os.path.join(run["out"], "fig_main.png")) > 10_000
    table = open(os.path.join(run["out"], "table.md")).read()
    assert "| W | s0 |" in table and "affirm_sad" in table
    s = run["sanity"]
    for name in ("rl_moved", "persona_installed", "instrument_valid"):
        assert s[name]["pass"], (name, s[name]["details"])
    assert s["all_pass"]
    assert "SANITY overall: PASS" in run["stdout"]


def test_tolerates_missing_and_rft(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_raw(str(raw), random.Random(1), per_seed=False)   # single-file names under {arm}/results/
    os.remove(raw / "B" / "results" / "s120" / "ladder.jsonl")
    os.remove(raw / "O" / "grpo" / "grpo_log.jsonl")
    (raw / "W" / "results" / "s80").rename(raw / "W" / "results" / "r2")
    analyze.main(["--raw", str(raw), "--out", str(out), "--n-boot", "50"])
    summ = json.load(open(out / "summary.json"))
    assert summ["ckpts"] == ["s0", "r2", "s40", "s80", "s120"]
    assert "B/s120/ladder" in summ["missing"]
    assert not json.load(open(out / "sanity.json"))["rl_moved"]["pass"]
    assert (out / "fig_main.png").exists()


def test_ckpt_order():
    assert sorted(["s120", "r3", "s40", "r1", "s0", "s80", "r2"], key=analyze.ckpt_key) == \
        ["s0", "r1", "r2", "r3", "s40", "s80", "s120"]
