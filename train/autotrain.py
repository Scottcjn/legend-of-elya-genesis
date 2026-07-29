#!/usr/bin/env python3
"""
autotrain.py — unattended alternating trainer for both Elya brains.

Alternates: Genesis (ternary, 2L/64d) -> N64 (Q8, 4L/128d) -> repeat.
Each round samples a hyperparameter variation, trains, and KEEPS THE
RESULT ONLY IF held-out eval loss beats the current champion for that
target. Champions and a full history live in train/champions/.

Run it and walk away:
    nohup python3 autotrain.py > autotrain.log 2>&1 &
Stop it:
    touch train/STOP        (finishes the current round, then exits)

Nothing is overwritten in-place: a new champion is copied into
champions/<target>_best.{bin,pt,json} and only then promoted to the
live res/ blob the ROM builds against.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAMP = HERE / "champions"
CHAMP.mkdir(exist_ok=True)
STOP = HERE / "STOP"
LEDGER = CHAMP / "ledger.jsonl"
RES = HERE.parent / "res"

# (lr, steps, emb_pin_weight, qa_repeat) — varied per round
GENESIS_GRID = [
    (5e-4, 25000, 1e-2, 1200),
    (3e-4, 40000, 1e-2, 1200),
    (5e-4, 40000, 3e-2, 1600),
    (2e-4, 60000, 1e-2, 1600),
    (8e-4, 25000, 3e-2,  800),
    (3e-4, 60000, 3e-3, 2000),
]

def champ_loss(target):
    f = CHAMP / f"{target}_best.json"
    if f.exists():
        return json.loads(f.read_text()).get("eval_loss", 1e9)
    return 1e9

def log(rec):
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[autotrain] {rec}", flush=True)

def run_genesis(rnd):
    lr, steps, pin, rep = GENESIS_GRID[rnd % len(GENESIS_GRID)]
    env = dict(os.environ,
               EG_LR=str(lr), EG_PIN=str(pin), EG_QA_REPEAT=str(rep))
    t0 = time.time()
    p = subprocess.run([sys.executable, str(HERE / "train_elya_genesis.py"),
                        str(steps)],
                       cwd=HERE, env=env, capture_output=True, text=True)
    out = p.stdout
    (HERE / "autotrain_last_genesis.log").write_text(out + p.stderr)
    ev = 1e9
    for line in out.splitlines():
        if line.startswith("Done. best="):
            try: ev = float(line.split("best=")[1].split()[0])
            except Exception: pass
    rec = {"t": int(time.time()), "target": "genesis", "round": rnd,
           "lr": lr, "steps": steps, "pin": pin, "qa_repeat": rep,
           "eval_loss": ev, "secs": int(time.time() - t0),
           "rc": p.returncode}
    if p.returncode == 0 and ev < champ_loss("genesis"):
        shutil.copy(RES / "elya_genesis.bin", CHAMP / "genesis_best.bin")
        shutil.copy(HERE / "elya_genesis.pt", CHAMP / "genesis_best.pt")
        shutil.copy(HERE / "test_vectors.json",
                    CHAMP / "genesis_best_vectors.json")
        (CHAMP / "genesis_best.json").write_text(json.dumps(rec, indent=2))
        rec["promoted"] = True
    else:
        # not a champion: restore the reigning blob so the ROM never
        # regresses just because a round finished
        if (CHAMP / "genesis_best.bin").exists():
            shutil.copy(CHAMP / "genesis_best.bin", RES / "elya_genesis.bin")
        rec["promoted"] = False
    log(rec)

def run_n64(rnd):
    script = HERE / "train_elya_n64.py"
    if not script.exists():
        log({"t": int(time.time()), "target": "n64", "round": rnd,
             "skipped": "train_elya_n64.py not present yet"})
        return
    steps = [30000, 45000, 60000][rnd % 3]
    t0 = time.time()
    p = subprocess.run([sys.executable, str(script), str(steps)],
                       cwd=HERE, capture_output=True, text=True)
    (HERE / "autotrain_last_n64.log").write_text(p.stdout + p.stderr)
    ev = 1e9
    for line in p.stdout.splitlines():
        if line.startswith("Done. best="):
            try: ev = float(line.split("best=")[1].split()[0])
            except Exception: pass
    rec = {"t": int(time.time()), "target": "n64", "round": rnd,
           "steps": steps, "eval_loss": ev,
           "secs": int(time.time() - t0), "rc": p.returncode}
    if p.returncode == 0 and ev < champ_loss("n64"):
        for src, dst in [("sophia_weights_gen.bin", "n64_best.bin"),
                         ("elya_n64.pt", "n64_best.pt")]:
            if (HERE / src).exists():
                shutil.copy(HERE / src, CHAMP / dst)
        (CHAMP / "n64_best.json").write_text(json.dumps(rec, indent=2))
        rec["promoted"] = True
    else:
        rec["promoted"] = False
    log(rec)

def main():
    rnd = 0
    log({"t": int(time.time()), "event": "autotrain start",
         "genesis_champ": champ_loss("genesis"), "n64_champ": champ_loss("n64")})
    while not STOP.exists():
        run_genesis(rnd)
        if STOP.exists():
            break
        run_n64(rnd)
        rnd += 1
    log({"t": int(time.time()), "event": "autotrain stopped", "rounds": rnd})

if __name__ == "__main__":
    main()
