"""Referência adicional: SGD sem momento, com o mesmo MLP e orçamento do D3.

A busca usa apenas a validação. As execuções antigas do protocolo ficam intactas.
Uma configuração salva pode ser reproduzida com --replay; o snapshot deste arquivo
é chamado pelo reproduzir.py do pacote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_code():
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / "treino.py", parent / "03_codigo" / "treino.py"):
            if candidate.is_file():
                sys.path.insert(0, str(candidate.parent))
                return candidate
    raise FileNotFoundError("treino.py necessário para reproduzir o SGD")


CORE = find_code()
from treino import Config, MLP, append_summary, load_data, metrics, norm, sha, write_json  # noqa: E402

SEEDS = (17, 29, 43)
STEPS_SEARCH = 800
STEPS_FULL = 2400
DATA = Path(__file__).resolve().parent.parent / "04_resultados/rodada_duracao_completa/protocolo_final/dados"
OUT = DATA.parent


def run(cfg: Config, run_id: str, phase: str, out: Path, dataset=None):
    """Treino idêntico nos controles, trocando somente o otimizador por SGD puro."""
    assert cfg.optimizer == "sgd_plain" and cfg.schedule == "constant"
    out = Path(out)
    logdir = out / "logs" / run_id
    logdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    tensors, prep, audit, _ = dataset or load_data(out)
    x, y = tensors["x_train"], tensors["y_train"]
    xv, yv = tensors["x_val"], tensors["y_val"]
    model = MLP(x.shape[1])
    named = list(model.named_parameters())
    groups = [
        {"params": [p for _, p in named if p.ndim == 2], "weight_decay": cfg.weight_decay},
        {"params": [p for _, p in named if p.ndim != 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.SGD(groups, lr=cfg.lr, momentum=0.0, nesterov=False)
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    generator = torch.Generator().manual_seed(cfg.seed + 10000)
    order = torch.cat([torch.randperm(len(x), generator=generator)
                       for _ in range(math.ceil(cfg.steps * cfg.batch_size / len(x)))])[:cfg.steps * cfg.batch_size]
    batch_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
    source_hash = sha(__file__)
    core_hash = sha(CORE)
    snapshot = out / "logs/codigo" / f"treino_{source_hash}.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(__file__, snapshot)
    command = (f'python "{Path(__file__).resolve()}" --replay "{(logdir / "config.json").resolve()}" '
               f'--out "{out.resolve()}" --run-id "{run_id}_replica"')
    metadata = {
        "config": asdict(cfg), "run_id": run_id, "phase": phase,
        "hypothesis": "SGD sem momento como referência básica; taxa escolhida só na validação",
        "source_sha256": source_hash, "core_sha256": core_hash,
        "data_sha256": audit["data_sha256"], "dataset_npz_sha256": prep["dataset_npz_sha256"],
        "initial_parameters_sha256": initial_hash, "batch_order_sha256": batch_hash,
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(), "threads": 1, "device": "cpu",
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "metric_selection": "MAE da validação no último passo; sem early stopping",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer_groups": [{"class": "SGD", "groups": [
            {k: v for k, v in group.items() if k != "params"} for group in opt.param_groups]}],
    }
    write_json(logdir / "config.json", metadata)
    records = []
    status, reason, completed = "complete", "", 0
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        idx = order[(step - 1) * cfg.batch_size:step * cfg.batch_size]
        loss = (model(x[idx]) - (y[idx] - prep["target_mean"])).square().mean()
        if not torch.isfinite(loss):
            status, reason = "diverged", f"perda não finita no passo {step}"
            break
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            status, reason = "diverged", f"gradiente não finito no passo {step}"
            break
        logging = step == 1 or step % cfg.log_every == 0 or step == cfg.steps
        if logging:
            before = [p.detach().clone() for p in model.parameters()]
            record = {"step": step, "train_batch_mse": loss.item(), "lr": cfg.lr,
                      "grad_global": norm([p.grad for p in model.parameters() if p.grad is not None]),
                      "theta_global": norm(model.parameters())}
        if cfg.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
        completed = step
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            status, reason = "diverged", f"parâmetro não finito no passo {step}"
            break
        if logging:
            record["update_global"] = norm([p.detach() - old for p, old in zip(model.parameters(), before)])
            record["rho_global"] = record["update_global"] / max(record["theta_global"], 1e-12)
            record["clipped"] = cfg.clip_norm > 0 and record["grad_global"] > cfg.clip_norm
            for name, value in metrics(model, xv, yv, prep["target_mean"]).items():
                record[f"val_{name}"] = value
            records.append(record)
    elapsed = time.perf_counter() - started
    pd.DataFrame(records).to_csv(logdir / "curvas.csv", index=False)
    final = metrics(model, xv, yv, prep["target_mean"]) if status == "complete" else {}
    if status == "complete":
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "n_features": x.shape[1],
                    "target_mean": prep["target_mean"], "source_sha256": source_hash,
                    "data_sha256": audit["data_sha256"]}, logdir / "modelo.pt")
    result = {
        "run_id": run_id, "phase": phase, "hypothesis": metadata["hypothesis"],
        "status": status, "diverged": status != "complete", "reason": reason,
        "steps_completed": completed, "val_mae": final.get("mae"),
        "val_rmse": final.get("rmse"), "val_mse": final.get("mse"),
        "seconds": elapsed, "train_n": len(x), "val_n": len(xv),
        "source_sha256": source_hash, "data_sha256": audit["data_sha256"],
        "initial_parameters_sha256": initial_hash, "batch_order_sha256": batch_hash,
        "command": command, "config": asdict(cfg),
    }
    write_json(logdir / "resultado.json", result)
    append_summary(out, result)
    print(f"{run_id}: {status}, MAE={final.get('mae')}, {elapsed:.1f}s", flush=True)
    return result


def experiment(out: Path):
    out = Path(out)
    dataset = load_data(out)
    base = Config(optimizer="sgd_plain", schedule="constant")
    rates = [0.001, 0.0031622776601683794, 0.01, 0.03162277660168379, 0.1]
    trials = []
    while True:
        for lr in rates:
            tag = f"{lr:.8g}".replace(".", "p")
            run_id = f"sgd_plain_lr_{tag}"
            file = out / "logs" / run_id / "resultado.json"
            result = json.loads(file.read_text(encoding="utf-8")) if file.exists() else run(
                replace(base, lr=lr, steps=STEPS_SEARCH), run_id, "sgd_plain_lr", out, dataset)
            if result["status"] == "complete":
                trials.append(result)
        best = min(trials, key=lambda item: item["val_mae"])
        lr = best["config"]["lr"]
        if lr == min(rates):
            rates.append(min(rates) / math.sqrt(10))
        elif lr == max(rates):
            rates.append(max(rates) * math.sqrt(10))
        else:
            break
        if len(rates) > 12:
            raise RuntimeError("SGD ainda tem melhor taxa na borda da busca")
    values = []
    for seed in SEEDS:
        run_id = f"sgd_plain_base_s{seed}"
        file = out / "logs" / run_id / "resultado.json"
        result = json.loads(file.read_text(encoding="utf-8")) if file.exists() else run(
            replace(base, lr=lr, seed=seed, steps=STEPS_FULL), run_id, "sgd_plain_base", out, dataset)
        assert result["status"] == "complete" and result["steps_completed"] == STEPS_FULL
        values.append(result["val_mae"])
    write_json(out / "sgd_plain_resumo.json", {
        "role": "referência básica adicional; executada depois da campanha original",
        "optimizer": "SGD", "momentum": 0.0, "nesterov": False, "schedule": "constant",
        "lr_selected": lr, "selection": "menor MAE após 800 passos, semente 17; ampliar grade se melhor na borda",
        "screen_steps": STEPS_SEARCH, "full_steps": STEPS_FULL, "seeds": SEEDS,
        "values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
        "runs": [f"sgd_plain_base_s{s}" for s in SEEDS],
        "data_sha256": dataset[2]["data_sha256"], "dataset_npz_sha256": dataset[1]["dataset_npz_sha256"],
        "source_sha256": sha(__file__), "core_sha256": sha(CORE),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--run-id", default="replica")
    args = parser.parse_args()
    if args.replay:
        meta = json.loads(args.replay.read_text(encoding="utf-8"))
        assert meta["source_sha256"] == sha(__file__)
        assert meta["core_sha256"] == sha(CORE)
        run(Config(**meta["config"]), args.run_id, meta["phase"], args.out)
    else:
        experiment(args.out)


if __name__ == "__main__":
    main()
