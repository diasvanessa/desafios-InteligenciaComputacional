"""Treina o MLP final a partir de base_treino.csv e grava medicoes_treino.csv."""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "base_treino.csv"
DEFAULT_LR = {"sgd_plain": 0.03162277660168379, "sgd": 0.03162277660168379,
              "nesterov": 0.03162277660168379, "adam": 0.01,
              "adamw": 0.1, "muon": 0.03162277660168379}
TARGET_MEAN = 6.186003439526361
OPTIMIZERS = ("sgd_plain", "sgd", "nesterov", "adam", "adamw", "muon")
SCHEDULES = ("constant", "cosine", "warmup_cosine", "step")


class MLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.hidden = nn.ModuleList(
            [nn.Linear(inputs, 32), nn.Linear(32, 32), nn.Linear(32, 32)]
        )
        self.head = nn.Linear(32, 1)
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.head(x).squeeze(-1)


def norm(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


def rate_factor(step, steps, schedule):
    progress = (step - 1) / max(1, steps - 1)
    if schedule == "constant":
        return 1.0
    if schedule == "step":
        return 1.0 if progress < 0.5 else (0.1 if progress < 0.75 else 0.01)
    if schedule == "cosine":
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    warmup = int(steps * 0.1)
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizers(model, name, lr, aux_lr):
    named = list(model.named_parameters())
    matrices = [p for n, p in named if n.startswith("hidden.") and p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [(n, p) for n, p in named if id(p) not in matrix_ids]

    def groups(parameters):
        parameters = list(parameters)
        return [
            {"params": [p for _, p in parameters if p.ndim == 2], "weight_decay": 0.01},
            {"params": [p for _, p in parameters if p.ndim != 2], "weight_decay": 0.0},
        ]

    if name == "muon":
        return [
            torch.optim.Muon(matrices, lr=lr, momentum=0.95, nesterov=True,
                             weight_decay=0.01, ns_steps=5, adjust_lr_fn="original"),
            torch.optim.AdamW(groups(auxiliary), lr=aux_lr),
        ], [lr, aux_lr]
    parameters = groups(named)
    if name in ("sgd", "sgd_plain", "nesterov"):
        opt = torch.optim.SGD(parameters, lr=lr,
                               momentum=0.0 if name == "sgd_plain" else 0.9,
                               nesterov=name == "nesterov")
    else:
        cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        opt = cls(parameters, lr=lr)
    return [opt], [lr]


def load_dataset():
    frame = pd.read_csv(DATA)
    metadata = {"id", "split", "vote_average", "duracao_minutos"}
    features = [column for column in frame if column not in metadata]
    if not {"train", "val"}.issubset(set(frame["split"])):
        raise ValueError("A base precisa conter as partições train e val")
    if (frame["duracao_minutos"] <= 0).any():
        raise ValueError("A base contém duração ausente ou não positiva")
    if frame["id"].duplicated().any():
        raise ValueError("Há IDs repetidos entre as partições")
    if "vote_average" in features or "vote_count" in features:
        raise ValueError("Alvo ou contagem de votos não podem entrar como atributo")
    train = frame.loc[frame.split.eq("train")]
    val = frame.loc[frame.split.eq("val")]
    x_train = torch.tensor(train[features].to_numpy(dtype=np.float32))
    y_train = torch.tensor(train.vote_average.to_numpy(dtype=np.float32))
    x_val = torch.tensor(val[features].to_numpy(dtype=np.float32))
    y_val = torch.tensor(val.vote_average.to_numpy(dtype=np.float32))
    if not all(torch.isfinite(x).all() for x in (x_train, y_train, x_val, y_val)):
        raise ValueError("A base contém valores não finitos")
    return x_train, y_train, x_val, y_val, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="muon")
    parser.add_argument("--schedule", choices=SCHEDULES, default="warmup_cosine")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--aux-lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 10 <= args.log_every <= 50:
        parser.error("steps e batch-size devem ser positivos; log-every deve ficar entre 10 e 50")
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.optimizer]
    if lr <= 0 or args.aux_lr <= 0:
        parser.error("as taxas de aprendizado devem ser positivas")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    x, y, xv, yv, feature_names = load_dataset()
    target_mean = TARGET_MEAN
    model = MLP(x.shape[1])
    optimizers, base_lrs = make_optimizers(model, args.optimizer, lr, args.aux_lr)
    groups = {"global": list(model.parameters())}
    groups.update({f"hidden{i + 1}": list(layer.parameters()) for i, layer in enumerate(model.hidden)})
    groups["head"] = list(model.head.parameters())

    generator = torch.Generator().manual_seed(args.seed + 10000)
    count = math.ceil(args.steps * args.batch_size / len(x))
    order = torch.cat([torch.randperm(len(x), generator=generator) for _ in range(count)])[:args.steps * args.batch_size]
    rows = []
    status, diverged = "complete", False
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        factor = rate_factor(step, args.steps, args.schedule)
        used_lrs = [base * factor for base in base_lrs]
        for opt, used_lr in zip(optimizers, used_lrs):
            for group in opt.param_groups:
                group["lr"] = used_lr
            opt.zero_grad(set_to_none=True)
        indices = order[(step - 1) * args.batch_size:step * args.batch_size]
        loss = (model(x[indices]) - (y[indices] - target_mean)).square().mean()
        if not torch.isfinite(loss):
            status, diverged = f"non-finite loss at step {step}", True
            break
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            status, diverged = f"non-finite gradient at step {step}", True
            break

        logging = step == 1 or step % args.log_every == 0 or step == args.steps
        before = ({id(p): p.detach().clone() for p in model.parameters()} if logging else None)
        record = {"step": step, "train_batch_mse": loss.item(), "lr": used_lrs[0],
                  "lr_aux": used_lrs[-1], "lr_factor": factor}
        if logging:
            for name, params in groups.items():
                record[f"grad_{name}"] = norm([p.grad for p in params if p.grad is not None])
                record[f"theta_{name}"] = norm(params)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        for opt in optimizers:
            opt.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            status, diverged = f"non-finite parameter at step {step}", True
            break
        if logging:
            with torch.no_grad():
                for name, params in groups.items():
                    delta = norm([p - before[id(p)] for p in params])
                    record[f"update_{name}"] = delta
                    record[f"rho_{name}"] = delta / max(record[f"theta_{name}"], 1e-12)
                residual = model(xv) + target_mean - yv
                record["val_mse"] = residual.square().mean().item()
                record["val_mae"] = residual.abs().mean().item()
            rows.append(record)

    elapsed = time.perf_counter() - started
    run_id = f"{args.optimizer}_{args.schedule}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = ROOT / "resultados_treino"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"medicoes_{run_id}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    final_mae = rows[-1].get("val_mae") if rows else None
    summary = {"execucao": run_id, "otimizador": args.optimizer, "taxa": lr,
               "agenda": args.schedule, "semente": args.seed, "passos": args.steps,
               "lote": args.batch_size, "mae_validacao": final_mae,
               "divergiu": diverged, "situacao": status, "tempo_segundos": elapsed}
    summary_path = output_dir / "execucoes.csv"
    with summary_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Entradas: {len(feature_names)} | treino: {len(x)} | validação: {len(xv)}")
    print(f"Otimizador: {args.optimizer} | agenda: {args.schedule} | semente: {args.seed}")
    print(f"MAE final de validação: {final_mae if final_mae is not None else 'não disponível'}")
    print(f"Divergiu: {diverged} | tempo: {elapsed:.2f}s")
    print(f"Medições: {output} | resumo: {summary_path}")


if __name__ == "__main__":
    main()
