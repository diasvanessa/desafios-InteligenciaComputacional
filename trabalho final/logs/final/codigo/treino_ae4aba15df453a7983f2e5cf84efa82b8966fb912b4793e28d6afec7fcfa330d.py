"""D3/TMDB: preparação temporal, MLP e treino instrumentado em CPU.

Execute --help. Os CSVs brutos nunca são alterados. Seleção usa somente validação.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent / "04_resultados" / "rodada_inicial"
DATA = CODE.parent.parent / "tmdb"
OPTIMIZERS = ("sgd", "nesterov", "adam", "adamw", "muon")
SCHEDULES = ("constant", "cosine", "warmup_cosine", "step")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data_dir=DATA, out=ROOT):
    """Fit de vocabulários e escala EXCLUSIVAMENTE nos filmes de treino."""
    out = Path(out)
    dest = out / "dados"
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(data_dir).glob("tmdb_movies_*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV TMDB em {data_dir}")
    manifest = [{"file": f.name, "sha256": sha(f)} for f in files]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    frames = [pd.read_csv(f).assign(source=f.name, source_row=lambda x: np.arange(len(x))) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    audit = {"raw_rows": len(raw), "unique_ids": int(raw.id.nunique()),
             "raw_columns": list(raw.columns[:-2]), "raw_missing": raw.isna().sum().astype(int).to_dict(),
             "raw_zero": {k: int((raw[k] == 0).sum()) for k in ["vote_count", "budget", "runtime", "revenue"]},
             "data_sha256": digest, "files": manifest,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "dedup_rule": "maior vote_count; empate: nome do CSV ascendente, linha ascendente",
             "target_disagreement_duplicate_ids": int((raw.groupby("id").vote_average.nunique() > 1).sum())}
    assert raw.id.notna().all()
    frame = raw.sort_values(["id", "vote_count", "source", "source_row"],
                            ascending=[True, False, True, True], kind="stable").drop_duplicates("id").copy()
    audit["duplicate_rows_removed"] = len(raw) - len(frame)
    frame["date"] = pd.to_datetime(frame.release_date, errors="coerce")
    # Filtros sequenciais: as contagens removidas são disjuntas e somam corretamente.
    audit["filters"] = []
    for name, mask in [
        ("estreia entre 1920-01-01 e 2025-12-31", frame.date.between("1920-01-01", "2025-12-31")),
        ("pelo menos 10 votos", frame.vote_count.ge(10)),
        ("nota observada entre 0 e 10", frame.vote_average.between(0, 10)),
    ]:
        before = len(frame)
        frame = frame.loc[mask.reindex(frame.index)].copy()
        audit["filters"].append({"rule": name, "removed": before-len(frame), "remaining": len(frame)})
    frame = frame.sort_values(["date", "id"]).reset_index(drop=True)
    frame["split"] = np.where(frame.date.dt.year.le(2020), "train",
                              np.where(frame.date.dt.year.le(2022), "val", "test"))
    assert frame.id.is_unique
    frame["genres"] = frame.genre_ids.map(lambda s: sorted(set(int(g) for g in ast.literal_eval(s))))
    tr = frame[frame.split == "train"]
    genres = sorted({g for gs in tr.genres for g in gs})
    counts = tr.original_language.value_counts()
    languages = sorted(counts[counts >= 100].index.tolist())
    year_mean, year_std = float(tr.date.dt.year.mean()), float(tr.date.dt.year.std(ddof=0))
    target_mean = float(tr.vote_average.mean())
    feature_names = ["year_z", "month_sin", "month_cos"] + [f"genre_{g}" for g in genres]
    feature_names += ["genre_missing", "genre_unknown"] + [f"lang_{x}" for x in languages] + ["lang_other"]
    prep = {"genres": genres, "languages_min100_train": languages, "year_mean": year_mean,
            "year_std": year_std, "target_mean": target_mean, "train_median": float(tr.vote_average.median()),
            "feature_names": feature_names, "fit_split": "train", "missing": "gênero vazio: indicador; idioma raro/ausente: other; data inválida: excluída"}
    arrays = {}
    audit["splits"] = {}
    for split in ["train", "val", "test"]:
        f = frame[frame.split == split]
        month = f.date.dt.month.to_numpy()
        x = np.column_stack([(f.date.dt.year.to_numpy()-year_mean)/year_std,
                             np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)] +
                            [[float(g in gs) for gs in f.genres] for g in genres] +
                            [[float(not gs) for gs in f.genres], [float(any(g not in genres for g in gs)) for gs in f.genres]] +
                            [(f.original_language == lang).to_numpy().astype(float) for lang in languages] +
                            [(~f.original_language.isin(languages)).to_numpy().astype(float)]).astype("float32")
        y = f.vote_average.to_numpy(dtype="float32")
        assert np.isfinite(x).all() and np.isfinite(y).all()
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x, y
        arrays[f"id_{split}"] = f.id.to_numpy(dtype="int64")
        audit["splits"][split] = {"n": len(f), "date_min": str(f.date.min().date()), "date_max": str(f.date.max().date()),
                                 "unknown_languages": int((~f.original_language.isin(languages)).sum())}
        if split != "test":
            audit["splits"][split]["target"] = f.vote_average.describe().to_dict()
    audit["eligible"] = len(frame)
    audit["coverage_unique"] = len(frame)/len(raw.drop_duplicates("id"))
    audit["n_features"] = len(feature_names)
    audit["excluded_inputs"] = ["vote_average", "vote_count", "revenue", "popularity", "id", "imdb_id", "title", "original_title", "budget", "runtime", "production_companies", "director", "cast"]
    frame[["id", "title", "release_date", "split", "source", "source_row", "vote_count"]].to_csv(dest / "particoes.csv", index=False)
    np.savez_compressed(dest / "dataset.npz", **arrays)
    prep["dataset_npz_sha256"] = sha(dest / "dataset.npz")
    write_json(dest / "preprocessamento.json", prep)
    write_json(dest / "auditoria.json", audit)
    print(json.dumps({"eligible": len(frame), "features": len(feature_names), "splits": {k: v["n"] for k,v in audit["splits"].items()}}), flush=True)
    return audit


def load_data(out=ROOT, test=False):
    folder = Path(out) / "dados"
    prep = json.loads((folder / "preprocessamento.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "auditoria.json").read_text(encoding="utf-8"))
    if sha(folder / "dataset.npz") != prep["dataset_npz_sha256"]:
        raise ValueError("dataset.npz foi alterado")
    with np.load(folder / "dataset.npz") as data:
        splits = ["test"] if test else ["train", "val"]
        tensors = {f"{k}_{s}": torch.from_numpy(data[f"{k}_{s}"].copy()) for s in splits for k in ["x", "y"]}
        ids = {s: data[f"id_{s}"].copy() for s in splits}
    return tensors, prep, audit, ids


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(n_features, 32), nn.Linear(32, 32), nn.Linear(32, 32)])
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


@dataclass
class Config:
    optimizer: str = "adam"
    lr: float = 0.001
    schedule: str = "constant"
    seed: int = 17
    steps: int = 2400
    batch_size: int = 128
    log_every: int = 20
    weight_decay: float = 0.01
    clip_norm: float = 5.0
    aux_lr: float = 0.001
    warmup_fraction: float = 0.1
    min_lr_factor: float = 0.01


def schedule_factor(step, cfg):
    """step é 1..T; o valor retornado será USADO neste passo, antes de opt.step."""
    progress = (step-1)/max(1, cfg.steps-1)
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "step":
        return 1.0 if progress < .5 else (.1 if progress < .75 else .01)
    if cfg.schedule == "warmup_cosine":
        warm = max(1, int(cfg.steps*cfg.warmup_fraction))
        if step <= warm:
            return step/warm
        progress = (step-warm)/max(1, cfg.steps-warm)
    return cfg.min_lr_factor + (1-cfg.min_lr_factor)*0.5*(1+math.cos(math.pi*progress))


def make_optimizers(model, cfg):
    def groups(named):
        named = list(named)
        return [{"params": [p for n,p in named if p.ndim == 2], "weight_decay": cfg.weight_decay},
                {"params": [p for n,p in named if p.ndim != 2], "weight_decay": 0.0}]
    if cfg.optimizer == "muon":
        matrices = [p for n,p in model.named_parameters() if n.startswith("hidden.") and p.ndim == 2]
        muon_ids = {id(p) for p in matrices}
        auxiliary = [(n,p) for n,p in model.named_parameters() if id(p) not in muon_ids]
        assert muon_ids.isdisjoint({id(p) for _,p in auxiliary})
        assert len(muon_ids) + len(auxiliary) == len(list(model.parameters()))
        return [torch.optim.Muon(matrices, lr=cfg.lr, momentum=.95, nesterov=True,
                                 weight_decay=cfg.weight_decay, ns_steps=5, adjust_lr_fn="original"),
                torch.optim.AdamW(groups(auxiliary), lr=cfg.aux_lr)], [cfg.lr, cfg.aux_lr]
    params = groups(model.named_parameters())
    if cfg.optimizer in ["sgd", "nesterov"]:
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=.9, nesterov=cfg.optimizer == "nesterov")
    else:
        opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg.optimizer](params, lr=cfg.lr)
    return [opt], [cfg.lr]


@torch.no_grad()
def metrics(model, x, y, mean):
    residual = model(x) + mean - y
    return {"mse": residual.square().mean().item(), "mae": residual.abs().mean().item(),
            "rmse": residual.square().mean().sqrt().item()}


def norm(items):
    # Float64 para medir também sinais grandes, sem overflow prematuro do logger.
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in items))


def append_summary(out, result):
    flat = {k:v for k,v in result.items() if k != "config"}
    flat.update(result["config"])
    dest = Path(out) / "experimentos.csv"
    with dest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(flat)


def train(cfg, run_id, phase, hypothesis, out=ROOT, dataset=None):
    assert cfg.optimizer in OPTIMIZERS and cfg.schedule in SCHEDULES
    assert cfg.lr > 0 and cfg.aux_lr > 0 and cfg.steps > 0 and cfg.batch_size > 0
    assert 10 <= cfg.log_every <= 50
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
    opts, base_lrs = make_optimizers(model, cfg)
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    generator = torch.Generator().manual_seed(cfg.seed + 10000)
    # Épocas embaralhadas concatenadas: todo passo tem exatamente batch_size exemplos.
    order = torch.cat([torch.randperm(len(x), generator=generator) for _ in range(math.ceil(cfg.steps*cfg.batch_size/len(x)))])[:cfg.steps*cfg.batch_size]
    batch_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
    probe = torch.randperm(len(x), generator=torch.Generator().manual_seed(90210))[:2048]
    source_hash = sha(__file__)
    snapshot = out / "logs" / "codigo" / f"treino_{source_hash}.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(__file__, snapshot)
    command = f'python "{CODE / "reproduzir.py"}" --config "{logdir.resolve() / "config.json"}" --dados "{out.resolve() / "dados"}"'
    metadata = {"config": asdict(cfg), "run_id": run_id, "phase": phase, "hypothesis": hypothesis,
                "source_sha256": source_hash, "data_sha256": audit["data_sha256"],
                "dataset_npz_sha256": prep["dataset_npz_sha256"], "initial_parameters_sha256": initial_hash,
                "batch_order_sha256": batch_hash, "torch": torch.__version__, "numpy": np.__version__,
                "python": sys.version, "platform": platform.platform(), "threads": 1, "device": "cpu",
                "created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                "metric_selection": "MAE da validação no último passo; sem early stopping ou escolha de checkpoint",
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "optimizer_groups": [{"class": type(o).__name__, "groups": [{k:v for k,v in g.items() if k!='params'} for g in o.param_groups]} for o in opts]}
    write_json(logdir / "config.json", metadata)
    groups = {"global": list(model.parameters())}
    groups.update({f"hidden{i+1}": list(layer.parameters()) for i,layer in enumerate(model.hidden)})
    groups["head"] = list(model.head.parameters())
    records = []
    status, completed, reason = "complete", 0, ""
    started = time.perf_counter()
    for step in range(1, cfg.steps+1):
        factor = schedule_factor(step, cfg)
        used_lrs = [base*factor for base in base_lrs]
        for opt, lr in zip(opts, used_lrs):
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
        idx = order[(step-1)*cfg.batch_size:step*cfg.batch_size]
        loss = (model(x[idx]) - (y[idx]-prep["target_mean"])).square().mean()
        if not torch.isfinite(loss):
            status, reason = "diverged", f"perda não finita antes do passo {step}"
            records.append({"step": step, "event": reason, "lr": used_lrs[0]})
            break
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            status, reason = "diverged", f"gradiente não finito antes do passo {step}"
            records.append({"step": step, "event": reason, "train_batch_mse": loss.item(), "lr": used_lrs[0]})
            break
        logging = step == 1 or step % cfg.log_every == 0 or step == cfg.steps
        if logging:
            before = {id(p): p.detach().clone() for p in model.parameters()}
            rec = {"step": step, "train_batch_mse": loss.item(), "lr": used_lrs[0],
                   "lr_aux": used_lrs[-1], "lr_factor": factor}
            for name, params in groups.items():
                rec[f"grad_{name}"] = norm([p.grad for p in params if p.grad is not None])
                rec[f"theta_{name}"] = norm(params)
        if cfg.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        for opt in opts:
            opt.step()
        completed = step
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            status, reason = "diverged", f"parâmetro não finito depois do passo {step}"
            records.append({"step": step, "event": reason, "lr": used_lrs[0]})
            break
        if logging:
            for name, params in groups.items():
                delta = norm([p.detach()-before[id(p)] for p in params])
                rec[f"update_{name}"] = delta
                rec[f"rho_{name}"] = delta/max(rec[f"theta_{name}"], 1e-12)
            rec["clipped"] = cfg.clip_norm > 0 and rec["grad_global"] > cfg.clip_norm
            for name, value in metrics(model, xv, yv, prep["target_mean"]).items():
                rec[f"val_{name}"] = value
            rec["train_probe_mse"] = metrics(model, x[probe], y[probe], prep["target_mean"])["mse"]
            records.append(rec)
    elapsed = time.perf_counter()-started
    pd.DataFrame(records).to_csv(logdir / "curvas.csv", index=False)
    final = metrics(model, xv, yv, prep["target_mean"]) if status == "complete" else {}
    if any(not math.isfinite(v) for v in final.values()):
        status, reason, final = "diverged", "métrica final não finita", {}
    if status == "complete":
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "n_features": x.shape[1],
                    "target_mean": prep["target_mean"], "source_sha256": source_hash,
                    "data_sha256": audit["data_sha256"]}, logdir / "modelo.pt")
    result = {"run_id": run_id, "phase": phase, "hypothesis": hypothesis, "status": status,
              "diverged": status == "diverged", "reason": reason, "steps_completed": completed,
              "val_mae": final.get("mae"), "val_rmse": final.get("rmse"), "val_mse": final.get("mse"),
              "seconds": elapsed, "train_n": len(x), "val_n": len(xv), "source_sha256": source_hash,
              "data_sha256": audit["data_sha256"], "initial_parameters_sha256": initial_hash,
              "batch_order_sha256": batch_hash, "command": command, "config": asdict(cfg)}
    write_json(logdir / "resultado.json", result)
    append_summary(out, result)
    print(f'{run_id}: {status}, MAE={final.get("mae")}, {elapsed:.1f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--schedule", choices=SCHEDULES, default="constant")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--aux-lr", type=float, default=.001)
    parser.add_argument("--run-id")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.data_dir, args.out)
        return
    if args.replay:
        meta = json.loads(args.replay.read_text(encoding="utf-8"))
        if meta["source_sha256"] != sha(__file__):
            raise ValueError("Código mudou: use o snapshot original em logs/codigo para reproduzir")
        cfg = Config(**meta["config"])
        audit = json.loads((args.out/"dados/auditoria.json").read_text(encoding="utf-8"))
        if audit["data_sha256"] != meta["data_sha256"]:
            raise ValueError("Dados diferentes da execução original")
    else:
        cfg = Config(**{k: getattr(args,k) for k in ["optimizer", "lr", "schedule", "seed", "steps", "batch_size", "log_every", "aux_lr"]})
    run_id = args.run_id or datetime.now().strftime("manual_%Y%m%d_%H%M%S")
    train(cfg, run_id, "reproduction" if args.replay else "manual", "Reprodução ou execução avulsa explícita", args.out)


if __name__ == "__main__":
    main()
