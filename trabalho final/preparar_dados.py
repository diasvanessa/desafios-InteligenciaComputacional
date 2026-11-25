"""Gera a base de regressão com atributos de pré-lançamento do TMDB."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 42
SOURCE_COLUMNS = [
    "id", "release_date", "vote_average", "vote_count", "genre_ids",
    "original_language", "runtime", "budget", "director", "cast",
    "production_companies",
]
GENRES = (12, 14, 16, 18, 27, 28, 35, 36, 37, 53, 80, 99, 878, 9648, 10402,
          10749, 10751, 10752, 10770)


def read_movies(source):
    files = sorted(source.glob("tmdb_movies_*.csv"))
    if not files:
        raise ValueError(f"Nenhum CSV do TMDB em {source}")
    raw = pd.concat(
        [pd.read_csv(path, usecols=SOURCE_COLUMNS) for path in files],
        ignore_index=True,
    )
    raw["release_date"] = pd.to_datetime(raw["release_date"], format="%Y-%m-%d", errors="coerce")
    for column in ("id", "vote_average", "vote_count", "runtime", "budget"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    eligible = (
        raw["id"].gt(0) & raw["release_date"].notna()
        & raw["vote_average"].between(0, 10) & raw["vote_count"].gt(0)
    )
    # Em duplicatas, usa o registro com maior cobertura de votos para o rótulo.
    movies = raw.loc[eligible].sort_values("vote_count", ascending=False, kind="stable")
    movies = movies.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    audit = {
        "arquivos_lidos": len(files),
        "linhas_lidas": len(raw),
        "linhas_sem_id_data_ou_nota_valida": int((~eligible).sum()),
        "duplicatas_removidas": int(eligible.sum()) - len(movies),
    }
    return movies, audit


def stratified_split(target):
    bands = pd.qcut(target, q=5, duplicates="drop")
    if bands.isna().any() or bands.nunique() < 2:
        raise ValueError("Notas insuficientes para estratificar em quantis")
    rng = np.random.default_rng(SEED)
    splits = pd.Series(index=target.index, dtype="string")
    for _, band in bands.groupby(bands, observed=True):
        shuffled = rng.permutation(band.index)
        train_end = round(len(shuffled) * 0.70)
        val_end = round(len(shuffled) * 0.85)
        splits.loc[shuffled[:train_end]] = "train"
        splits.loc[shuffled[train_end:val_end]] = "val"
        splits.loc[shuffled[val_end:]] = "test"
    return splits


def parse_names(value):
    return frozenset(
        normalized
        for name in value.split(",")
        if (normalized := " ".join(unicodedata.normalize("NFKC", name).casefold().split()))
    )


def add_categories(features, tokens, training, prefix, minimum, limit=None):
    counts = Counter(name for names in tokens.loc[training] for name in names)
    ranked = sorted(counts, key=lambda name: (-counts[name], name))
    known = [name for name in ranked if counts[name] >= minimum][:limit]
    for number, name in enumerate(known, start=1):
        features[f"{prefix}_{number:03d}"] = tokens.map(lambda names: name in names)
    known_names = frozenset(known)
    features[f"{prefix}_outro"] = tokens.map(lambda names: bool(names - known_names))
    features[f"{prefix}_ausente"] = tokens.map(lambda names: not names)
    return known


def standardize(values, training):
    observed = values.loc[training].dropna()
    if observed.empty:
        raise ValueError(f"Sem valores válidos no treino para {values.name}")
    median = float(observed.median())
    mean = float(observed.mean())
    scale = float(observed.std(ddof=0)) or 1.0
    return (values.fillna(median) - mean) / scale, {
        "mediana": median, "media": mean, "desvio_padrao": scale,
    }


def build_features(movies, training):
    features = {}
    numeric = {}
    features["ano_lancamento_z"], numeric["ano"] = standardize(
        movies["release_date"].dt.year, training
    )
    month = (movies["release_date"].dt.month - 1) * 2 * np.pi / 12
    features["mes_seno"] = np.sin(month)
    features["mes_cosseno"] = np.cos(month)
    for original, label in (("runtime", "duracao"), ("budget", "orcamento")):
        values = movies[original]
        observed = values.where(np.isfinite(values) & values.gt(0))
        features[f"{label}_ausente"] = observed.isna()
        features[f"{label}_log_z"], numeric[label] = standardize(np.log1p(observed), training)

    genres = movies["genre_ids"].fillna("").map(
        lambda value: frozenset(int(code) for code in re.findall(r"\d+", value))
    )
    for code in GENRES:
        features[f"genero_{code}"] = genres.map(lambda codes: code in codes)
    features["genero_ausente"] = genres.map(lambda codes: not codes)
    features["genero_desconhecido"] = genres.map(lambda codes: bool(codes.difference(GENRES)))

    categories = {}
    for original, prefix, minimum, limit in (
        ("original_language", "idioma", 100, None),
        ("director", "direcao", 10, 50),
        ("cast", "elenco", 20, 80),
        ("production_companies", "produtora", 20, 50),
    ):
        tokens = movies[original].fillna("").map(parse_names)
        names = add_categories(features, tokens, training, prefix, minimum, limit)
        categories[prefix] = {
            "frequencia_minima_treino": minimum,
            "colunas": {f"{prefix}_{number:03d}": name for number, name in enumerate(names, 1)},
        }
    frame = pd.DataFrame(features)
    binary_columns = frame.select_dtypes(include="bool").columns
    frame[binary_columns] = frame[binary_columns].astype("uint8")
    return frame, numeric, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonte", type=Path, default=ROOT.parent / "dataset")
    args = parser.parse_args()

    movies, audit = read_movies(args.fonte)
    splits = stratified_split(movies["vote_average"])
    training = splits.eq("train")
    features, numeric, categories = build_features(movies, training)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("A preparação gerou atributos não finitos")

    base = movies[["id", "vote_average"]].assign(split=splits)
    base = pd.concat([base[["id", "split", "vote_average"]], features], axis=1)
    audit.update({
        "semente_divisao": SEED,
        "particoes": {
            split: {"filmes": len(group), "nota_media": float(group["vote_average"].mean())}
            for split, group in base.groupby("split")
        },
        "numero_atributos": len(features.columns),
        "fracao_ausente": {
            column: float(features[column].mean())
            for column in features if column.endswith("_ausente")
        },
        "estatisticas_treino": numeric,
        "categorias_treino": categories,
    })
    base.to_csv(ROOT / "base_pre_lancamento.csv", index=False, float_format="%.9g")
    (ROOT / "preprocessamento_pre_lancamento.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(base):,} filmes, {len(features.columns)} atributos")
    print(base["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
