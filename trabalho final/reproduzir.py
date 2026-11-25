"""Reproduz uma linha do índice com o código e os dados exatos daquela execução.

Na raiz do D3: python 03_codigo/reproduzir.py
No ZIP extraído: python reproduzir.py
--execucao escolhe outra linha; --out define uma pasta nova para a réplica.
Os registros originais nunca são sobrescritos e o teste não é reavaliado.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.name == "03_codigo"
BASE = CODE.parent if WORKSPACE else CODE
INDEX = BASE / ("04_resultados/experimentos.csv" if WORKSPACE else "experimentos.csv")
DEFAULT = "final__confirm_muon_warmup_cosine_s17"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--execucao", default=None, help="Chave da coluna execucao no índice")
    source.add_argument("--config", type=Path, help="Configuração salva de uma execução")
    parser.add_argument("--dados", type=Path, help="Pasta com dataset.npz, preprocessamento.json e auditoria.json")
    parser.add_argument("--out", type=Path, help="Pasta nova; padrão: _apoio/reproducoes/<data> no workspace")
    args = parser.parse_args()
    if args.config:
        config = args.config.resolve()
        data = args.dados.resolve() if args.dados else config.parents[2] / "dados"
    else:
        with INDEX.open(encoding="utf-8", newline="") as f:
            found = [r for r in csv.DictReader(f) if r["execucao"] == (args.execucao or DEFAULT)]
        if len(found) != 1:
            parser.error("Execução ausente ou ambígua; consulte a coluna execucao do índice.")
        config = BASE / found[0]["config_path"]
        data = BASE / found[0]["dados_path"]
    meta = json.loads(config.read_text(encoding="utf-8"))
    snapshot = config.parent.parent / "codigo" / ("treino_" + meta["source_sha256"] + ".py")
    if sha(snapshot) != meta["source_sha256"]:
        raise ValueError("Código original difere do hash registrado.")
    prep = json.loads((data / "preprocessamento.json").read_text(encoding="utf-8"))
    audit = json.loads((data / "auditoria.json").read_text(encoding="utf-8"))
    digest = sha(data / "dataset.npz")
    if digest != prep["dataset_npz_sha256"] or digest != meta["dataset_npz_sha256"]:
        raise ValueError("Tensores diferentes da execução original.")
    if audit["data_sha256"] != meta["data_sha256"]:
        raise ValueError("Manifesto de dados diferente da execução original.")
    destination = args.out.resolve() if args.out else BASE / ("_apoio/reproducoes" if WORKSPACE else "reproducoes") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "dados").mkdir()
    for name in ["dataset.npz", "preprocessamento.json", "auditoria.json"]:
        shutil.copy2(data / name, destination / "dados" / name)
    # O snapshot é autossuficiente; a verificação do SHA original continua ativa.
    command = [sys.executable, str(snapshot), "--out", str(destination),
               "--replay", str(config), "--run-id", "replica"]
    subprocess.run(command, check=True)
    curves = destination / "logs/replica/curvas.csv"
    original_curves = config.parent / "curvas.csv"
    result = json.loads((destination / "logs/replica/resultado.json").read_text(encoding="utf-8"))
    original = json.loads((config.parent / "resultado.json").read_text(encoding="utf-8"))
    receipt = {
        "original_config": str(config), "codigo_sha256": meta["source_sha256"],
        "dataset_sha256": digest, "curvas_identicas_byte_a_byte": curves.read_bytes() == original_curves.read_bytes(),
        "metricas_identicas": all(result[k] == original[k] for k in ["status", "steps_completed", "val_mae", "val_rmse", "val_mse"]),
        "inicializacao_identica": result["initial_parameters_sha256"] == meta["initial_parameters_sha256"],
        "lotes_identicos": result["batch_order_sha256"] == meta["batch_order_sha256"],
        "teste_reavaliado": False,
    }
    (destination / "verificacao.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saida": str(destination), **receipt}, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    main()

