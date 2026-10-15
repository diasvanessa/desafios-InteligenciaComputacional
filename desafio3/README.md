# Desafio 3 — TMDB

[Slide](slide.pdf) · [Painel](painel.pdf) · [Execuções](experimentos.csv) · [Resumo](resumo.json)

A baseline é Adam com taxa constante 0,001, nas sementes 17, 29 e 43. A receita escolhida entre os três finalistas é Muon com aquecimento + cosseno. As avaliações de 2023–2025 e 2026 são retrospectivas em períodos já consultados.

## Arquivos

| Arquivo ou pasta | Conteúdo |
| --- | --- |
| `treino.py` | Treino configurável da rede atual de 44 entradas. |
| `base_treino.csv` | Dados preparados da rede atual, treino e validação. |
| `experimentos.csv` | 67 execuções, configuração, número de entradas, métricas e comando de reprodução. |
| `logs/` | Configuração, curvas, resultados, checkpoint e snapshots do código de todas as execuções. |
| `dados/controle/` | Tensores e preparação dos seis controles de 43 entradas. |
| `dados/final/` | Tensores e preparação das execuções com 44 entradas. |

## Executar o modelo atual

Na pasta extraída `desafio3/`:

```powershell
python -m pip install -r requirements.txt
python treino.py --optimizer muon --lr 0.03162277660168379 --aux-lr 0.1 --schedule warmup_cosine --seed 17 --steps 2400
```

`--lr` é a taxa máxima das matrizes ocultas atualizadas por Muon. `--aux-lr` é a taxa máxima do AdamW que atualiza a saída e os vieses. A agenda varia as duas taxas com o mesmo fator.

## Reproduzir uma linha do índice

Use a coluna `execucao` ou copie o comando da coluna `command`. A reprodução seleciona os dados de 43 ou 44 entradas e o snapshot correspondente ao hash registrado.

```powershell
python reproduzir.py --execucao final__baseline_s17
python reproduzir.py --execucao final__confirm_muon_warmup_cosine_s17
python reproduzir.py --execucao controle__baseline_s17
```

A réplica fica em uma nova pasta `reproducoes/`. Os registros originais são preservados e os testes temporais não são reavaliados. O JSON de verificação compara curvas, métricas, inicialização e ordem dos lotes.

Mantenha `slide.pdf` e `painel.pdf` juntos para usar o link entre os PDFs. Extraia o ZIP antes de executar os comandos ou abrir os documentos.
