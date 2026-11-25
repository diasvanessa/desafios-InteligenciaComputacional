# Trabalho final — previsão de notas antes do lançamento

O alvo é a nota média observada no TMDB. Os atributos usam data de estreia,
gêneros, idioma, duração, orçamento, direção, elenco e produtoras.
A contagem de votos serve para selecionar filmes com nota observada e
resolver IDs duplicados. Contagem de votos, popularidade e receita ficam
fora das entradas.

## Preparar os dados e treinar

Na pasta `trabalho final`, com os CSVs anuais do TMDB em `../dataset`:

```powershell
python preparar_dados.py
python treino.py --optimizer adam --lr 0.001 --schedule constant
```

A preparação gera `base_pre_lancamento.csv` e
`preprocessamento_pre_lancamento.json`. O JSON registra a auditoria, as
estatísticas e o nome associado a cada coluna de categoria. Use `--fonte`
para indicar outro diretório de CSVs.

A divisão aleatória de aproximadamente 70%/15%/15% usa cinco faixas da nota
e semente 42. O treino usa as partições de treino e validação. O teste
fica reservado para a avaliação final. Duração e orçamento sem valor positivo
recebem um indicador de ausência e são preenchidos pela mediana do treino
na escala logarítmica. A padronização e os vocabulários de idioma e créditos
também são ajustados apenas com dados do treino.

Os dados são retrospectivos e o TMDB não informa a data em que cada atributo
ficou disponível. Em produção, use os valores conhecidos no momento da previsão.

## Histórico do Desafio 3

[Execuções](experimentos.csv) · [Resumo](resumo.json)

A baseline é Adam com taxa constante 0,001, nas sementes 17, 29 e 43. A receita escolhida entre os três finalistas é Muon com aquecimento + cosseno. As avaliações de 2023–2025 e 2026 são retrospectivas em períodos já consultados.

## Arquivos

| Arquivo ou pasta | Conteúdo |
| --- | --- |
| `treinar_modelo.py` | Script histórico da rede de 44 entradas. |
| `base_treino.csv` | Base histórica de treino e validação com corte por ano. |
| `experimentos.csv` | 67 execuções, configuração, número de entradas, métricas e comando de reprodução. |
| `logs/` | Configuração, curvas, resultados, checkpoint e snapshots do código de todas as execuções. |
| `dados/controle/` | Tensores e preparação dos seis controles de 43 entradas. |
| `dados/final/` | Tensores e preparação das execuções com 44 entradas. |

## Executar o modelo histórico

Na pasta `trabalho final`:

```powershell
python -m pip install -r requirements.txt
python treinar_modelo.py --optimizer muon --lr 0.03162277660168379 --aux-lr 0.1 --schedule warmup_cosine --seed 17 --steps 2400
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

Extraia o ZIP antes de executar os comandos ou abrir os documentos.
