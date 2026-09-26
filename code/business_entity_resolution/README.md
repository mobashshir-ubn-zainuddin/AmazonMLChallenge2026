# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline (v2, package `src/er`) that produces `output/candidate_pairs.tsv`
and `output/matching_results.tsv` from the raw challenge TSVs. Design rationale:
`Implementation_Plan_v2.txt` at the repository root.

## Approach in one paragraph
Each S2/S3 record belongs to at most one S1 record (verified on the training labels),
so retrieval runs in reverse: every S2/S3 record retrieves its top S1 records inside
its country. It uses exact brute-force cosine over dense hashed char-3-gram TF-IDF
vectors of the name and the address (GPU matmuls). An XGBoost pair classifier scores
the candidates using string-similarity, address-number and competition features
(margin to the second-best S1, ranks). Each record is then assigned to its best S1
if the probability clears a threshold tuned for macro F0.5 on held-out S1 records,
with per-S1 caps (≤5 S2, ≤6 S3). Native-script names and state names are mapped
to Latin script with dictionaries learned from the training labels only. No
external data is used.

## Run on Kaggle (recommended: GPU T4 x2, Internet on)
Open `kaggle_pipeline.ipynb`, set `DATA_ROOT` / `CODE_ROOT`, and Run All.
(Avoid P100: recent PyTorch builds have no kernels for it.)

## Run locally / anywhere
```bash
export PYTHONPATH=<repo>/code          # PowerShell: $env:PYTHONPATH="$PWD\code"
M=business_entity_resolution.src.er.run
python -m $M prepare  --split train --data dataset --work work
python -m $M retrieve --split train --work work          # prints retrieval recall + oracle F0.5 ceiling
python -m $M features --split train --work work
python -m $M train    --work work                        # prints validation macro F0.5
python -m $M prepare  --split test  --data dataset --work work
python -m $M retrieve --split test  --work work
python -m $M predict  --work work --out output
python dataset/utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/dataset/test
```
Quick smoke test on a 2% sample (a few minutes on CPU):
```bash
python -m $M sample --data dataset --out sample --fraction 0.02
python -m $M prepare --split train --data sample --work work_s   # ...then the stages above with --work work_s
```

## Stages and artifacts (`--work` folder)
| stage | output |
|---|---|
| prepare | `<split>/s1.pkl`, `q.pkl`, `*_nums.npy`, `q_true_s1.npy` (train), `translit.json` |
| retrieve | `<split>/cands.npz`, `retrieval_report.pkl` (train) |
| features | `train/X.npy`, `pairs_meta.npz`, `val_s1.npy` |
| train | `train/matcher.json`, `matcher_meta.json` (threshold, validation F0.5) |
| predict | `output/candidate_pairs.tsv`, `output/matching_results.tsv` |

Useful knobs: `--k-name/--k-addr/--k-comb` (candidates per record), `--dim` (vector
size), `--gpu-budget-gb`, `--train-query-frac`, `--val-frac`, `--rounds`.

## Code map (`src/er`)
- `text.py` normalization (names, addresses, leet/accents/legal/domain handling) + translit learning
- `prepare.py` raw → normalized tables, GT arrays
- `vectors.py` hashed TF-IDF → dense vectors
- `retrieve.py` GPU/CPU top-K reverse retrieval
- `features.py` pair features; `train.py` XGBoost + decision rule; `predict.py` outputs
- `metrics.py` exact challenge metric (per-S1 macro F0.5) and oracle ceiling

Earlier experiments (12-rule blocking, token/exact retrieval) live in git history (commit aa2292d) and are no longer part of the pipeline.
