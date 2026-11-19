# C4: NVIDIA Uzbek FastConformer, fine-tuned on our USC train split (our contribution)

Start checkpoint: [`nvidia/stt_uz_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc), the same model that is used as-is for the baselines A8/A9 (exact revision in the run log). Here it is trained further on our 25 h train split, so A8/A9 vs C4 shows what our training adds.

The model has one encoder and two decoders, so C4 gives two rows:

| Row | Head | Decoding |
|---|---|---|
| C4a | Transducer (RNN-T) | greedy |
| C4b | CTC | greedy |

## Settings

| Setting | Value |
|---|---|
| Training | both heads together, with the model's own loss weights |
| Tokenizer | the start model's SentencePiece tokenizer, unchanged; our apostrophe is written in the form the tokenizer knows |
| Text | `data/normalize.py` (lowercase, no punctuation), so the fine-tuned model also writes lowercase without punctuation |
| Audio | `data/clean/train.csv`; silence trimming through `offset`/`duration` in the NeMo manifest |
| Learning rate | chosen on dev: 5e-5, 1e-4, 2e-4 and 4e-4 each train 1 epoch from the start model and are scored on the same 1,000 dev clips; the best one is used (`benchmark/results/C4_lr_search.json`) |
| Optimizer | AdamW (0.9, 0.98), weight decay 1e-3, warmup 500 steps, then cosine to 1e-6 |
| Batch | 16 clips x 2 accumulation steps = 32 |
| Other | fp16 mixed precision, gradient clipping 1.0, the model's own SpecAugment, seed 42 |
| Checkpoint choice | dev WER (transducer head) on a fixed sample of 1,000 dev clips (seed 42) after every epoch; the best epoch is kept. The start model is scored on the same clips before training, so the gain is visible in the log |
| Length | as many epochs as fit in the Kaggle time (from the speed measured in the learning-rate search), at most 30; `--max-hours` stops it in any case |

## Files

- `train.py`: writes the NeMo manifests, trains, keeps the best epoch, saves `<out-dir>/best.nemo` and `train_log.jsonl`.
- `predict.py`: C4a and C4b on `dev` / `test` (test needs `--allow-test`), from `best.nemo` or from Hugging Face.
- `push_to_hub.py`: uploads `best.nemo` with a model card to Hugging Face. The token comes from Kaggle Secrets or `huggingface-cli login`, never from code.
- `requirements.txt`: pinned NeMo packages (same versions as A8/A9). `numba-cuda` runs the RNN-T loss on the GPU; the Kaggle notebook installs it only when numba cannot run CUDA kernels yet.

## Run (from the repo root)

```bash
pip install -r nvidia_finetune/requirements.txt
python nvidia_finetune/train.py --out-dir runs/c4
python nvidia_finetune/predict.py --model runs/c4/best.nemo --split dev
python nvidia_finetune/predict.py --model runs/c4/best.nemo --split test --allow-test
python nvidia_finetune/push_to_hub.py --nemo runs/c4/best.nemo --repo <hf-user>/<name>
```

Outputs: `benchmark/predictions/C4a_fastconformer-usc-ft-rnnt__<split>.jsonl`, `benchmark/predictions/C4b_fastconformer-usc-ft-ctc__<split>.jsonl`, `benchmark/results/C4a_*.json`, `benchmark/results/C4b_*.json` (settings, revisions, best epoch, RTF, package versions), `benchmark/results/C4_fastconformer-usc-ft_train_log.jsonl`. Official WER/CER come from `benchmark/evaluate.py`. Model weights are not committed (Hugging Face link in the README).
