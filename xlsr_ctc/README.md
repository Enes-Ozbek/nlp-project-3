# C2: XLS-R 300M + CTC, fine-tuned on our USC train split (our contribution)

Start checkpoint: [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (exact revision in the run log). A new CTC output layer is added and the model is fine-tuned on our 25 h train split.

## Settings

| Setting | Value |
|---|---|
| Output units | characters of the normalized train transcripts, `\|` = word separator, `[PAD]` = CTC blank |
| Text | `data/normalize.py` (lowercase, one apostrophe form, no punctuation) |
| Frozen | CNN feature encoder |
| Optimizer | AdamW, lr 3e-4, linear warmup 500 steps, then linear decay |
| Batch | 8 clips x 4 accumulation steps = 32, clips of similar length together |
| Other | fp16, gradient checkpointing, SpecAugment-style time masking 0.05, dropouts 0, seed 42 |
| Checkpoint choice | dev WER every 1000 steps; the lowest dev WER is kept |
| Time limit | `--max-hours` (default 11): after 50 steps the schedule is shortened to fit, so it ends inside one Kaggle session |

## Files

- `train.py`: training, dev evaluation, best checkpoint, full checkpoint for resuming (`--resume-from`).
- `predict.py`: greedy decoding for `dev` / `test` (test needs `--allow-test`); saves log-probabilities per clip for the KenLM decoder (C3, `lm/`).
- `push_to_hub.py`: uploads the best checkpoint with a model card to Hugging Face. The token comes from Kaggle Secrets or `huggingface-cli login`, never from code.
- `xlsr_lib.py`: vocabulary, data loading, batching, decoding.

## Run (from the repo root)

```bash
python xlsr_ctc/train.py --out-dir runs/xlsr
python xlsr_ctc/predict.py --model runs/xlsr/best --split dev --logits-dir runs/logits
python xlsr_ctc/predict.py --model runs/xlsr/best --split test --allow-test --logits-dir runs/logits
python xlsr_ctc/push_to_hub.py --model-dir runs/xlsr/best --repo <hf-user>/<name>
```

Logits format for C3: `<logits-dir>/<split>/<clip id>.npy`, float16 log-probabilities, shape (frames, vocabulary), 20 ms per frame; `<logits-dir>/labels.json` is the token list in id order (`""` = blank, `" "` = word separator), ready for `pyctcdecode.build_ctcdecoder`.

Outputs: `benchmark/predictions/C2_xlsr-300m-ctc__<split>.jsonl`, `benchmark/results/C2_xlsr-300m-ctc.json` (all settings, revision, chosen step, dev WER, RTF, package versions), `benchmark/results/C2_xlsr-300m-ctc_train_log.jsonl` (loss and dev WER during training). Model weights and logits are not committed (Hugging Face link in the README).
