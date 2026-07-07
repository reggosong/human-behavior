# human-behavior

Multimodal machine learning for human behavior understanding — a collaborative
project building the data and objective layer for multi-task behavior modeling
across affective, pathology, social, cognitive, and personality dimensions.

**Status: work in progress.** Dataloaders and losses are implemented; the
model and training loop are under development.

## What's here

- **Dataloaders** (`dataloaders/`)
  - `mosei_loader.py` — CMU-MOSEI sentiment: loads preprocessed
    `mosei_senti_data.pkl`, returns aligned vision (50×35), audio (50×74), and
    text (50×300) tensors per sample.
  - `meld.py` — MELD emotion recognition: reads `{split}_sent_emo.csv` plus
    utterance `.mp4` clips, 7-class emotion map, lazy video loading.
  - `daic_loader.py` — DAIC-WOZ depression screening: per-subject zip archives
    (transcripts, COVAREP audio features, CLNF visual features), PHQ-8 labels.
- **Multi-task loss** (`loss/multi_task_loss.py`) — weighted combination of
  per-task objectives: CCC loss for valence/arousal regression, MSE for
  pathology and cognitive scores, BCE for social and personality traits.
- **Dataset abstractions** (`dataset/template.py`) — `MultimodalSample` and
  `BaseMultimodalDataset` covering text, audio, video, face, gesture, physio,
  and EEG modalities.
- **Smoke tests** (`test_scripts/dataloader_test.py`) and a MELD download
  helper (`test_scripts/download_datasets.py`, via kagglehub).

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run the dataloader smoke test

```bash
python -m test_scripts.dataloader_test   # expects datasets/MOSEI/mosei_senti_data.pkl
```

Expected dataset layout:

```
datasets/
├── MOSEI/mosei_senti_data.pkl
├── MELD/{train,dev,test}_sent_emo.csv + {split}/{split}_splits/*.mp4
└── DAIC-WOZ/{id}_P.zip + split CSVs
```

## Contributors

Built with [Dewei Feng](https://github.com/DeweiFeng) (project lead: losses,
MELD/DAIC-WOZ loaders, dataset templates) and collaborators.

My contributions (reggosong): the CMU-MOSEI dataloader
(`dataloaders/mosei_loader.py`), its test harness in
`test_scripts/dataloader_test.py`, and project tooling (`.gitignore`).
