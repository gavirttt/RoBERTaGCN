# BertGCN (Filipino) — Modern Implementation

This repository implements BertGCN (Lin et al., 2021) in modern PyTorch and Hugging Face Transformers. It mirrors the original repo structure but avoids deprecated libraries.

## Requirements

To ensure proper GPU utilization, PyTorch needs to be installed with CUDA support. Follow these steps for a clean installation:

1.  **Uninstall any existing PyTorch installation** (especially if it's a CPU-only version):
    ```bash
    pip uninstall torch torchvision torchaudio
    ```

2.  **Install core requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install PyTorch with CUDA support**:
    Visit the official PyTorch website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) to get the precise installation command for your system. Make sure to select your operating system (Windows), package manager (pip), Python version, and **CUDA 13.0** (or the closest compatible version if 13.0 is not directly listed, e.g., CUDA 12.1 or 11.8).

    An example command for CUDA 13.0 might look like this (please verify the exact `cuXXX` version and URL on the PyTorch website):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    ```

## Quick start
1.  **Preprocess your data**: Run `preprocess.py` to combine and clean your labeled and unlabeled data. This will generate `data/combined_tweets.clean.csv`.
    ```bash
    python preprocess.py
    ```
2.  **Train the model**: Use the `data/combined_tweets.clean.csv` file for training. Here's a recommended command for systems with limited VRAM (like an MX450 GPU):
    ```bash
    python main.py --data data/combined_tweets.clean.csv --encoder jcblaise/roberta-tagalog-base --epochs 1 --save_dir checkpoints --bert_batch 16 --batch_size 32 --max_len 64
    ```
    *Note: Adjust `--epochs`, `--bert_batch`, `--batch_size`, and `--max_len` based on your hardware and dataset size. `--bert_batch` is crucial for GPU memory.*

## Some quickrun configurations
1. Standard Recommended Run
```bash
python main.py --data data/combined_tweets.clean.csv --encoder jcblaise/roberta-tagalog-base --epochs 10 --save_dir checkpoints --bert_batch 32 --batch_size 64 --max_len 256 --lr_bert 1e-5 --lr_gcn 1e-2 --lmbda 0.7 --feat_dim 768 --gcn_hid 256 --max_vocab 32000 --dropout 0.1 --residual --seed 42

python main.py --data data/combined_tweets.clean.csv --encoder jcblaise/roberta-tagalog-base --epochs 200 --batch_size 32 --bert_batch 32 --max_len 128 --lr_bert 2e-5 --lr_gcn 1e-3 --lmbda 0.7 --dropout 0.5 --feat_dim 768 --gcn_hid 256 --max_vocab 20000 --min_df 2 --window_size 20 --weight_decay 0.1 --seed 42 --save_dir checkpoints --plot_cm

python main.py --data data/combined_tweets.clean.csv --encoder jcblaise/roberta-tagalog-base --epochs 30 --batch_size 48 --bert_batch 48 --max_len 96 --lr_bert 5e-6 --lr_gcn 1e-3 --lmbda 0.6 --dropout 0.4 --feat_dim 768 --gcn_hid 192 --max_vocab 15000 --min_df 2 --window_size 15 --weight_decay 1e-4 --seed 42 --save_dir checkpoints
```
2. Quick Test Run
```bash
python main.py --data data/combined_tweets.clean.csv --encoder jcblaise/roberta-tagalog-base --quickrun --save_dir checkpoints --lmbda 0.7 --dropout 0.1
```
3. Low-Resource Run
```bash
python main.py --data data/combined_tweets.clean.csv --encoder jcblaise/roberta-tagalog-base --epochs 15 --save_dir checkpoints --bert_batch 16 --batch_size 32 --max_len 256 --lr_bert 5e-6 --lr_gcn 1e-2 --lmbda 0.5 --feat_dim 768 --gcn_hid 256 --max_vocab 32000 --dropout 0.1 --residual
```
## Notes

* This repository uses a memory bank strategy as described in the paper.
* Graph is constructed using TF-IDF for doc-word edges and PPMI for word-word edges.
