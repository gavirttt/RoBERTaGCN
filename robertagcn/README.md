# BertGCN — Modern Implementation

This repository implements BertGCN (Lin et al., 2021) in modern PyTorch and Hugging Face Transformers. It mirrors the original repo structure but avoids deprecated libraries.

## Requirements

To ensure proper GPU utilization, PyTorch needs to be installed with CUDA support. Follow these steps for a clean installation:

1. **Setup and initialize the environment** :
    *   ```bash
        git clone https://github.com/gavirttt/RoBERTaGCN
        cd RoBERTaGCN/robertagcn
        ```
    *   ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
        ```

2.  **Uninstall any existing PyTorch installation** (especially if it's a CPU-only version):
    ```bash
    pip uninstall torch torchvision torchaudio
    ```

3.  **Install core requirements**:
    ```bash
    pip install -r requirements.txt
    ```
    The core requirements are: `transformers>=4.45`, `scikit-learn>=1.0`, `scipy>=1.9`, `pandas>=1.3`, `numpy`, `tqdm`, `matplotlib>=3.5`, `seaborn>=0.11`.

4.  **Install PyTorch with CUDA support**:
    Visit the official PyTorch website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) to get the precise installation command for your system. Make sure to select your operating system (Windows), package manager (pip), Python version, and the appropriate CUDA version.

    An example command for CUDA 13.0 might look like this (please verify the exact `cuXXX` version and URL on the PyTorch website):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    ```

## Quick start
1.  **Prepare your data**: Ensure the path to your datasets is specified in the `robertagcn/config.py`.

2.  **Train the model**: Use `main.py` with a preset configuration or custom arguments.
    *   **Using the default config (recommended)**:
        ```bash
        python main.py
        ```
        This will use the `default` configuration defined in `robertagcn/config.py`. Other presets include `quickrun`, `low_resource`, and `high_quality`.
    *   **Using custom arguments (overrides config)**:
        ```bash
        python main.py --labeled-data data/my_labeled_data.csv --epochs 10 --bert_batch 16 --batch_size 32 --max_len 64
        ```
        *Note: Adjust `--epochs`, `--bert_batch`, `--batch_size`, and `--max_len` based on your hardware and dataset size. `--bert_batch` is crucial for GPU memory.*

## Configuration Presets
The `main.py` script supports various configuration presets defined in `robertagcn/config.py`. You can use them with the `--preset` argument:

*   **Default Run**: Balanced configuration (recommended)
    ```bash
    python main.py --preset default
    ```
*   **Quick Test Run**: Fast testing with minimal data
    ```bash
    python main.py --preset quickrun
    ```
*   **Low-Resource Run**: For systems with limited GPU memory
    ```bash
    python main.py --preset low_resource
    ```
*   **High-Quality Run**: Best quality, requires good GPU
    ```bash
    python main.py --preset high_quality
    ```
## Notes

* This repository uses a memory bank strategy as described in the paper.
* Graph is constructed using TF-IDF for doc-word edges and PPMI for word-word edges.
