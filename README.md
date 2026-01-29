# Shapelet-based Few-Shot Learning for Multivariate Time Series Classification

Implementation of a discriminative shapelet extraction,
encoding and few-shot learning pipeline for multivariate time-series.

**Repository Goals:**
- Provide tools to extract discriminative shapelets from time-series data.
- Encode time-series using learned shapelets into fixed-size embeddings.
- Train few-shot classifiers (CNN-based encoder) on encoded data.
- Offer a Streamlit dashboard for visualization and inference.

**Key Features:**
- Discriminative multivariate shapelet extraction
- Shapelet-based encoding to vector representations
- CNN few-shot training pipeline
- Streamlit dashboard for visualization and inference

**Quick Links:**
- User guide: [UserGuide.txt](ITDSIU21109_TranBaoPhuc_UserGuide.txt)

**Requirements**
- Python 3.8+ (recommend 3.8–3.11)
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

**Quick Start (Windows PowerShell)**
- Activate environment (if present):

  ```powershell
  & "env\Scripts\Activate.ps1"
  ```

- Run shapelet extraction (example):

  ```powershell
  python Discriminative_Shapelet_Transform/Multivariate/multi_extract.py
  ```

- Train the few-shot model (example):

  ```powershell
  python Discriminative_Shapelet_Transform/Multivariate/CNN_FewShot.py
  ```

- Run the Streamlit dashboard:

  ```powershell
  streamlit run Discriminative_Shapelet_Transform/app_v2.py
  ```

**Repository Layout (important files and folders)**
- `data/` : dataset files in `.ts` or `.arff` format.
- `Discriminative_Shapelet_Transform/` : main implementation folder.
- `Discriminative_Shapelet_Transform/Multivariate/` : multivariate extraction & training scripts.
- `ITDSIU21109_TranBaoPhuc_UserGuide.txt` : original user guide reference.
- `requirements.txt` : Python package requirements.

**Usage Notes & Tips**
- Inspect the Python scripts to confirm expected CLI arguments and paths before running.
- Save models and encodings under `models/` and `encodings/` respectively for reproducibility.
- For GPU training, install a CUDA-enabled `torch` build and confirm device availability in scripts.

**Contributing**
- Fork the repository, create a feature branch, run tests, and open a pull request.

**Contact**
- For questions or collaboration, contact the repository owner.
