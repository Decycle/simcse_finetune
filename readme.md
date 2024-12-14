# SimCSE Analysis and Improvement Project Repository

This repository contains the code for our project report. The main scripts included are:

- **`train.py`**: For training the model.
- **`sweep.py`**: For hyperparameter tuning.
- **`evaluate.py`**: For model evaluation.

## Getting Started

### 1. Set Up the Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   .\venv\Scripts\activate    # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Login to Weights & Biases

Before running any scripts, log in to [Weights & Biases](https://wandb.ai):
```bash
wandb login
```

### 3. Running the Scripts

- **Train the model:**
  ```bash
  python train.py
  ```

- **Run hyperparameter tuning:**
  ```bash
  python sweep.py
  ```

- **Evaluate the model:**
  ```bash
  python evaluate.py
  ```
