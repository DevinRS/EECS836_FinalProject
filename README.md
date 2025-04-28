# 🧠 Deep Learning for IMU Classification

## 📌 Overview
This project investigates the use of deep learning models to classify human activity using time-series data from Inertial Measurement Units (IMUs). The classification task includes:

- **No Activity**
- **Walking**
- **Bending**

We use lab-collected IMU datasets to train and evaluate models, with a long-term goal of deploying on edge devices.

---

## 🎯 Objectives
- ✅ Train deep learning models for IMU-based activity classification.
- 📊 Compare model performance across different architectures (accuracy, F1-score).
- 📈 Present visualizations of evaluation results.
- ⏳ *Optional*: Compare global vs personalized models using transfer learning.
- 📦 *Optional*: Apply model compression techniques for edge deployment feasibility.

---

## 🚀 Expected Outcomes
- End-to-end pipeline for activity classification from IMU data.
- Performance benchmarks across model architectures.
- Insights into personalization and edge optimization.
- Final report and demo showcasing our results.

---

## 👥 Team Members & Roles

| Name             | Primary Focus                                   |
|------------------|--------------------------------------------------|
| **Maisoon Rahman** | Model architecture (CNNs, Transformers)         |
| **Liken Hananto**  | Data preprocessing & metrics evaluation         |
| **Devin Setiawan** | Transfer learning & model compression           |

> 🔄 **Note:** We will be working collaboratively on all parts of the project. The roles above indicate who is leading each component, not working in isolation.

---

# 📕 Tutorial

## 1. Environment Setup 🐍
You can use either Conda or venv to set up Python 3.12.3.
### Using Conda
```bash 
conda create -n myEnv python=3.12.3 -y 
conda activate myEnv
```
### Using venv
```bash 
python3.12 -m venv myEnv
source myEnv/bin/activate # On Windows use: myEnv\Scripts\activate
```
### Install the requirements needed
```bash 
pip install -r requirements.txt
```
### Set the PYTHONPATH
To make the utils folder available for imports throughout the project, set the PYTHONPATH:
```bash 
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
On Windows Command Prompt, use:
```bash 
set PYTHONPATH=%PYTHONPATH%;%cd%
```

---

## 2. Running the code 🏃
The project is organized as follows:
### File structure
```plaintext
EECS836_FinalProject/    # Main project folder (contains .git, run everything here)
├── phase1_uci_dataset/   # Approach I: Heavy preprocessing
│   ├── data_visualization.py    # Visualize data (PCA, t-SNE, class distribution)
│   ├── resource_analysis.py     # Analyze model sizes
│   └── simple_models.py         # Traditional models (Logistic Regression, SVM, Random Forest, XGBoost)
│
├── phase2_uci_minimal/   # Approach II: Minimal preprocessing (mean, std only on UCI)
│   ├── data_visualization.py
│   ├── resource_analysis.py
│   └── simple_models.py
│
└── phase3_lab_data/      # Approach II applied to Lab dataset
    ├── data_preprocessor.py    # Transform raw lab data using sliding window
    ├── data_visualization.py
    ├── resource_analysis.py
    └── simple_models.py
```
To run the code for each phase, navigate to the project root (EECS836_FinalProject) and use:
### Running the script
```bash
# Example: Run simple models in phase 1
python phase1_uci_dataset/simple_models.py

# Example: Visualize data for phase 2
python phase2_uci_minimal/data_visualization.py

# Example: Preprocess raw lab data and run models
python phase3_lab_data/data_preprocessor.py
python phase3_lab_data/simple_models.py
# (If 'python' does not work, try 'python3')
```
✅ Make sure you have already run the environment setup steps and have the correct dependencies installed.

✅ All code should be run from the root directory (EECS836_FinalProject), so relative imports and file paths work properly.



