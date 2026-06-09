
# Energy Forecasting Pipeline
End-to-end deep learning pipeline for multivariate time-series forecasting.

## 📌 Project Overview
This repository contains a production-grade Machine Learning Operations (MLOps) pipeline designed to forecast energy consumption. It strictly adheres to Software Engineering best practices, including the Single Responsibility Principle (SRP), automated data orchestration, and strict temporal data isolation.

The pipeline compares traditional Machine Learning baselines (Linear Regression, Random Forest, XGBoost) against a dynamically tuned Deep Learning architecture (CNN-LSTM) to determine the most robust forecasting strategy.

## 📂 Repository Structure
*Note: To adhere to strict version control best practices and data privacy standards, all raw and processed datasets have been intentionally excluded from this repository. The `data/` directory is automatically generated locally during Phase 1 of pipeline execution.*

```markdown

Energy-Forecasting-Pipeline/
│
├── models/                     # Serialized artifacts and visualizations
│   ├── images/                 # Generated evaluation charts
│   │   ├── 1_metric_comparison.png
│   │   ├── 2_timeseries_predictions.png
│   │   ├── 3_actual_vs_predicted_scatter.png
│   │   ├── 4_residual_scatter.png
│   │   ├── 5_residual_distribution.png
│   │   └── feature_importance.png
│   ├── cnn_lstm_tuned.keras    # Champion deep learning model
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
│
├── notebooks/                  # Jupyter environments for research & testing
│   ├── exploratory_data_analysis.ipynb  # Initial EDA and visual insights
│   └── test_pipeline.ipynb              # Testing environment for data transforms
│
├── src/                        # Source code for the pipeline
│   ├── build_dataset.py        # Master orchestrator for data fetching & engineering
│   ├── config.py               # Global path and variable configurations
│   ├── data_loader.py          # Data ingestion logic
│   ├── data_preprocessing.py   # Scaling and temporal splitting logic
│   ├── evaluate_models.py      # Final metrics and visualization orchestrator
│   ├── feature_engineering.py  # Generation of physics-based temporal features
│   ├── feature_selection.py    # Algorithmic feature reduction diagnostics
│   ├── reduce_dimensions.py    # Logic for dropping weak features
│   ├── train_baseline.py       # Baseline model training and serialization
│   ├── train_dl.py             # Initial Deep Learning baseline training
│   └── tune_model.py           # KerasTuner optimization for CNN-LSTM
│
├── .gitignore                       # Version control rules
├── README.md                        # Project documentation
└── requirements.txt                 # Frozen Python dependencies

```

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/yahal2005/Energy-Forecasting-Pipeline.git
cd Energy-Forecasting-Pipeline

```

**2. Create a Virtual Environment**
It is highly recommended to isolate dependencies using a virtual environment.

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

**3. Install Dependencies**
Install the required packages strictly from the frozen requirements file to ensure reproducibility.

```bash
python -m pip install -r requirements.txt

```

## 🚀 Execution Guide (The Pipeline)

To reproduce the findings, execute the pipeline scripts in the following order. The architecture is modular and strictly decoupled.

### Phase 1: Data Orchestration & Engineering (Mandatory)

*Because datasets are excluded from version control, this script must be run first.* It calls the modular helper files (`data_loader.py`, `feature_engineering.py`, etc.) to dynamically reconstruct the local `data/` directory, generate temporal features, scale the data, and enforce a strict 70/20/10 temporal split.

```bash
python src/build_dataset.py

```

### Phase 2: Diagnostic Feature Selection (Optional)

Generates a Random Forest feature importance chart (`models/images/feature_importance.png`) to mathematically validate the exclusion of weak features.

```bash
python src/feature_selection.py

```

### Phase 3: Baseline Training

Trains the Linear Regression, Random Forest, and XGBoost baselines on the training data and serializes them as `.joblib` artifacts inside the `models/` directory.

```bash
python src/train_baseline.py

```

### Phase 4: Deep Learning Optimization

Utilizes KerasTuner to dynamically hunt for the optimal CNN-LSTM architecture, saving the ultimate champion as a `.keras` artifact.

```bash
python src/tune_model.py

```

### Phase 5: The Final Showdown (Evaluation)

Loads all serialized artifacts and the heavily isolated test dataset to generate final metrics (MAE, RMSE) and saves 5 professional-grade visualizations to the `models/images/` directory.

```bash
python src/evaluate_models.py

```
