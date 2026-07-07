# 🏥 Earpiece Fall Detection — End-to-End ML Engineering on Databricks

A production-quality machine learning pipeline for **fall detection using wearable IMU sensors**, built entirely on Databricks. The project transforms raw sensor recordings from the SisFall dataset into trained, evaluated, and registered models — following real ML engineering practices.

> **Primary goal:** Deep hands-on experience with Databricks, Delta Lake, Unity Catalog, and MLflow while solving a clinically meaningful problem.

---

## 🎯 Problem

Falls are the leading cause of injury-related death in elderly people. A wearable earpiece equipped with an IMU sensor can detect falls in real time — but only with a reliable ML model trained on realistic sensor data.

This project builds that model from scratch: raw sensor data → signal processing → feature engineering → model selection → final evaluation on unseen subjects.

---

## 📊 Dataset — SisFall

| Property | Value |
|---|---|
| Subjects | 38 (SA01–SA23 young adults, SE01–SE15 elderly) |
| Fall types | 15 (F01–F15) |
| ADL types | 19 (D01–D19) |
| Sensor | 6-axis IMU — accelerometer + gyroscope @ 200 Hz |
| Raw recordings | 4,505 |
| **Test subjects** | SA23 (young adult), SE06 (only elderly subject with falls) |

Test subjects are completely isolated from development — never seen until final evaluation.

---

## 🏗️ Architecture — Medallion on Databricks

```
Raw ZIP on Unity Catalog Volume
        ↓
  🥉 Bronze Layer
  Raw signals + metadata → Delta table
        ↓
  🥈 Silver Layer
  Preprocessed signals (6, 800) → NumPy files
        ↓
  🥇 Gold Layer
  68 features per signal → Delta tables
        ↓
  MLflow Experiments → Model Registry
```

---

## 📁 Project Structure

```
fall_detection_project/
├── configs/
│   └── config.yaml                  ← single source of truth for all parameters
├── src/
│   ├── functions.py                 ← signal processing & feature extraction
│   └── config.py                    ← YAML loader
├── notebooks/
│   ├── 01_data_ingestion.py         ✅ Read ZIP, EDA, Bronze Delta table
│   ├── 02_preprocessing.py          ✅ Activity-specific pipeline, Silver layer
│   ├── 03_feature_engineering.py    ✅ 68 features, EDA, Gold Delta tables
│   ├── 04_model_training.py         ✅ 7 models, MLflow, binary + multiclass
│   ├── 05_feature_selection.py      🔄 Forward floating SFS per model (mlxtend)
│   ├── 06_hyperparameter_tuning.py  📋 Planned
│   ├── 07_evaluation.py             📋 Planned — unseen test set (SA23, SE06)
│   └── 08_model_registry.py         📋 Planned — MLflow Model Registry
└── tests/
    └── test_functions.py            ← pytest unit tests for src/functions.py
```

---

## ⚙️ Preprocessing Pipeline

Each activity group is preprocessed differently — based on its signal characteristics:

| Group | Activities | Method |
|---|---|---|
| Falls | F01–F15 | Peak-centred window (keep_from_peak) |
| Walking / Jogging | D01–D04 | Split 100s recording into 4s windows |
| Stairs | D05–D06 | Remove idle → extract 2 movement segments |
| Sit / Stand | D07–D10 | Extract sitting and standing separately |
| Lie / Rise | D12–D13 | Extract lying down and rising up separately |
| Bend / Straighten | D15–D16 | Extract bending and straightening separately |
| Other | D11, D18, D19 | Peak-centred window |
| **Excluded** | D14, D17 | D14: lateral rolling not a distinct ADL — D17: requires ~1000 samples |

All signals normalised to exactly **(6 axes × 800 samples) = 4 seconds at 200 Hz**.

---

## 🔢 Features — 68 per signal

| Group | Count | Description |
|---|---|---|
| Accelerometer | 33 | RMS, peak-to-peak, jerk, mean/std/skew/kurtosis per axis, spectral entropy, dominant frequency, cross-axis correlations |
| Gyroscope | 29 | Same as accelerometer (excluding velocity estimate and orientation features) |
| Cross-sensor | 6 | Correlations between accelerometer and gyroscope axes |

---

## 🤖 Models & Results

7 classifiers trained with **5-fold stratified cross-validation**, all runs tracked in MLflow:

| Model | Binary CV F1 | Multiclass CV F1 |
|---|---|---|
| **SVM (RBF)** | **0.9979** | **0.9606** |
| NeuralNetwork (MLP) | 0.9976 | 0.9518 |
| XGBoost | 0.9970 | 0.9545 |
| GradientBoosting | 0.9954 | 0.9440 |
| LogisticRegression | 0.9955 | 0.9100 |
| RandomForest | 0.9960 | 0.9261 |
| KNN | 0.9694 | 0.6915 |

**SVM is the best model for both binary and multiclass tasks.**

Scale-sensitive models (SVM, LogisticRegression, NeuralNetwork) trained on StandardScaler-normalised features.
Tree-based models trained on raw features — scaling has no effect on split-based decisions.

---

## 🔍 Feature Selection

Forward floating Sequential Feature Selection (SFS) via mlxtend, run per model on a stratified subset (frac=0.5, equal samples per activity class):

| Model | Binary features | Binary score | Time |
|---|---|---|---|
| SVM | 14 | 1.0000 | ~1h |
| RandomForest | 8 | 0.9988 | ~13h |
| NeuralNetwork | 23 | 1.0000 | ~4.8h |
| XGBoost | 19 | 0.9976 | ~39min |
| KNN | 16 | 0.9926 | ~6min |
| LogisticRegression | 50 | 0.9940 | ~7min |
| GradientBoosting | 43 | 0.9976 | ~34h |

Most consistent feature across all models: **`acc_rms_mag`**

Multiclass SFS in progress — run model-by-model with per-model JSON persistence to handle session timeouts.

---

## 🛠️ Tools & Platform

| Tool | Role |
|---|---|
| Databricks (Serverless) | Compute, notebooks, workspace |
| Unity Catalog | Data governance, volumes, schemas |
| Delta Lake | Medallion architecture (Bronze / Silver / Gold) |
| MLflow | Experiment tracking, model logging, registry |
| mlxtend | Sequential Feature Selection (forward floating SFS) |
| scikit-learn | Models, preprocessing, cross-validation |
| XGBoost | Gradient boosted trees |
| NumPy / Pandas / SciPy | Signal processing, feature extraction |
| Matplotlib / Seaborn | EDA visualisation |
| pytest | Unit testing for src/functions.py |
| GitHub + Databricks Repos | Version control, CI workflow |

---

## ✅ Progress

- [x] Data ingestion, EDA, Bronze layer (4505 signals across 38 subjects)
- [x] Activity-specific preprocessing pipeline — Silver layer (9033 train, 587 test signals)
- [x] 68-feature extraction + EDA + correlation analysis — Gold layer
- [x] Baseline training — 7 models, binary + multiclass, all runs tracked in MLflow
- [x] Binary feature selection — SFS complete for all 7 models
- [ ] Multiclass feature selection — SFS in progress (model-by-model)
- [ ] Hyperparameter tuning on selected features
- [ ] Final evaluation on unseen test set (SA23, SE06)
- [ ] MLflow Model Registry — register and promote best models

---

## 🔑 Key Design Decisions

**Subject-based train/test split**
SA23 and SE06 held out entirely — never seen during any stage of development. Prevents subject leakage that would inflate reported results.

**Activity-specific preprocessing**
Different signal types require different preprocessing strategies. Walking (100s recordings) needs splitting. Falls need peak-centring. Sit/stand movements need segment separation. A single preprocessing pipeline would corrupt signal quality for most activities.

**Config-driven pipeline**
Zero hardcoded values in notebooks. All parameters live in `configs/config.yaml`. Changing preprocessing parameters, model hyperparameters, or data paths requires editing one file — nothing else.

**Per-model feature selection**
Each model gets its own optimal feature subset via SFS. A single shared subset would be suboptimal — SVM needs 14 features while GradientBoosting needs 43 for equivalent performance.

**Stratified SFS sampling**
Equal samples per activity class during SFS prevents rare classes (lying=290, rising=290) from being underrepresented relative to common classes (walking=1798) during feature evaluation.

**Train/test split preserved at every layer**
Bronze, Silver, and Gold layers all maintain the train/test boundary. Test data flows through the same preprocessing and feature extraction pipeline as train data — but is never used for any fitting, scaling, or selection decisions.
