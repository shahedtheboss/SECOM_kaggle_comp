# Semiconductor Defect Detection Using Machine Learning

A machine learning project exploring sensor-based semiconductor defect detection using the SECOM dataset, with an emphasis on handling missing data, class imbalance, feature selection, model evaluation, and experimentation.

## Project Overview

Semiconductor manufacturing involves monitoring numerous process variables and sensor measurements. Identifying defective products from these measurements is a challenging binary classification problem, particularly when the dataset contains missing values and a highly imbalanced target.

In this project, I experimented with multiple machine learning algorithms to distinguish defective samples from normal samples and investigated how preprocessing and feature selection affect predictive performance.

**Current best public leaderboard ROC-AUC: 0.79887**

## Dataset

The project uses the SECOM semiconductor manufacturing dataset.

| Property | Details |
|---|---|
| Training samples | 1,253 |
| Test samples | 314 |
| Sensor features | 590 |
| Target type | Binary classification |
| Normal samples | 1,170 |
| Defective samples | 83 |
| Defective class proportion | Approximately 6.6% |

The dataset presents several challenges:

- **Class imbalance:** Defective samples make up only a small fraction of the labeled data.
- **Missing values:** Many sensor measurements are incomplete.
- **High-dimensional input:** Hundreds of sensor variables are available for a relatively small number of samples.
- **Feature relevance:** Not every sensor necessarily contributes useful predictive information.

## Machine Learning Workflow

The project follows an iterative experimental workflow.

1. **Data preprocessing**
   - Separated identifiers, timestamps, sensor measurements, and target labels.
   - Converted the original target labels into binary classes, with defective samples represented by `1` and normal samples by `0`.
   - Used median imputation to handle missing sensor measurements.

2. **Feature selection and engineering**
   - Removed constant features in selected experiments.
   - Compared different missing-value filtering thresholds.
   - Investigated engineered timestamp features, including elapsed time, hour, and day of the week.
   - Conducted feature ablation experiments to compare raw sensor measurements against filtered feature sets.

3. **Model development**
   - Trained and compared multiple classification algorithms.
   - Evaluated Extra Trees as a strong candidate for the final prediction model.
   - Tested rank-based blending of predictions from multiple models.

4. **Model evaluation**
   - Used stratified cross-validation to preserve class proportions across folds.
   - Evaluated models using ROC-AUC.
   - Compared fold-level performance to understand variability and avoid relying solely on a single split.

5. **Kaggle submission**
   - Trained the selected model on the full labeled training dataset.
   - Generated predictions for the unlabeled test dataset.
   - Submitted predictions to the public leaderboard.

## Models Explored

| Model | Purpose |
|---|---|
| Extra Trees Classifier | Main candidate and strongest observed individual model |
| Random Forest Classifier | Tree-based model comparison |
| HistGradientBoosting Classifier | Gradient-boosting comparison |
| Logistic Regression | Linear baseline with quantile transformation and feature selection |
| Rank-based ensemble | Combined predictions from multiple models |

## Best Model

The strongest observed public leaderboard result came from an **Extra Trees Classifier**.

### Configuration

| Parameter | Value |
|---|---|
| Number of trees | 1,000 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `max_features` | `"sqrt"` |
| `class_weight` | `None` |
| `random_state` | 42 |
| Missing-value strategy | Median imputation |

The model was trained using the filtered sensor feature set with engineered timestamp features.

### Results

| Experiment | Mean CV ROC-AUC | Standard deviation |
|---|---:|---:|
| Extra Trees — all 590 raw sensors | 0.7227 | 0.0507 |
| Extra Trees — filtered sensors only | 0.7122 | 0.0436 |
| Extra Trees — filtered sensors plus time features | 0.7204 | 0.0494 |
| Extra Trees — alternate missingness threshold of 60% | 0.7172 | 0.0595 |

**Public leaderboard milestone:** The Extra Trees submission achieved a ROC-AUC of **0.79887**, reaching 3rd place on the public leaderboard at the time of the result.

*Note:* Cross-validation and public leaderboard scores are measured on different samples and should not be compared as if they were the same evaluation. Small differences in cross-validation means may also reflect sampling variability.

## Key Findings

### 1. A single model can outperform an ensemble

The rank-based ensemble did not outperform the strongest individual Extra Trees submission on the public leaderboard. This reinforced the importance of validating whether combining models actually adds useful predictive information.

### 2. Feature filtering requires empirical validation

Removing features with high missingness did not consistently improve cross-validation performance. The all-sensor experiment achieved a mean ROC-AUC of 0.7227, while the filtered sensor-plus-time experiment achieved 0.7204.

The small difference is not enough to establish that one feature strategy is universally superior.

### 3. Class imbalance affects the problem

With only about 6.6% defective samples, accuracy alone would be a poor measure of performance. ROC-AUC provides a more informative view of how well the model ranks defective samples above normal samples.

### 4. Validation methodology matters

Cross-validation scores should be calculated separately for each validation fold and then averaged. Pooling predictions that have been independently rank-transformed within different folds can distort the overall ROC-AUC estimate.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- Jupyter Notebook / Google Colab
- Kaggle

## Repository Structure

A suggested structure for this repository:

```text
semiconductor-defect-detection/
├── README.md
├── notebooks/
│   └── secom_experiments.ipynb
├── submissions/
│   └── submission_et.csv
└── requirements.txt
```

The notebook and submission filenames are illustrative; adjust them to match the files actually included in your repository.

## Running the Project

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd semiconductor-defect-detection
```

### 2. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn jupyter
```

### 3. Prepare the dataset

Download the dataset from the relevant Kaggle competition and place the files in the location expected by your notebook.

The training data should contain the target column, while the test data should contain the same input features without the target.

### 4. Run the notebook

Open the notebook in Jupyter or Google Colab and execute the cells in order to reproduce the preprocessing, model experiments, and predictions.

## Future Improvements

Potential next steps include:

- Repeating cross-validation across multiple random seeds.
- Investigating feature importance and sensor-level contributions.
- Comparing additional models and carefully validated ensembles.
- Exploring methods designed for imbalanced classification.
- Improving probability calibration and threshold selection where appropriate.
- Evaluating model stability and generalization beyond the public leaderboard.

## What I Learned

This project helped me practice the complete machine learning experimentation cycle: data preprocessing, handling missing values, feature selection, model comparison, cross-validation, leaderboard evaluation, and iterative improvement.

It also demonstrated that a strong machine learning solution is not necessarily the one with the most features or the most complicated ensemble. Careful experimentation and reliable validation are essential.

## Acknowledgements

- The SECOM semiconductor manufacturing dataset and Kaggle competition.
- The open-source Python machine learning ecosystem, particularly scikit-learn.

---

**Status:** Ongoing — experimenting with feature selection, model validation, and generalization.
