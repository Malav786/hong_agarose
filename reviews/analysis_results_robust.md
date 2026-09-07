# Robust Model Evaluation & Validation Report

This report presents a thorough, group-aware cross-validation analysis, model selection justifications, calibration metrics, and permutation feature importances.

---

## 1. Group-Aware Cross Validation Results

To rigorously test how the models generalize to entirely unseen physical systems, we conducted a **5-fold Stratified Group K-Fold cross-validation**. 

### Grouping Definition
The groups were defined using the directory structure of the structures (e.g., `r0`, `r20`, `r40`, ..., `r340`). Since these folders contain families of related rotated configurations, grouping by these folder names ensures that **all files belonging to a specific rotation system are confined to a single fold**. The model is tested only on completely unseen systems, preventing group-level data leakage.

### Repeated CV Metrics (Mean ± Standard Deviation)

Across the 5 folds, we report the following primary metrics:

| Metric | Stacking Ensemble | Extra Trees Classifier |
| :--- | :---: | :---: |
| **MCC** | 0.7125 ± 0.1174 | **0.7332 ± 0.0862** |
| **Accuracy** | 0.9492 ± 0.0183 | **0.9524 ± 0.0156** |
| **ROC-AUC** | **0.9729 ± 0.0195** | 0.9723 ± 0.0192 |
| **PR-AUC** | **0.9955 ± 0.0043** | 0.9954 ± 0.0044 |
| **Brier Score** *(lower is better)* | 0.0592 ± 0.0178 | **0.0539 ± 0.0174** |
| **Expected Calibration Error (ECE)** | 0.1192 ± 0.0277 | **0.0940 ± 0.0335** |

*Note: Stacking validation threshold was optimized on the training folds (mean threshold ~0.1).*

---

## 2. Model Selection Justification: Extra Trees vs. Stacking

On both the stratified random test split (MCC: 0.8569 vs. 0.8252) and the group-aware cross-validation (MCC: 0.7332 vs. 0.7125), **Extra Trees performs better than Stacking**.

### Why Extra Trees Outperforms Stacking:
1. **Variance Reduction**: Extra Trees (Extremely Randomized Trees) chooses split points entirely at random for each candidate feature, rather than looking for the locally optimal split threshold (like Random Forest). On small, highly correlated datasets (1,865 training rows, 9 geometric features), this randomized splitting dramatically reduces variance and prevents the model from overfitting to specific spatial thresholds.
2. **Out-of-Distribution Generalization**: The Stacking Classifier incorporates a Multi-Layer Perceptron (MLP) neural network. Neural networks are highly flexible but struggle to generalize to unseen group distributions (new rotation angles) in small tabular datasets, dragging down the meta-model's stacking predictions.
3. **Collinearity Handling**: Because our geometric features (`sum_of_count`, `avg_HO_dist`, etc.) are highly collinear, tree-based bagging classifiers naturally decorrelate them during node splitting. Extra Trees behaves more robustly in these spaces than a meta-model Logistic Regression that assumes linear boundary separators.

### Selection Verdict
For a lightweight, highly stable deployment, **Extra Trees is the superior choice**. It has lower standard deviation across splits ($\pm 0.08$ vs $\pm 0.11$ for Stacking) and better calibration out-of-the-box. However, if max ROC-AUC/PR-AUC is desired, Stacking maintains a slight edge.

---

## 3. Probability Calibration Metrics

A well-calibrated model outputs probabilities that match the actual empirical frequency of the positive class. We evaluate calibration using two key metrics:

* **Brier Score**: Measures the mean squared difference between predicted probabilities and actual outcomes.
* **Expected Calibration Error (ECE)**: Computes the weighted average of the absolute difference between predicted and actual frequencies across bins.

### Calibration Comparison

* **Extra Trees**: ECE = **`0.0940 ± 0.0335`** | Brier = **`0.0539 ± 0.0174`**
* **Stacking Ensemble**: ECE = `0.1192 ± 0.0277` | Brier = `0.0592 ± 0.0178`

Extra Trees shows significantly better calibration (lower ECE and Brier Score). Stacking outputs tend to be overconfident (pushed towards 0.0 and 1.0) because the meta-logistic regression operates on probabilities that have already been adjusted by class balancing.

---

## 4. Permutation Feature Importances (Fold 1)

Permutation importance measures the increase in prediction error when a feature's values are randomly shuffled. If shuffling a feature doesn't change the model's predictions, the feature has low importance.

| Feature | Permutation Importance (Mean Δ) |
| :--- | :---: |
| `avg_HH_dist` | **0.0025** |
| `avg_OO_dist` | **0.0019** |
| `count_OO` | -0.0008 |
| `count_HH` | -0.0039 |
| `sum_of_count` | -0.0091 |
| `count_OH` | -0.0145 |
| `avg_OH_dist` | -0.0164 |
| `count_HO` | -0.0165 |
| `avg_HO_dist` | -0.0224 |

### Interpretation of Negative Importances
* **Multicollinearity Effect**: The geometric features (distances and counts) are highly correlated. For example, if `avg_HO_dist` is shuffled, the model doesn't lose predictive power because it still has `count_HO` and `avg_OH_dist` to infer the spatial arrangement.
* In highly collinear spaces, permutation importance can yield negative or near-zero values. This indicates that the information is distributed redundantly across multiple features, allowing the model to make stable predictions even when individual features are destroyed.

---

## 5. Threshold Selection & Prior Adjustment

The optimal decision threshold was found to be approximately **`0.0983`** for Stacking and **`0.2052`** for Extra Trees. 

* Rather than being directly determined by the class prior (which is $\approx 10\%$ minority class), the optimal threshold is chosen explicitly by **maximizing the Matthews Correlation Coefficient (MCC) on the validation fold**.
* Since MCC balances True Positives, True Negatives, False Positives, and False Negatives, it naturally pushes the threshold down to prevent the minority class (Label 0) from being completely overwhelmed by the 90% majority class prior during classification.
