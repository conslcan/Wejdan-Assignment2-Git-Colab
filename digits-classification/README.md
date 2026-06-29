# Handwritten Digit Classification

Image classification project comparing Random Forest and SVM on the sklearn digits dataset 
(1,797 images of handwritten digits 0-9, 8x8 pixels each).

## Contents
- `Wejdan_digits_classification.ipynb` — main analysis notebook
- `figures/` — confusion matrix, feature importance heatmap, model comparison chart
- `report/` — 2-page PDF and DOCX summary report

## Summary
Random Forest (tuned via GridSearchCV over 108 parameter combinations) achieved 96.39% test accuracy. 
SVM (tuned via GridSearchCV with feature standardization) achieved 97.50% test accuracy, outperforming 
Random Forest by ~1.1 percentage points across accuracy, precision, recall, and F1-score.
