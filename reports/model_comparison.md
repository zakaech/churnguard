| stage | model | strategy | cv_pr_auc_mean | cv_pr_auc_std | cv_roc_auc_mean | cv_precision_at_recall_mean | val_threshold | val_precision | val_recall | val_f1 | val_pr_auc | val_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_cv_defaut | logistic_regression | none | 0.666 | 0.025 | 0.848 | 0.526 |  |  |  |  |  |  |
| A_cv_defaut | logistic_regression | class_weight | 0.665 | 0.026 | 0.848 | 0.526 |  |  |  |  |  |  |
| A_cv_defaut | logistic_regression | smotenc | 0.651 | 0.030 | 0.840 | 0.517 |  |  |  |  |  |  |
| A_cv_defaut | random_forest | none | 0.622 | 0.005 | 0.823 | 0.476 |  |  |  |  |  |  |
| A_cv_defaut | random_forest | class_weight | 0.616 | 0.010 | 0.825 | 0.483 |  |  |  |  |  |  |
| A_cv_defaut | random_forest | smotenc | 0.601 | 0.009 | 0.819 | 0.470 |  |  |  |  |  |  |
| A_cv_defaut | xgboost | none | 0.619 | 0.016 | 0.820 | 0.474 |  |  |  |  |  |  |
| A_cv_defaut | xgboost | class_weight | 0.624 | 0.008 | 0.818 | 0.466 |  |  |  |  |  |  |
| A_cv_defaut | xgboost | smotenc | 0.620 | 0.013 | 0.822 | 0.476 |  |  |  |  |  |  |
| B_recherche | logistic_regression | none | 0.668 | 0.024 | 0.848 | 0.526 |  |  |  |  |  |  |
| B_recherche | random_forest | none | 0.675 | 0.012 | 0.849 | 0.530 |  |  |  |  |  |  |
| B_recherche | xgboost | class_weight | 0.677 | 0.015 | 0.849 | 0.527 |  |  |  |  |  |  |
| C_validation | logistic_regression | none | 0.668 | 0.024 | 0.848 | 0.526 | 0.243 | 0.499 | 0.802 | 0.615 | 0.643 | 0.837 |
| C_validation | random_forest | none | 0.675 | 0.012 | 0.849 | 0.530 | 0.271 | 0.508 | 0.802 | 0.622 | 0.641 | 0.837 |
| C_validation | xgboost | class_weight | 0.677 | 0.015 | 0.849 | 0.527 | 0.489 | 0.508 | 0.805 | 0.623 | 0.647 | 0.840 |
