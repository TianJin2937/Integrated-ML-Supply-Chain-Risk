# Integrated Machine Learning for Enhanced Supply Chain Risk Prediction

**Paper**: [ACM EITCE 2024](https://dl.acm.org/doi/10.1145/3711129.3711341) | 23 Citations

## Abstract

Supply chain risk prediction has become increasingly critical as organizations navigate complex and volatile environments. This study proposes an innovative integrated model that combines Random Forest, Gradient Boosting Machine (GBM), and Neural Networks to enhance prediction accuracy and reliability in supply chain risk assessment. By employing comprehensive data preprocessing techniques alongside advanced algorithmic strategies, the model effectively addresses the limitations of traditional approaches.

## Dataset

[DataCo Global Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) — Real-world supply chain data covering orders, shipping, and customer information.

## Methods

- **Random Forest** — Optimized tree pruning with entropy reduction, dynamic feature importance weighting
- **Gradient Boosting Machine (GBM)** — Adaptive learning rate decay, stochastic gradient boosting, L2 regularization
- **Neural Network** — Swish activation, batch normalization, dropout, multi-head attention mechanism
- **Ensemble** — Weighted averaging with grid-search optimized weights

## Repository Structure

```
├── src/
│   ├── preprocessing.py       # Data cleaning, normalization, outlier detection
│   ├── random_forest.py       # Random Forest with optimized pruning
│   ├── gbm_model.py           # GBM with adaptive learning rate
│   ├── neural_network.py      # NN with multi-head attention
│   ├── ensemble.py            # Weighted ensemble combination
│   └── evaluate.py            # Evaluation metrics
├── notebooks/
│   └── full_pipeline.ipynb    # End-to-end training pipeline
├── requirements.txt
└── README.md
```

## Results

| Model | Accuracy (%) | F1 Score | MSE |
|-------|-------------|----------|-----|
| Logistic Regression | 76.5 | 0.74 | 0.024 |
| SVM | 78.2 | 0.76 | 0.022 |
| Random Forest | 82.5 | 0.81 | 0.019 |
| XGBoost | 84.1 | 0.82 | 0.017 |
| Deep Neural Network | 83.7 | 0.83 | 0.018 |
| **Proposed Ensemble** | **85.4** | **0.85** | **0.015** |

## Citation

```bibtex
@inproceedings{jin2024integrated,
  title={Integrated Machine Learning for Enhanced Supply Chain Risk Prediction},
  author={Jin, Tian},
  booktitle={Proceedings of the 2024 8th International Conference on Electronic Information Technology and Computer Engineering (EITCE)},
  pages={1254--1259},
  year={2024},
  organization={ACM},
  doi={10.1145/3711129.3711341}
}
```

## Requirements

```
numpy
pandas
scikit-learn
xgboost
tensorflow>=2.6
matplotlib
seaborn
```

## License

MIT
