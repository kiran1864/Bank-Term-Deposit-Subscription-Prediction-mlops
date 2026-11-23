# Model Comparison

| Model | Size MB | Train time s | Predict ms | Train acc % | Val acc % | Gap % | Eligible | Status |
|---|---|---|---|---|---|---|---|---|
| KNN | 3.4795 | 0.0427 | 7.4621 | 74.2 | 70.68 | 3.52 | Yes | eligible_but_below_accuracy_tolerance |
| Decision Tree | 0.0423 | 0.043 | 5.269 | 72.68 | 69.47 | 3.21 | Yes | eligible_but_below_accuracy_tolerance |
| SVM | 2.2283 | 2.9191 | 4.6835 | 75.66 | 73.28 | 2.39 | Yes | champion |
| Gaussian NB | 0.0069 | 0.0411 | 4.6829 | 68.73 | 68.51 | 0.22 | Yes | eligible_but_below_accuracy_tolerance |
| Random Forest | 23.3289 | 1.6222 | 66.5097 | 77.79 | 73.04 | 4.75 | Yes | finalist_runner_up |
| Gradient Boosting | 1.2305 | 4.4967 | 5.0613 | 77.28 | 73.92 | 3.36 | Yes | finalist_runner_up |

**Champion:** SVM
