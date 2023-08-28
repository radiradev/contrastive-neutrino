## Classifiers

| Classifier | Test Accuracy | Electron | Gamma | Muon | Pion | Proton |
| --- | --- | --- | --- | --- | --- | --- |
| Not Augmented | 0.8417 | 0.8457 | 0.7685 | 0.8771 | 0.4988 | 0.9749 |
| Augmented | 0.8579 | 0.8664 | 0.7997 | 0.9068 | 0.4444 | 0.9855 |

## Linear Classifiers (Freeze Network -> Fit a Logistic Regression )

| Classifier | Balanced Accuracy | Accuracy |
| --- | --- | --- | 
| Starting to overfit `efficient spaceship-157 v39` | 0.830 | 0.765 |
| Augmented `model-iwma6imo:v39` | 0.855 | 0.784 |
| Nominal | 0.826 | 0.726
