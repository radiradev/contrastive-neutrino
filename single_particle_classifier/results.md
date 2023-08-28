## Classifiers

Classifer not augmented 
Test Accuracy: 0.8417469263076782
Per-Class Accuracy: tensor([0.8457, 0.7685, 0.8771, 0.4988, 0.9749], device='cuda:0')

Classifer augmented

Test Accuracy: 0.85791015625
Per-Class Accuracy: tensor([0.8664, 0.7997, 0.9068, 0.4444, 0.9855], device='cuda:0')

['Electron', 'Gamma', 'Muon', 'Pion', 'Proton']


## Linear Classifiers (Freeze Network -> Fit a Logistic Regression)

- Fitting on classifier that started to overfit (efficient spaceship-157 v39) - corresponds to validation accuracy
Accuracy:  0.8302612882854437
Balanced Accuracy:  0.7652734569743487

- Fitting on correctly fit classifier augmented - model-iwma6imo:v39

Accuracy:  0.8552849700031575
Balanced Accuracy:  0.7844187112225591



Accuracy:  0.8260380486264604
Balanced Accuracy:  0.7265630444899692


