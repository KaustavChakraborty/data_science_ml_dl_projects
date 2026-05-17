# Observations: PyTorch MLP Training Lab

This document records the main observations, interpretations, and learning outcomes from the project containing:

```text
training_loop.py
experiments.py
```

The project studies a PyTorch multilayer perceptron trained on the UCI Red Wine Quality dataset. The original wine-quality score is converted into a binary classification problem:

```text
quality >= 7  →  good wine      → label 1
quality < 7   →  ordinary wine  → label 0
```

The project has two main purposes:

1. To understand the complete neural-network training pipeline.
2. To understand how important hyperparameters change training behavior.

The most important lesson from the project is:

> A neural network should not be judged only by final accuracy. It should be understood through learning curves, validation loss, class-wise metrics, confusion matrix, overfitting behavior, and hyperparameter sensitivity.

---

# 1. High-Level Summary of the Project

This project is a compact deep-learning laboratory for tabular binary classification.

The first script, `training_loop_v1.py`, demonstrates a full end-to-end workflow:

```text
data loading
→ train/validation/test splitting
→ train-only scaling
→ tensor conversion
→ DataLoader construction
→ model definition
→ loss function
→ optimizer
→ training loop
→ validation loop
→ learning-curve plotting
→ final test evaluation
→ classification report
→ confusion matrix
→ model saving
→ model reloading
→ inference on a new sample
```

The second script, `experiments_v1.py`, performs controlled hyperparameter experiments:

```text
Experiment A: network depth
Experiment B: network width
Experiment C: learning rate
Experiment D: activation function
Experiment E: batch size
```

Together, the two scripts show both the mechanics of training and the experimental reasoning needed to understand neural-network behavior.

---

# 2. Dataset Observations

The dataset loaded successfully:

```text
Loaded from URL: (1599, 12)
```

This means the red wine dataset contains:

```text
1599 samples
12 columns
```

The 12 columns consist of:

```text
11 physicochemical input features
1 original wine-quality score
```

After creating the binary target label, the model uses:

```text
Input dimension  = 11
Output dimension = 1
```

The output is interpreted as:

```text
P(good wine)
```

The binary label is created as:

```python
df["label"] = (df["quality"] >= 7).astype(int)
```

This creates the following classification task:

```text
ordinary wine: quality < 7
good wine    : quality >= 7
```

---

# 3. Class Imbalance Observation

One of the most important observations is that the dataset is imbalanced.

From the training run:

```text
Positive rate — Train: 13.40%
Positive rate — Val  : 15.06%
Positive rate — Test : 12.86%
```

This means only around 13–15% of the samples are labeled as `good wine`.

So the majority class is:

```text
ordinary wine
```

The minority class is:

```text
good wine
```

This has a major consequence:

> Accuracy can look high even if the model mostly predicts ordinary wine.

For example, in the test set:

```text
ordinary wines = 210
good wines     = 31
total samples  = 241
```

A naive model that predicts every test sample as ordinary would achieve:

```text
210 / 241 = 87.14% accuracy
```

Therefore, a model with around 88–90% accuracy is not automatically excellent. It must be checked using class-wise precision, recall, F1-score, and the confusion matrix.

---

# 4. Data Splitting Observation

The dataset is split into:

```text
Train: 1119 samples
Val  :  239 samples
Test :  241 samples
```

This corresponds approximately to:

```text
70% training
15% validation
15% testing
```

This is a good practical split for a small dataset.

The training set is used to fit the model.

The validation set is used to monitor generalization during training.

The test set is used only after training to estimate final performance.

---

# 5. Preprocessing Observation: Correct Train-Only Scaling

The project correctly avoids data leakage by fitting the scaler only on the training set.

Correct procedure:

```text
split first
fit scaler on training data only
transform train/validation/test using the training scaler
```

This is important because the validation and test sets must behave like unseen data.

If the scaler were fitted on the entire dataset before splitting, information from the validation and test sets would leak into the training process.

The project correctly does:

```python
scaler.fit(X_train_np)
X_train = scaler.transform(X_train_np)
X_val   = scaler.transform(X_val_np)
X_test  = scaler.transform(X_test_np)
```

This is one of the best-practice parts of the project.

---

# 6. Device Observation: CPU Used Because CUDA Driver Was Too Old

The script printed a CUDA warning:

```text
CUDA initialization: The NVIDIA driver on your system is too old
```

Then it selected:

```text
Device: cpu
```

This means PyTorch could not use the GPU because the NVIDIA driver was incompatible with the installed PyTorch CUDA build.

This is not an error in the model or code.

The dataset is small, so CPU training is perfectly acceptable here.

Observed training times were only a few seconds per experiment, so GPU acceleration is not necessary for this specific project.

---

# 7. Model Architecture Observation

The default model in `training_loop_v1.py` is:

```text
11 → 64 → ReLU → 32 → ReLU → 1 → Sigmoid
```

This means:

```text
Input layer       : 11 features
Hidden layer 1    : 64 neurons
Activation 1      : ReLU
Hidden layer 2    : 32 neurons
Activation 2      : ReLU
Output layer      : 1 neuron
Output activation : Sigmoid
```

The model has:

```text
Trainable parameters: 2,881
```

Parameter breakdown:

```text
Layer 1: 11 × 64 + 64 = 768
Layer 2: 64 × 32 + 32 = 2080
Layer 3: 32 × 1  + 1  = 33
Total  = 768 + 2080 + 33 = 2881
```

This is a small model.

The size is appropriate for a dataset with only 1119 training samples.

---

# 8. Loss Function Observation

The project uses:

```python
criterion = nn.BCELoss()
```

This is binary cross-entropy loss.

The model output is passed through a sigmoid, so the output is already between 0 and 1.

Binary cross-entropy is:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

where:

```text
y = true label
p = predicted probability of good wine
```

Important observation:

> Binary cross-entropy measures both correctness and confidence.

This is why validation loss can increase even when validation accuracy stays nearly constant.

A model may classify the same number of samples correctly but become more confident on wrong predictions. In that case, accuracy stays similar, but loss increases.

---

# 9. Optimizer Observation

The project uses:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Adam with learning rate 0.001 is a strong default choice.

Adam adapts the update size for each parameter using gradient statistics.

The learning-rate experiment confirms that 0.001 is a stable choice for this project.

---

# 10. Training Loop Observation

Each training batch follows the standard PyTorch pattern:

```python
predictions = model(X_batch)
loss = criterion(predictions, y_batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This is the core of deep learning.

The observations are:

```text
forward pass computes predictions
loss compares predictions with true labels
zero_grad clears old gradients
backward computes new gradients
step updates model parameters
```

The project correctly uses:

```python
model.train()
```

during training and:

```python
model.eval()
```

during validation and testing.

Even though this model does not use dropout or batch normalization, using `train()` and `eval()` correctly is good practice.

---

# 11. Main Training-Loop Results

The training run gave:

```text
Epoch   Train Loss   Val Loss    Val Acc  Train Acc
----------------------------------------------------
1       0.5845       0.4904      84.94%   84.45%
10      0.2544       0.2900      88.28%   89.01%
20      0.2306       0.2885      88.70%   89.81%
30      0.2096       0.2856      89.54%   90.71%
40      0.1842       0.2872      90.79%   92.14%
50      0.1597       0.2992      90.79%   93.03%
60      0.1370       0.3262      89.96%   94.19%
70      0.1187       0.3517      90.38%   95.17%
80      0.0988       0.3860      90.38%   96.43%
```

Key observations:

```text
training loss decreases continuously
training accuracy increases continuously
validation loss improves early, then worsens
validation accuracy improves slightly, then plateaus
```

This is a classic overfitting pattern.

---

# 12. Overfitting Observation

The best validation loss occurred at:

```text
Best validation epoch: 29
val_loss = 0.2832
```

After this point:

```text
training loss continues decreasing
validation loss starts increasing
```

This means the model continues to fit the training data, but generalization begins to degrade.

The final epoch is not necessarily the best model.

Important conclusion:

> The model should use early stopping or save the best validation-loss checkpoint.

Current behavior:

```text
final epoch model is used
```

Better behavior:

```text
best validation-loss model should be saved and used
```

---

# 13. Learning Curve Observation

The learning curve shows two major behaviors.

## Training Loss

Training loss decreases smoothly from about:

```text
0.58 → 0.10
```

This means the model is learning the training data very well.

## Validation Loss

Validation loss decreases initially but later increases:

```text
0.49 → 0.28 → 0.39
```

This means the model first learns generalizable patterns, then starts overfitting.

## Training Accuracy

Training accuracy rises to:

```text
96.43%
```

This is much higher than validation accuracy.

## Validation Accuracy

Validation accuracy saturates around:

```text
90%
```

This gap between training and validation performance is another sign of overfitting.

---

# 14. Test Evaluation Observation

The final test evaluation gave:

```text
Test Loss     : 0.4772
Test Accuracy : 88.80%
```

At first glance, 88.80% accuracy seems good.

However, because the dataset is imbalanced, this number must be interpreted carefully.

The majority-class baseline on the test set is approximately:

```text
87.14%
```

So the model improves only modestly over the naive majority-class predictor in terms of raw accuracy.

The more useful information comes from the classification report and confusion matrix.

---

# 15. Classification Report Observation

The classification report was:

```text
              precision    recall  f1-score   support

ordinary       0.92      0.96      0.94       210
good wine      0.59      0.42      0.49        31

accuracy                           0.89       241
macro avg      0.75      0.69      0.71       241
weighted avg   0.88      0.89      0.88       241
```

Important observations:

## Ordinary wine performance is strong

For ordinary wine:

```text
precision = 0.92
recall    = 0.96
F1-score  = 0.94
support   = 210
```

The model identifies ordinary wines well.

## Good-wine performance is weak

For good wine:

```text
precision = 0.59
recall    = 0.42
F1-score  = 0.49
support   = 31
```

This is the main weakness.

The model catches only:

```text
42%
```

of truly good wines.

That means it misses more than half of the good wines.

---

# 16. Confusion Matrix Observation

The confusion matrix was:

```text
                    Predicted ordinary    Predicted good
Actual ordinary            201                  9
Actual good                 18                 13
```

This gives:

```text
True negatives  = 201
False positives = 9
False negatives = 18
True positives  = 13
```

Interpretation:

```text
201 ordinary wines were correctly classified as ordinary
9 ordinary wines were incorrectly classified as good
18 good wines were incorrectly classified as ordinary
13 good wines were correctly classified as good
```

The model is conservative.

It tends to predict ordinary wine more often than good wine.

This is expected because the training data is dominated by ordinary wines.

---

# 17. Minority-Class Recall Observation

Good-wine recall is:

```text
13 / (13 + 18) = 13 / 31 = 0.419
```

So the model catches about:

```text
41.9%
```

of truly good wines.

This is the most important weakness of the trained model.

A model with high accuracy but low minority-class recall is not ideal if the goal is to find good wines.

Potential fixes:

```text
class weighting
threshold tuning
oversampling good wines
undersampling ordinary wines
PR-AUC optimization
focal loss
balanced batch sampling
```

---

# 18. Model Saving and Reloading Observation

The script saved the model as:

```text
wine_mlp.pth
```

Then it reloaded the model and compared predictions.

The result was:

```text
Identical: True
```

This confirms that:

```text
model.state_dict() was saved correctly
model.state_dict() was loaded correctly
the reloaded model reproduces the original model output
```

This is an important production-readiness check.

---

# 19. Inference Observation

The script tested a new wine sample:

```text
[[7.8, 0.58, 0.02, 2.0, 0.073, 9.0, 18.0, 0.9968, 3.36, 0.57, 9.5]]
```

The model predicted:

```text
P(good wine) = 0.2355
Prediction   = ordinary wine
```

Because:

```text
0.2355 < 0.5
```

the model classified the sample as ordinary.

Important inference observation:

> The new sample was correctly scaled using the same scaler fitted on the training data.

This is essential.

For real inference, one must never fit a new scaler on the new sample.

---

# 20. Experiment A Observation: Network Depth

Experiment A tested:

```text
1 hidden  [64]
2 hidden  [64,32]
3 hidden  [64,64,32]
4 hidden  [64,64,32,16]
6 hidden  [64,64,32,32,16,8]
```

Final validation accuracies:

```text
1 hidden  [64]                  89.96%
2 hidden  [64,32]               89.96%
3 hidden  [64,64,32]            89.54%
4 hidden  [64,64,32,16]         89.96%
6 hidden  [64,64,32,32,16,8]    89.54%
```

Main observation:

> Increasing depth did not improve validation accuracy.

The shallow models performed as well as the deeper models.

The deeper models showed worse validation-loss behavior.

Especially the 6-hidden-layer network showed a strongly increasing validation loss.

This indicates overfitting or overconfidence.

Conclusion:

```text
For this small tabular dataset, 1–2 hidden layers are enough.
```

---

# 21. Why Deeper Networks Did Not Help

Deep networks are useful when the data contains hierarchical structure, such as:

```text
images
language
audio
large molecular graphs
large simulation fields
```

This project uses:

```text
11 tabular features
1119 training samples
binary labels
```

This is not enough data or complexity to justify a deep network.

More depth increases capacity, but that capacity is not useful here.

Instead, it increases the risk of overfitting.

---

# 22. Experiment B Observation: Network Width

Experiment B tested:

```text
width=8    [8,8]
width=32   [32,32]
width=64   [64,64]
width=128  [128,128]
width=512  [512,512]
```

Final validation accuracies:

```text
width=8      85.77%
width=32     90.79%
width=64     89.96%
width=128    90.79%
width=512    91.21%
```

Parameter counts:

```text
width=8      177 parameters
width=32     1,473 parameters
width=64     4,993 parameters
width=128    18,177 parameters
width=512    269,313 parameters
```

Main observation:

> Width helps up to a point, but very large width can overfit.

The width=8 model underfits.

The width=32 and width=128 models are strong practical choices.

The width=512 model gives the highest final validation accuracy, but its validation loss becomes very large later.

This indicates overconfidence and overfitting.

---

# 23. Width Experiment Interpretation

The width experiment shows three regimes:

## Too narrow

```text
width=8
```

The model does not have enough capacity.

It underfits.

## Moderate width

```text
width=32
width=64
width=128
```

The model has enough capacity to learn useful patterns.

This is the best practical region.

## Too wide

```text
width=512
```

The model has enormous capacity relative to the dataset size.

It may fit training-specific patterns and become overconfident.

Even if accuracy is slightly higher, the loss curve suggests reduced reliability.

---

# 24. Experiment C Observation: Learning Rate

Experiment C tested:

```text
lr=1.0
lr=0.1
lr=0.01
lr=0.001
lr=0.0001
```

Final validation accuracies:

```text
lr=1.0      84.94%
lr=0.1      84.94%
lr=0.01     89.96%
lr=0.001    89.96%
lr=0.0001   88.70%
```

Main observation:

> Learning rate strongly controls whether training succeeds or fails.

High learning rates failed.

Moderate learning rates worked.

Very small learning rate learned slowly.

---

# 25. Learning Rate Interpretation

## lr = 1.0

This is too high.

The optimizer takes huge steps and cannot settle into a good solution.

The model remains close to majority-class behavior.

## lr = 0.1

Still too high.

The validation loss is unstable and the final accuracy is again near the majority-class baseline.

## lr = 0.01

This learns quickly and reaches good validation accuracy.

However, it can be slightly more unstable than 0.001.

## lr = 0.001

This is the safest default.

It gives stable learning and good accuracy.

## lr = 0.0001

This is too slow for 80 epochs.

It moves in the right direction but does not fully converge within the chosen epoch budget.

---

# 26. Experiment D Observation: Activation Function

Experiment D tested:

```text
ReLU
LeakyReLU
Tanh
Sigmoid
ELU
```

Final validation accuracies:

```text
ReLU       89.96%
LeakyReLU  89.96%
Tanh       89.96%
Sigmoid    87.45%
ELU        90.38%
```

Main observation:

> Sigmoid hidden-layer activation performed worst.

ReLU, LeakyReLU, Tanh, and ELU all worked reasonably well.

ELU gave the best final validation accuracy, but the improvement over ReLU was small.

---

# 27. Why Sigmoid Performed Worse

Sigmoid hidden activations can saturate.

The sigmoid function becomes flat for large positive or negative inputs.

When the function is flat, gradients are small.

Small gradients make learning slow.

This is the vanishing-gradient problem.

Important distinction:

```text
Sigmoid at output layer for binary classification: acceptable
Sigmoid inside hidden layers: often not preferred
```

The project uses sigmoid at the output in `training_loop_v1.py`, which is appropriate with `BCELoss`.

But Experiment D shows that sigmoid hidden layers are weaker.

---

# 28. Experiment E Observation: Batch Size

Experiment E tested:

```text
batch=8
batch=32
batch=128
batch=512
```

Final validation accuracies:

```text
batch=8      90.79%
batch=32     89.96%
batch=128    92.05%
batch=512    89.54%
```

Number of weight updates:

```text
batch=8      11120 updates
batch=32     2720 updates
batch=128    640 updates
batch=512    160 updates
```

Main observation:

> Batch size changes both optimization dynamics and generalization behavior.

Batch=128 gave the best final validation accuracy in this run.

Batch=8 showed noisy and overfitting-like validation-loss behavior.

Batch=512 had too few updates and performed worse.

---

# 29. Batch Size Interpretation

## Small batch: batch=8

Small batches create noisy gradient estimates.

Advantages:

```text
many updates
can explore the loss landscape
can escape some poor regions
```

Disadvantages:

```text
noisy curves
possible instability
possible overfitting after many updates
```

In this project, batch=8 obtained good validation accuracy but validation loss increased strongly later.

## Standard batch: batch=32

This is a safe and common default.

It gives a balance between noise and stability.

## Larger batch: batch=128

This performed best in final validation accuracy.

It likely provided a smoother gradient estimate while still allowing enough updates.

## Near-full batch: batch=512

This had very few updates.

The gradients were smooth, but learning was less effective.

This shows that very large batches are not always better.

---

# 30. Final Hyperparameter Observations

Summary of best final validation accuracies:

```text
Depth        : 1 hidden [64] or 2 hidden [64,32] or 4 hidden [64,64,32,16]
Width        : width=512
Learning rate: lr=0.01 or lr=0.001
Activation   : ELU
Batch size   : batch=128
```

However, the best final number is not always the best practical model.

Important caution:

> Do not blindly combine all individual winners.

For example, width=512 gives high validation accuracy but shows poor validation-loss behavior.

A more practical model would be:

```python
hidden_sizes = [64, 32]
activation   = nn.ReLU or nn.ELU
lr           = 0.001
batch_size   = 32 or 128
```

with early stopping.

---

# 31. Important Observation: Final Accuracy Is Not Enough

Many plots show cases where:

```text
validation accuracy is acceptable
but validation loss gets worse
```

This means the model is becoming more confident, especially on wrong predictions.

Accuracy only checks whether:

```text
p >= 0.5
```

or:

```text
p < 0.5
```

Loss checks how good the probability values are.

Therefore:

```text
validation accuracy = correctness after thresholding
validation loss     = correctness + confidence quality
```

A model with slightly lower accuracy but much better validation loss may be more reliable.

---

# 32. Important Observation: Small Validation Set Causes Quantized Accuracy

The validation set has:

```text
239 samples
```

One sample corresponds to:

```text
1 / 239 = 0.00418 = 0.418%
```

So changes such as:

```text
89.96% → 90.38%
```

represent only about one sample.

Therefore, very small accuracy differences should not be over-interpreted.

A difference of 0.4% on this validation set is not strong evidence that one model is truly better.

Learning curves and repeated runs with different seeds would provide more reliable conclusions.

---

# 33. Important Observation: Majority-Class Baseline

The repeated value:

```text
84.94%
```

appears for bad learning-rate settings.

This is approximately the majority-class baseline on the validation set.

That means those models likely learned a trivial solution:

```text
predict ordinary wine for almost everything
```

This is why class imbalance must always be considered.

A model that reaches 84.94% validation accuracy is not necessarily learning useful minority-class detection.

---

# 34. Best Practical Model Suggested by Observations

Based on all observations, a reasonable practical configuration is:

```python
hidden_sizes = [64, 32]
activation   = nn.ELU
learning_rate = 0.001
batch_size    = 128
epochs        = use early stopping
```

Another safer and simpler configuration is:

```python
hidden_sizes = [64, 32]
activation   = nn.ReLU
learning_rate = 0.001
batch_size    = 32
epochs        = use early stopping
```

The second option is more standard.

The first option is slightly more informed by the experiment results.

---

# 35. Recommended Improvement: Early Stopping

The clearest improvement is early stopping.

The training-loop result showed:

```text
best validation epoch = 29
```

but training continued until epoch 80.

A better approach is:

```text
monitor validation loss
save model whenever validation loss improves
stop training if validation loss does not improve for several epochs
reload the best model
```

Expected benefit:

```text
less overfitting
better test loss
more reliable probabilities
```

---

# 36. Recommended Improvement: Better Loss Function

The current model uses:

```text
Sigmoid output + BCELoss
```

A more stable modern approach is:

```text
raw logits output + BCEWithLogitsLoss
```

This means removing the final sigmoid from the model and using:

```python
criterion = nn.BCEWithLogitsLoss()
```

For prediction, probabilities can be computed manually:

```python
probs = torch.sigmoid(logits)
```

This is more numerically stable.

---

# 37. Recommended Improvement: Class Weighting

Because good wine is the minority class, the loss can be modified to penalize mistakes on good wine more strongly.

With `BCEWithLogitsLoss`, one can use:

```python
pos_weight = torch.tensor([num_negative / num_positive]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This tells the model:

```text
missing a good wine is more costly
```

This may improve recall for the minority class.

---

# 38. Recommended Improvement: Threshold Tuning

The default threshold is:

```text
0.5
```

But for imbalanced classification, 0.5 may not be optimal.

If the goal is to detect more good wines, one can lower the threshold:

```text
0.5 → 0.4 → 0.3
```

Lowering the threshold usually:

```text
increases recall for good wine
decreases precision for good wine
```

The best threshold depends on the application.

For example:

```text
If missing a good wine is costly, prefer higher recall.
If falsely labeling ordinary wine as good is costly, prefer higher precision.
```

---

# 39. Recommended Improvement: Better Metrics

The project should later include:

```text
balanced accuracy
macro F1-score
ROC-AUC
PR-AUC
precision-recall curve
minority-class recall
minority-class precision
```

For imbalanced data, PR-AUC and minority-class recall are often more informative than raw accuracy.

---

# 40. Recommended Improvement: Cross-Validation

Because the dataset is small, a single train/validation/test split can be noisy.

A stronger evaluation would use repeated runs or k-fold cross-validation.

Suggested approach:

```text
repeat experiment for multiple random seeds
report mean ± standard deviation
```

This would make the hyperparameter conclusions more reliable.

---

# 41. Recommended Improvement: Regularization

Some models overfit strongly.

Possible regularization methods:

```text
weight decay
dropout
early stopping
smaller models
less training time
data augmentation, if meaningful
```

For tabular data, useful first steps are:

```text
weight decay
early stopping
smaller architecture
```

Example Adam with weight decay:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

# 42. Recommended Improvement: Cleaner Output Structure

Currently, outputs are saved in the root directory.

A cleaner project structure would use:

```text
outputs/figures/
outputs/models/
outputs/logs/
```

Example:

```text
outputs/figures/06_learning_curves.png
outputs/figures/06_confusion_matrix.png
outputs/figures/07A_depth.png
outputs/models/wine_mlp.pth
```

This would make the GitHub repository cleaner.

---

# 43. Recommended Improvement: Configurable Scripts

The project could be improved by adding command-line arguments.

Example:

```bash
python training_loop_v1.py --epochs 100 --batch-size 128 --lr 0.001
```

This would make experiments easier without editing the source code.

Useful arguments:

```text
--epochs
--batch-size
--lr
--hidden-sizes
--activation
--seed
--output-dir
```

---

# 44. Scientific Learning Observation

Although the dataset is wine quality, the project is broadly useful for scientific machine learning.

The same structure can be applied to:

```text
soft-matter simulation data
protein folding descriptors
molecular dynamics observables
phase-separation descriptors
rheology data
materials property prediction
biophysical classification problems
```

The reusable template is:

```text
structured numerical features
→ scaling
→ MLP
→ binary or regression target
→ validation curves
→ class-wise or regression metrics
→ hyperparameter experiments
```

---

# 45. Soft-Matter Physics Analogy Observations

The hyperparameter experiments can be connected to physical intuition.

## Learning rate

Analogous to step size in Monte Carlo or numerical minimization.

```text
too large → unstable, overshoots
too small → slow convergence
proper    → efficient exploration
```

## Network depth

Analogous to increasing the complexity/order of a model basis.

```text
too shallow → cannot represent complex structure
too deep    → unnecessary complexity and overfitting
```

## Network width

Analogous to increasing resolution in feature space.

```text
too narrow → coarse representation
moderate   → useful representation
too wide   → memorization risk
```

## Batch size

Analogous to statistical sample size for estimating a force or gradient.

```text
small batch → noisy estimate
large batch → smooth estimate
too large   → less stochastic exploration
```

## Activation function

Analogous to the response function of a unit.

```text
ReLU      → piecewise linear response
Sigmoid   → saturable response
Tanh      → symmetric saturable response
ELU       → smooth nonlinear response
```

---

# 46. Main Lessons Learned

The project teaches the following core lessons:

```text
1. Data leakage must be avoided by fitting preprocessing only on training data.
2. Accuracy alone is not reliable for imbalanced classification.
3. Confusion matrices reveal class-specific weaknesses.
4. Validation loss can reveal overfitting even when accuracy looks stable.
5. Deeper networks are not automatically better.
6. Wider networks can help but may overfit.
7. Learning rate is critical for optimization.
8. Sigmoid hidden activations can suffer from vanishing gradients.
9. Batch size changes gradient noise and optimization dynamics.
10. Early stopping is needed when validation loss starts increasing.
11. The final epoch is not necessarily the best model.
12. Model saving and reloading should always be verified.
13. Inference must use the same preprocessing fitted during training.
```

---

# 47. Final Project Conclusion

This project successfully demonstrates the complete practical workflow of neural-network training on tabular data.

The default MLP learns meaningful patterns, but it also overfits after around 29 epochs.

The final test accuracy is high, but the confusion matrix reveals that the model is much better at identifying ordinary wines than good wines.

The hyperparameter experiments show that moderate model complexity is usually better than excessive depth or width for this small dataset.

The strongest practical improvements would be:

```text
early stopping
class weighting
threshold tuning
better metrics
regularization
cross-validation
```

The final conceptual takeaway is:

> Deep learning is not only about building larger models. It is about controlling optimization, generalization, data leakage, class imbalance, and evaluation quality.

This project is therefore a strong educational foundation for more advanced deep-learning work in scientific and engineering applications.
