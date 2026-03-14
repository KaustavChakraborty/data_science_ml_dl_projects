# Observations on Class Imbalance Handling with Logistic Regression

## Objective

The purpose of this experiment was to study how different strategies for handling **class imbalance** affect the performance of a **logistic regression classifier** on a binary classification problem where the minority class is rare. The dataset used in the experiment had approximately:

- **90% samples from class 0**
- **10% samples from class 1**

This setup is representative of many real-world scientific and engineering tasks, such as anomaly detection, rare-event prediction, uncommon phase classification, rare material failure detection, and similar problems where the event of interest is much less frequent than the background class.

The strategies evaluated were:

1. **Baseline logistic regression**
2. **Logistic regression with `class_weight='balanced'`**
3. **SMOTE oversampling**
4. **Threshold optimization on the class-weighted model**

The study used several metrics more appropriate for imbalanced classification than accuracy, including:

- Precision
- Recall
- F1 score
- MCC (Matthews Correlation Coefficient)
- AUC-ROC
- AUC-PR

---

## General Observation: Accuracy Alone Is Misleading

The first and most important observation is that **accuracy is not a reliable performance indicator in imbalanced classification**.

The baseline model achieved a relatively high accuracy, but this did not reflect meaningful minority-class detection. Since the dataset is dominated by class 0, a model can obtain apparently good accuracy simply by predicting the majority class most of the time. Therefore, in imbalanced settings, accuracy must always be interpreted alongside metrics that explicitly capture minority-class performance.

This experiment strongly demonstrates that a model with high accuracy may still be unusable if it fails to identify rare positive cases.

---

## Observation 1: Baseline Logistic Regression Failed on the Minority Class

The baseline logistic regression model showed the following behavior:

- it achieved high overall accuracy,
- but it had **zero precision, zero recall, and zero F1 score** for the minority class,
- and its MCC was slightly negative.

This indicates that the baseline model effectively collapsed to majority-class prediction at the default threshold of 0.5.

### Interpretation

This does **not** necessarily mean that the model learned nothing. The baseline still showed nontrivial AUC-ROC and AUC-PR values, which means the model had some ability to rank positive examples above negative ones. However, when probabilities were converted into hard labels using the default threshold, the model became too conservative and failed to call any samples positive in a useful way.

### Key insight

The baseline model was a poor classifier at threshold 0.5, even though its probability scores contained some useful signal. This distinction between **ranking ability** and **thresholded classification performance** is one of the central lessons of the experiment.

---

## Observation 2: Class Weighting Substantially Improved Minority-Class Detection

Applying `class_weight='balanced'` to logistic regression dramatically improved the classifier’s ability to detect the minority class.

Compared with the baseline model, class weighting produced:

- much higher recall,
- nonzero precision,
- a large increase in F1 score,
- a significant improvement in MCC,
- a moderate improvement in AUC-ROC,
- and a slight improvement in AUC-PR.

### Interpretation

Class weighting makes errors on minority-class samples more costly during training. As a result, the model shifts its decision boundary to become more sensitive to class 1. This leads to a much greater number of positive predictions.

The result is a model that:

- recovers a large fraction of actual positives,
- but also makes more false-positive predictions.

This is a classic tradeoff in imbalanced learning. The classifier becomes more useful for rare-event detection, but its increased sensitivity comes at the cost of lower precision and lower accuracy.

### Important takeaway

The drop in accuracy after class weighting is **not** a sign of worse performance. On the contrary, the model became far more useful because it started identifying minority-class samples. In imbalanced problems, improved recall and F1 often matter far more than raw accuracy.

---

## Observation 3: Precision–Recall Tradeoff Became Clear After Imbalance Handling

The class-weighted model exhibited a clear precision–recall tradeoff:

- recall increased substantially,
- but precision remained modest.

This means the model became much more willing to call samples positive, thereby catching more true positives, but it also introduced many false positives.

### Interpretation

This is expected in rare-event classification. When the minority class is rare, increasing sensitivity usually requires accepting more false alarms. In many scientific or safety-critical applications, this may be the correct tradeoff. For example:

- in anomaly detection, it is often acceptable to inspect some false alarms if true anomalies are not missed,
- in rare material failure detection, it may be preferable to identify most risky cases even if some normal cases are flagged,
- in screening tasks, high recall is often more valuable than very high precision.

This experiment shows that imbalance-aware training shifts the classifier from being overly conservative to being operationally useful.

---

## Observation 4: SMOTE Produced Results Very Similar to Class Weighting

SMOTE oversampling generated a fully balanced training set and resulted in:

- precision almost identical to the class-weighted model,
- recall very close to the class-weighted model,
- nearly identical F1 and MCC values,
- almost the same AUC-ROC and AUC-PR.

### Interpretation

This indicates that, for this specific dataset and for logistic regression, **SMOTE did not provide a major practical advantage over class weighting**.

This is not surprising. Logistic regression is a linear model, and both class weighting and oversampling can influence the fitted boundary in similar ways. In this case, both methods effectively increased the model’s attention to the minority class, leading to similar improvements.

### Practical conclusion

For this problem, `class_weight='balanced'` appears to be a simpler and equally effective strategy compared with SMOTE. Since class weighting is easier to implement and does not require synthetic data generation, it may be the preferable option when using logistic regression on similarly structured data.

---

## Observation 5: Threshold Optimization Produced the Best Final Classifier

Threshold optimization on the class-weighted model gave the best overall results in terms of:

- **F1 score**
- **MCC**

The optimized threshold was approximately:

- **0.678**, which is higher than the default threshold of 0.5.

At this threshold:

- precision increased,
- recall decreased relative to the default weighted model,
- but the overall balance between the two improved,
- resulting in the highest F1 and MCC among all tested strategies.

### Interpretation

This result is one of the most important findings of the experiment.

The ranking ability of the class-weighted model, as measured by AUC-ROC and AUC-PR, remained unchanged. The model itself did not become intrinsically better. What improved was the **decision rule used to convert probabilities into class labels**.

This shows that once a model produces informative probability scores, the final performance can often be improved significantly simply by choosing a threshold aligned with the actual task objective.

### Key takeaway

In this experiment, **threshold selection mattered more than the difference between class weighting and SMOTE**. This is a highly valuable practical lesson. Often, better performance does not require a more complicated model or resampling method, but rather a more appropriate operating threshold.

---

## Observation 6: The Optimal Threshold Was Higher Than 0.5

A particularly interesting result is that the threshold maximizing F1 was around **0.678**, not below 0.5.

### Why this matters

It is common to assume that imbalanced classification always requires lowering the decision threshold in order to detect more positives. However, this experiment shows that such a rule is not universal.

Here, the class-weighted model at threshold 0.5 was already very sensitive and produced many false positives. Increasing the threshold made the classifier more selective. This reduced false positives enough to increase precision significantly, and although recall dropped somewhat, the net effect on F1 was positive.

### Interpretation

This means that after class weighting, the default threshold of 0.5 had become too permissive. The threshold optimization step corrected this by selecting a more balanced operating point.

### Broader lesson

The optimal threshold depends on:

- the score distribution produced by the model,
- the degree of overlap between classes,
- and the metric being optimized.

Therefore, threshold should be treated as a tunable parameter, not assumed to be fixed at 0.5.

---

## Observation 7: AUC-PR Changed Very Little Across the Better Methods

One notable feature of the results is that AUC-PR values were very close for:

- class weighting,
- SMOTE,
- and threshold optimization.

### Interpretation

This indicates that the overall ranking quality of the positive class did not change much across these methods. In other words, all three methods learned broadly similar probability orderings of the data points.

The major differences between them arose not from dramatic changes in ranking ability, but from how those scores were translated into hard class labels.

### Important conclusion

This confirms that the experiment is mainly about improving the **operating behavior** of the classifier rather than discovering a completely new separation structure in the feature space.

---

## Observation 8: MCC Was an Especially Informative Metric

Among all the reported metrics, MCC was particularly useful because:

- it incorporates all four entries of the confusion matrix,
- it is much less sensitive to class imbalance than accuracy,
- it penalizes degenerate models strongly,
- and it provides a balanced single-number summary of performance.

The baseline model had an MCC close to zero or slightly negative, indicating poor and potentially misleading classification behavior. The class-weighted, SMOTE, and threshold-optimized models all produced clearly positive MCC values, with the threshold-optimized model achieving the best score.

### Interpretation

This supports the conclusion that the threshold-optimized classifier was the most balanced and reliable of the methods tested.

### Practical note

For imbalanced binary classification, MCC should be considered one of the strongest single metrics for model comparison.

---

## Observation 9: The Threshold Sweep Plot Clearly Illustrated Metric Tradeoffs

The threshold sweep for the class-weighted model showed that:

- **recall decreases as threshold increases,**
- **precision generally increases as threshold increases,**
- **F1 reaches a peak at an intermediate threshold.**

### Interpretation

This behavior is exactly what is expected from a probabilistic classifier:

- at low thresholds, the classifier labels many points as positive, producing high recall but low precision,
- at high thresholds, the classifier becomes more conservative, improving precision but missing many positives,
- between these extremes lies a threshold that best balances the two.

The F1 peak near 0.678 visually confirms that the default threshold of 0.5 was not optimal for this problem.

### Broader significance

This plot provides a strong visual demonstration of why threshold tuning should be included in serious classification workflows, especially when the classes are imbalanced.

---

## Observation 10: The Precision–Recall Curve Confirmed the Presence of Real Predictive Signal

The precision–recall curve remained consistently above the random baseline determined by the positive-class prevalence.

### Interpretation

This confirms that the classifier is learning genuine structure and not simply performing at chance level.

However, the AUC-PR was still moderate rather than very high. This suggests that:

- the classes are not trivially separable,
- the features have limited discriminative power,
- or the linear logistic regression model cannot fully capture the true decision boundary.

### Scientific implication

The model is useful, but the dataset remains challenging. The remaining limitation is likely due not only to class imbalance but also to intrinsic overlap in feature space or limited model expressiveness.

---

## Observation 11: Class Imbalance Handling Improved Decision Bias More Than Intrinsic Separability

A deeper pattern across the results is that imbalance-handling methods improved classification outcomes without dramatically changing ranking metrics.

### Interpretation

This suggests that the major issue in the original baseline model was not the total absence of signal, but rather a **decision bias toward the majority class**.

Class weighting and SMOTE corrected this bias, making the minority class visible to the decision rule. Threshold tuning then selected the best point on that already-learned ranking structure.

### Important conclusion

For this dataset, the main gains came from:

1. correcting the model’s bias toward the majority class,
2. and choosing a better classification threshold,

rather than from fundamentally transforming the model’s ability to separate the two classes.

---

## Observation 12: Logistic Regression Was Able to Learn Useful Structure but Has Natural Limits

The experiment also reveals the strengths and limitations of logistic regression.

### Strengths
- simple and interpretable,
- responds well to class weighting,
- produces probability outputs suitable for threshold tuning,
- performs reasonably well even on an imbalanced dataset.

### Limitations
- it is a linear model,
- it may not capture complex nonlinear structure,
- the final AUC-PR remains modest,
- and resampling alone cannot overcome intrinsic class overlap.

### Interpretation

This suggests that while logistic regression is an excellent starting point and a very good educational model for this task, further gains on similar data may require:

- more informative features,
- nonlinear models,
- calibration analysis,
- or more advanced imbalance-aware methods.

---

## Overall Conclusions

The experiment leads to the following main conclusions:

### 1. Baseline logistic regression is not sufficient for imbalanced classification
Ignoring class imbalance causes the model to favor the majority class so strongly that minority detection can collapse completely.

### 2. Accuracy is not a trustworthy metric in rare-event problems
A model can appear accurate while failing entirely on the minority class. Metrics such as F1, MCC, and AUC-PR are much more informative.

### 3. Class weighting is a simple and highly effective strategy
It significantly improves minority-class recall and overall usefulness with minimal implementation effort.

### 4. SMOTE was not meaningfully better than class weighting for this experiment
For logistic regression on this dataset, both methods produced nearly identical results.

### 5. Threshold optimization provided the best final performance
It improved F1 and MCC without changing the model itself, showing that threshold is a critical part of the classification pipeline.

### 6. The main challenge was operational, not purely representational
The model already contained useful ranking information. The major improvements came from using that information more appropriately.

---

## Final Summary

This study demonstrates that in imbalanced binary classification, the most important improvements often come from:

- **making the model aware of minority-class importance during training,**
- and **choosing a decision threshold aligned with the desired performance objective.**

Among the tested approaches, the best practical strategy was:

- **logistic regression with `class_weight='balanced'`**
- followed by **threshold optimization**

This combination provided the best balance between sensitivity and precision and yielded the strongest overall classifier according to F1 and MCC.

---

## Recommended Next Observations for Future Work

Future extensions of this experiment could investigate:

- RandomUnderSampler
- SMOTE + Tomek links
- repeated cross-validation
- validation-based threshold tuning instead of test-based tuning
- confusion matrix analysis for each strategy
- comparison with nonlinear models such as Random Forest or XGBoost
- probability calibration analysis
- performance on real imbalanced scientific datasets

These would help determine whether the observed behavior is specific to this synthetic setup or extends more broadly across more realistic data distributions.
