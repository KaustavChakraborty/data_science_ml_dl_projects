# ============================================================
# CODE 5: STATISTICAL INFERENCE — COEFFICIENTS, P-VALUES,
#         CONFIDENCE INTERVALS, LIKELIHOOD RATIO TESTS
# ============================================================
# This script demonstrates how to perform logistic regression
# not just for prediction, but for statistical inference.
#
# Key idea:
#   - scikit-learn is excellent for predictive modeling
#   - statsmodels is better when you need formal statistical output:
#       * coefficient estimates
#       * standard errors
#       * z-statistics
#       * p-values
#       * confidence intervals
#       * likelihood-based model comparison
#
# The script covers:
#   1. Logistic regression fitting with statsmodels
#   2. Full regression summary output
#   3. Extraction of coefficients and confidence intervals
#   4. Conversion of coefficients to odds ratios
#   5. Publication-style forest plot of odds ratios
#   6. Likelihood Ratio Test (LRT) for nested models
#   7. Variance Inflation Factor (VIF) for multicollinearity diagnosis
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# statsmodels.api provides the Logit model and helper functions such as add_constant.
import statsmodels.api as sm

# VIF is used to quantify multicollinearity among predictors.
# High multicollinearity can make coefficient estimates unstable and
# inflate standard errors, even if the model's prediction accuracy is good.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# StandardScaler standardizes each feature to zero mean and unit variance.
# This is useful because:
#   - coefficient magnitudes become more comparable across features
#   - numerical optimization is often more stable
#   - odds ratios then correspond to a one-standard-deviation increase
from sklearn.preprocessing import StandardScaler

# Breast cancer dataset is a built-in binary classification dataset in sklearn.
# It is often used for demonstrations because it is real, moderately sized,
# and contains meaningful continuous predictors.
from sklearn.datasets import load_breast_cancer

# Imported here, though not used in the current script.
# Often useful if you want to separate training and testing data for predictive work.
from sklearn.model_selection import train_test_split

# Suppress warnings to keep output cleaner for demonstration.
# This avoids clutter from convergence warnings or other informational messages.
import warnings
warnings.filterwarnings('ignore')

# Fixed random seed for reproducibility.
SEED = 42


# ─────────────────────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────────────────────
# Load the breast cancer dataset.
# This dataset is binary:
#   - target = 0 or 1
# The exact meaning of target labels depends on sklearn's encoding,
# but for the purposes of logistic regression inference, the main point
# is that y is binary.
data = load_breast_cancer()

# Use only the first 8 features for clarity and easier interpretation.
# If all features were used, the output would become longer and possibly
# harder to discuss in a compact example.
#
# X = predictor matrix
# y = binary response variable
X, y = data.data[:, :8], data.target

# Store the names of the first 8 features so they can be displayed
# in tables and plots.
feature_names = list(data.feature_names[:8])

# Standardize the predictors.
# Standardization transforms each feature:
#   x_scaled = (x - mean) / std
#
# This helps interpret coefficients on a comparable scale and improves
# numerical behavior during maximum likelihood estimation.
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# statsmodels does NOT automatically include an intercept term.
# Therefore, we explicitly add a constant column of 1s.
#
# This creates a design matrix:
#   X_aug = [1, x1, x2, ..., x8]
#
# The intercept term corresponds to the baseline log-odds when all
# standardized predictors are zero.
X_aug = sm.add_constant(X_sc)

# Show the shape of the final design matrix.
# Since we started with 8 predictors and added 1 intercept column,
# the shape should be:
#   (number_of_samples, 9)
print(f"Design matrix shape: {X_aug.shape}")


# ─────────────────────────────────────────────────────────────
# Fit Logistic Regression with Full Statistical Output
# ─────────────────────────────────────────────────────────────
# sm.Logit fits a logistic regression model using maximum likelihood estimation.
#
# Model form:
#   logit(P(y=1)) = \beta_0 + \beta_1 x1 + \beta_2 x2 + ... + \beta_8 x8
#
# where:
#   - \beta_0 is the intercept
#   - \beta_j are coefficients for predictors
#
# The model estimates coefficients by maximizing the log-likelihood.
logit_model = sm.Logit(y, X_aug)

# Fit the model using Newton-Raphson optimization.
#
# Arguments:
#   method='newton'
#       Uses Newton-Raphson iterations, a standard method for logistic regression MLE.
#
#   maxiter=200
#       Allows up to 200 optimization iterations.
#       This is generous and helps ensure convergence.
#
#   disp=True
#       Prints fitting progress and convergence information.
result = logit_model.fit(method='newton',
                          maxiter=200,
                          disp=True)

# Print a section header before the regression summary.
print("\n" + "="*60)
print("FULL LOGISTIC REGRESSION STATISTICAL SUMMARY")
print("="*60)

# Print the full statsmodels summary.
#
# result.summary2() typically includes:
#   - number of observations
#   - pseudo-R²
#   - log-likelihood
#   - coefficient estimates
#   - standard errors
#   - z-statistics
#   - p-values
#   - confidence intervals
#
# This is the central output for formal statistical interpretation.
print(result.summary2())


# ─────────────────────────────────────────────────────────────
# Extract and visualize coefficients with confidence intervals
# ─────────────────────────────────────────────────────────────
# Construct a summary DataFrame containing the most important quantities
# for each coefficient.
#
# The rows correspond to:
#   - Intercept
#   - each of the 8 features
#
# The columns include:
#   - coefficient estimate
#   - standard error
#   - Wald z-statistic
#   - p-value
#   - lower and upper confidence interval bounds
#   - odds ratio = exp(coefficient)
#   - confidence interval for the odds ratio

conf_int = result.conf_int()


coef_df = pd.DataFrame({
    # Feature labels: start with the intercept, then the actual predictor names
    'Feature':    ['Intercept'] + feature_names,

    # Estimated coefficients \beta_j
    # Interpretation:
    #   positive coefficient  -> predictor increases log-odds of class 1
    #   negative coefficient  -> predictor decreases log-odds of class 1
    'Coefficient': result.params,

    # Standard errors of the coefficient estimates
    # These quantify uncertainty in the estimated β_j values.
    'StdError':    result.bse,

    # Wald z-statistic = coefficient / standard error
    # Used to test H0: \beta_j = 0
    'z_stat':      result.tvalues,

    # Two-sided p-value associated with the Wald test.
    # Small p-value suggests the feature is statistically significant.
    'p_value':     result.pvalues,

    # Lower bound of the coefficient confidence interval
    # 'CI_lower':    result.conf_int().iloc[:, 0],

    'CI_lower':    conf_int[:, 0],

    # Upper bound of the coefficient confidence interval
    # 'CI_upper':    result.conf_int().iloc[:, 1],

    'CI_upper':    conf_int[:, 1],

    # Odds ratio = exp(\beta_j)
    #
    # Why exponentiate?
    # Logistic regression coefficients are on the log-odds scale.
    # Exponentiating converts them to multiplicative effects on odds.
    #
    # Interpretation:
    #   OR > 1  -> predictor increases odds of class 1
    #   OR < 1  -> predictor decreases odds of class 1
    #   OR = 1  -> no effect
    # 'Odds_Ratio':  np.exp(result.params),

    'Odds_Ratio':  np.exp(result.params),


    # Confidence interval lower bound for the odds ratio
    # 'OR_CI_lower': np.exp(result.conf_int().iloc[:,0]),

    'OR_CI_lower': np.exp(conf_int[:, 0]),

    # Confidence interval upper bound for the odds ratio
    # 'OR_CI_upper': np.exp(result.conf_int().iloc[:,1]),

    'OR_CI_upper': np.exp(conf_int[:, 1]),
})

# Print a compact, readable coefficient table.
# float_format is used to keep the output neat and publication-like.
print("\nCoefficient Summary:")
print(coef_df.to_string(index=False, float_format='{:.4f}'.format))


# ─────────────────────────────────────────────────────────────
# Forest Plot (Odds Ratios with 95% CIs) — publication style
# ─────────────────────────────────────────────────────────────
# Create a horizontal forest plot showing odds ratios and their
# 95% confidence intervals.
#
# Forest plots are commonly used in medical and scientific papers
# to summarize effect sizes visually.
fig, ax = plt.subplots(figsize=(9, 6))

# Exclude the intercept from the forest plot.
# The intercept is usually not of substantive interest in this type of plot;
# the main interest is in the explanatory variables.
df_plot = coef_df[coef_df['Feature'] != 'Intercept'].copy().reset_index(drop=True)

# y positions for the horizontal plot
y_pos = np.arange(len(df_plot))

# Assign bar colors based on statistical significance.
#   red   -> significant at p < 0.05
#   gray  -> not significant
#
# This visually separates important predictors from weaker ones.
colors_fp = ['#F44336' if p < 0.05 else '#9E9E9E' for p in df_plot['p_value']]

# Draw horizontal bars representing odds ratios.
#
# The bar starts at 1.0 and extends to the estimated odds ratio.
# Since OR = 1 means "no effect", anchoring visually around 1 is intuitive.
#
# The expression:
#   df_plot['Odds_Ratio'] - 1
# means the width is measured relative to OR = 1.
ax.barh(y_pos,
        df_plot['Odds_Ratio'] - 1,
        left=1.0,
        height=0.5, color=colors_fp, alpha=0.7)

# Overlay point estimates and horizontal error bars for 95% confidence intervals.
#
# xerr is specified asymmetrically:
#   left error  = OR - lower_CI
#   right error = upper_CI - OR
#
# This is necessary because odds-ratio confidence intervals are often asymmetric.
ax.errorbar(
    df_plot['Odds_Ratio'], y_pos,
    xerr=[df_plot['Odds_Ratio'] - df_plot['OR_CI_lower'],
          df_plot['OR_CI_upper'] - df_plot['Odds_Ratio']],
    fmt='o', color='black', capsize=4, lw=1.5, markersize=6
)

# Vertical reference line at OR = 1.
# This is the null-effect line:
#   OR = 1 means no change in odds.
# If a confidence interval crosses this line, the effect is not
# statistically distinguishable from no effect at that confidence level.
ax.axvline(1.0, color='black', lw=1.5, linestyle='--', label='OR = 1.0 (no effect)')

# Set y-axis ticks and labels to feature names
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['Feature'])

# Add p-value significance stars next to the confidence interval.
#
# Convention used:
#   *    p < 0.05
#   **   p < 0.01
#   ***  p < 0.001
#
# These are commonly used in figures to quickly indicate significance level.
for i, (_, row) in enumerate(df_plot.iterrows()):
    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01
          else ("*" if row['p_value'] < 0.05 else ""))

    # Only annotate if the feature is statistically significant
    if sig:
        ax.text(row['OR_CI_upper'] + 0.05, i, sig, va='center',
                fontsize=12, color='#F44336', fontweight='bold')

# Axis label
ax.set_xlabel("Odds Ratio (OR) with 95% Confidence Interval")

# Figure title
ax.set_title("Forest Plot: Odds Ratios for Each Feature\n"
             "Red bars = significant (p < 0.05). Stars: *p<0.05, **p<0.01, ***p<0.001")

# Add legend and x-axis grid
ax.legend(); ax.grid(True, alpha=0.3, axis='x')

# Improve spacing so labels and annotations fit nicely
plt.tight_layout()

# Save the figure to disk
plt.savefig("forest_plot_or.png", dpi=150, bbox_inches='tight')

# Display the plot
# plt.show()


# ─────────────────────────────────────────────────────────────
# Likelihood Ratio Test (comparing nested models)
# ─────────────────────────────────────────────────────────────
# The Likelihood Ratio Test (LRT) compares two nested models:
#
#   Reduced model ⊂ Full model
#
# Here:
#   - Full model uses all 8 predictors
#   - Reduced model uses only the first 4 predictors
#
# The question is:
#   "Do the extra predictors significantly improve model fit?"
#
# LRT statistic:
#   2 * [logLik(full) - logLik(reduced)]
#
# Under regular conditions, this statistic follows approximately
# a chi-square distribution with degrees of freedom equal to the
# difference in the number of model parameters.
print("\n" + "="*60)
print("LIKELIHOOD RATIO TEST: Full vs Reduced Model")
print("="*60)

# Fit the full model again explicitly.
# This uses all 8 predictors + intercept.
logit_full    = sm.Logit(y, X_aug).fit(disp=False)

# Fit a reduced model using only:
#   - intercept column
#   - first 4 standardized predictors
#
# Since X_aug contains:
#   column 0 = intercept
#   columns 1..8 = predictors
#
# X_aug[:, :5] means:
#   intercept + first 4 predictors
logit_reduced = sm.Logit(y, X_aug[:, :5]).fit(disp=False)

# Compute the LRT statistic.
#
# llf = log-likelihood value at the fitted optimum
#
# Larger log-likelihood means better fit.
# If the full model improves fit substantially, lrt_stat will be large.
lrt_stat = 2 * (logit_full.llf - logit_reduced.llf)

# Difference in model degrees of freedom.
# This is the number of extra parameters tested by moving from
# the reduced model to the full model.
df_diff  = logit_full.df_model - logit_reduced.df_model

# Import scipy.stats locally here, exactly as in the original code.
from scipy import stats

# Compute the p-value from the chi-square distribution.
# This is the upper-tail probability:
#   P(Chi-square >= observed statistic)
p_lrt = 1 - stats.chi2.cdf(lrt_stat, df_diff)

# Print results of the model comparison.
print(f"Full model log-likelihood:    {logit_full.llf:.4f}")
print(f"Reduced model log-likelihood: {logit_reduced.llf:.4f}")
print(f"LRT statistic: {lrt_stat:.4f}  (χ² with {df_diff:.0f} df)")
print(f"p-value: {p_lrt:.6f}")

# Interpret the p-value at the conventional alpha = 0.05 threshold.
if p_lrt < 0.05:
    print("The additional features in the full model significantly improve fit (p < 0.05)")
else:
    print("The additional features do NOT significantly improve fit (p ≥ 0.05)")


# ─────────────────────────────────────────────────────────────
# Variance Inflation Factor (VIF) — detect multicollinearity
# ─────────────────────────────────────────────────────────────
# VIF quantifies how strongly each predictor is linearly explained
# by the other predictors.
#
# Interpretation:
#   VIF = 1    -> no multicollinearity
#   VIF < 5    -> usually acceptable
#   VIF > 10   -> often considered severe multicollinearity
#
# Why this matters:
#   - multicollinearity inflates standard errors
#   - coefficients may become unstable
#   - p-values may become misleading
#   - interpretation of individual predictors becomes difficult
print("\n" + "="*60)
print("VARIANCE INFLATION FACTORS (multicollinearity check)")
print("="*60)
print("VIF > 10: severe multicollinearity → coefficient estimates unreliable")
print("VIF < 5:  acceptable")
print()

# Build a DataFrame containing VIF values for each predictor.
#
# Important:
#   VIF is computed only on the predictors X_sc here, not on the added constant.
#
# For each feature index i:
#   variance_inflation_factor(X_sc, i)
# regresses feature i on all other features and computes:
#
#   VIF_i = 1 / (1 - R_i^2)
#
# where R_i^2 is the R^@ from that auxiliary regression.
vif_data = pd.DataFrame({
    'Feature': feature_names,
    'VIF': [variance_inflation_factor(X_sc, i) for i in range(X_sc.shape[1])]
}).sort_values('VIF', ascending=False)

# Print VIF values in descending order so the worst multicollinearity
# issues appear first.
print(vif_data.to_string(index=False, float_format='{:.2f}'.format))
