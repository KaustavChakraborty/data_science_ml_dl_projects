# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH — EXTREMELY DOCUMENTED
# ============================================================
#
# PURPOSE OF THIS SCRIPT
# ----------------------
# This script implements binary logistic regression directly from
# the mathematics, using only NumPy for numerical operations.
#
# It does NOT rely on scikit-learn's LogisticRegression estimator.
# Instead, it explicitly builds:
#
#   1. the sigmoid function
#   2. the binary cross-entropy loss
#   3. the gradient of that loss
#   4. gradient descent optimization
#   5. regularization terms (L1, L2, ElasticNet)
#   6. prediction functions
#   7. visualization / diagnostics
#
# The script then demonstrates three experiments:
#
#   EXPERIMENT 1:
#       Train logistic regression on a simple 2D linear dataset,
#       evaluate train/test accuracy and test log loss,
#       and visualize the loss curve and decision boundary.
#
#   EXPERIMENT 2:
#       Compare different regularization choices:
#       no regularization, L2, stronger L2, L1, ElasticNet.
#       Then visualize coefficient magnitudes and sparsity.
#
#   EXPERIMENT 3:
#       Compare optimization behavior for different batch sizes:
#       full batch, mini-batch, smaller mini-batch, and SGD.
#       Then visualize convergence behavior through loss curves.
#
# MATHEMATICAL MODEL
# ------------------
# For each sample x, logistic regression computes:
#
#     z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
#
# This is a linear score (sometimes called the logit).
#
# The score is converted into a probability using the sigmoid:
#
#     sigma(z) = 1 / (1 + exp(-z))
#
# So:
#
#     P(y=1 | x) = sigma(z)
#
# and
#
#     P(y=0 | x) = 1 - sigma(z)
#
# During training, the model chooses β to minimize binary
# cross-entropy loss, optionally with regularization.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Synthetic dataset generator used for experiments
from sklearn.datasets import make_classification

# Used to split dataset into training and testing subsets
from sklearn.model_selection import train_test_split

# Used to standardize features (important for optimization)
from sklearn.preprocessing import StandardScaler

# Used only for evaluation convenience
# accuracy_score compares predicted labels vs true labels
# log_loss evaluates the quality of predicted probabilities
from sklearn.metrics import accuracy_score, log_loss


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------
# Setting a random seed makes results reproducible:
# - same random dataset
# - same random train/test split
# - same random mini-batch selection pattern (given same calls)
SEED = 42
np.random.seed(SEED)


class LogisticRegressionScratch:
    """
    Binary logistic regression implemented from first principles.

    ============================================================
    WHAT THIS CLASS STORES
    ============================================================
    The model learns a weight vector:

        beta_ = [\beta_0, \beta_1, \beta_2, ..., \beta_p]

    where:
        \beta_0 = intercept (bias term)
        \beta_1, \beta_2, ..., \beta_p = coefficients for p input features

    During training, we repeatedly update beta_ using gradient
    descent in order to reduce cross-entropy loss.

    ============================================================
    IMPORTANT HYPERPARAMETERS
    ============================================================
    lr : float
        Learning rate \alpha.
        Controls the size of each parameter update.

    n_iter : int
        Maximum number of optimization iterations.

    penalty : None, 'l1', 'l2', or 'elasticnet'
        Type of regularization to apply.

    C : float
        Inverse regularization strength.
        This follows the scikit-learn convention:
            λ = 1 / C
        So:
            smaller C -> larger λ -> stronger regularization

    l1_ratio : float
        Only used for ElasticNet.
        l1_ratio = 1 means pure L1
        l1_ratio = 0 means pure L2
        values in between mix the two

    batch_size : None or int
        None  -> full batch gradient descent
        1     -> stochastic gradient descent (SGD)
        k>1   -> mini-batch gradient descent

    tol : float
        Convergence tolerance.
        If the gradient norm becomes smaller than tol, training stops.

    verbose : int
        If nonzero, print progress every `verbose` iterations.

    ============================================================
    LEARNED ATTRIBUTES
    ============================================================
    beta_ : np.ndarray or None
        Parameter vector after fitting.
        Shape = (p + 1,)
        Includes intercept in position 0.

    loss_history_ : list[float]
        Stores the training loss value at each iteration.
        Useful for convergence diagnostics and plotting.
    """

    def __init__(self, lr=0.1, n_iter=1000, penalty='l2', C=1.0,
                 l1_ratio=0.5, batch_size=None, tol=1e-5, verbose=100):

        # Store optimization hyperparameters
        self.lr        = lr
        self.n_iter    = n_iter

        # Store regularization settings
        self.penalty   = penalty
        self.C         = C          # λ = 1/C
        self.l1_ratio  = l1_ratio

        # Store batching / convergence / logging settings
        self.batch_size= batch_size
        self.tol       = tol
        self.verbose   = verbose

        # These will be set during fit()
        self.beta_     = None   # [β₀, β₁, ..., βₚ] — the learned weights
        self.loss_history_ = []

    # =========================================================
    # CORE MATHEMATICAL FUNCTIONS
    # =========================================================

    @staticmethod
    def sigmoid(z):
        """
        Compute the sigmoid of z:
            sigma(z) = 1 / (1 + exp(-z))

        WHY SIGMOID?
        ------------
        Logistic regression needs an output between 0 and 1
        so that it can be interpreted as a probability.

        INPUT
        -----
        z : scalar or numpy array
            Linear score(s), potentially any real number.

        OUTPUT
        ------
        scalar or numpy array
            Probability value(s) in (0, 1).

        NUMERICAL STABILITY
        -------------------
        If z is very large or very negative, exp(-z) can overflow.
        To prevent that, we clip z into a safe numerical range.
        This does not meaningfully change practical outputs because:
        - very large positive z -> sigmoid already ~1
        - very large negative z -> sigmoid already ~0
        """


        # Clip to prevent exp overflow (doesn't affect sigmoid output in practice)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


    def _linear_combination(self, X_aug):
        """
        Compute the linear score for all samples at once.

        MATHEMATICALLY
        --------------
        If X_aug is the augmented design matrix with a leading 1-column,
        then this computes:

            z = X_aug @ beta_

        which is the vectorized form of:
        \beta_1, \beta_2, ..., \beta_p

            z_i =  \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_p x_{ip}

        INPUT
        -----
        X_aug : np.ndarray of shape (n_samples, p + 1)
            Feature matrix after adding an intercept column of ones.

        OUTPUT
        ------
        z : np.ndarray of shape (n_samples,)
            One linear score per sample.
        """
       
        return X_aug @ self.beta_   # shape: (n,)

    def _compute_loss(self, y, y_hat):
        """
        Compute total objective value:
            cross-entropy loss + regularization penalty

        BINARY CROSS-ENTROPY
        --------------------
        For binary classification, the loss is:

            J(\beta) = -(1/n) * \sum [ y log(y_hat) + (1-y) log(1-y_hat) ]

        where:
            y     = true label in {0,1}
            y_hat = predicted probability P(y=1|x)

        INTERPRETATION
        --------------
        - If the model gives high probability to the correct class,
          the loss is small.
        - If the model gives high probability to the wrong class,
          the loss becomes large.

        NUMERICAL STABILITY
        -------------------
        We clip y_hat to avoid log(0), which would be undefined.

        INPUT
        -----
        y : np.ndarray of shape (n_samples,)
            True labels.
        y_hat : np.ndarray of shape (n_samples,)
            Predicted probabilities for class 1.

        OUTPUT
        ------
        float
            Total loss = data loss + regularization loss
        """

        eps = 1e-15   # small constant to prevent log(0)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        n = len(y)

        # Cross-entropy over all samples
        ce_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # Add regularization penalty to the loss
        reg_loss = self._regularization_loss()
        return ce_loss + reg_loss

    def _regularization_loss(self):
        """
        Compute the regularization contribution to the loss.

        IMPORTANT
        ---------
        The intercept \beta_0 is not regularized.

        ElasticNet:
            \Lambda * [\rho * |\beta|_1 + (1-\rho)/2 * |\beta|_2^2]
            Mixes L1 and L2 behavior.

        RETURNS
        -------
        float
            Penalty value added to the loss.
        """

        lam = 1.0 / self.C          # λ = 1/C (sklearn convention)
        # Exclude intercept from regularization
        beta_no_intercept = self.beta_[1:]   

        if self.penalty == 'l2':
            return (lam / 2.0) * np.sum(beta_no_intercept ** 2)

        elif self.penalty == 'l1':
            return lam * np.sum(np.abs(beta_no_intercept))

        elif self.penalty == 'elasticnet':
            rho = self.l1_ratio
            l1_term = rho * np.sum(np.abs(beta_no_intercept))
            l2_term = (1 - rho) / 2.0 * np.sum(beta_no_intercept ** 2)
            return lam * (l1_term + l2_term)
        else:
            # No penalty
            return 0.0

    def _compute_gradient(self, X_aug, y, y_hat):
        """
        Compute the gradient of the objective with respect to beta.

        CORE RESULT
        -----------
        For logistic regression with cross-entropy loss, the gradient is:

            \grad J = (1/n) X_aug^T (y_hat - y)

        RESIDUALS
        ---------
        residuals = y_hat - y

        If the model predicts too high for class 1 when true y=0,
        the residual is positive.
        If the model predicts too low for class 1 when true y=1,
        the residual is negative.

        Multiplying X_aug^T by residuals aggregates the effect over all
        samples and gives one gradient component per parameter.

        REGULARIZATION GRADIENT
        -----------------------
        We add the derivative (or subgradient for L1) of the regularization
        term to \beta_1...\beta_p, but NOT to \beta_0.

        RETURNS
        -------
        np.ndarray of shape (p + 1,)
            Gradient vector for all parameters including intercept.
        """

        n = len(y)

        # Prediction error in probability space
        residuals = y_hat - y           # shape (n,)

        # Data gradient from cross-entropy loss
        grad = (X_aug.T @ residuals) / n  # shape (p+1,)

        # Add regularization gradient only to non-intercept weights
        grad_reg = self._regularization_gradient()
        grad[1:] += grad_reg

        return grad

    def _regularization_gradient(self):
        """
        Compute derivative/subgradient of the regularization penalty.

        RETURNS SHAPE
        -------------
        This returns only the gradient for \beta_1...\beta_p,
        not for the intercept \beta_0.
        """

        lam = 1.0 / self.C
        beta_no_intercept = self.beta_[1:]

        if self.penalty == 'l2':
            return lam * beta_no_intercept

        elif self.penalty == 'l1':
            return lam * np.sign(beta_no_intercept)

        elif self.penalty == 'elasticnet':
            rho = self.l1_ratio
            return lam * (rho * np.sign(beta_no_intercept)
                          + (1 - rho) * beta_no_intercept)

        else:
            return np.zeros_like(beta_no_intercept)


    # =========================================================
    # TRAINING
    # =========================================================

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        TRAINING WORKFLOW
        -----------------
        Step 1:
            Augment X with a leading column of ones so the intercept
            can be learned as part of beta.

        Step 2:
            Initialize all parameters to zero.

        Step 3:
            Repeat for many iterations:
                a) choose full batch or mini-batch
                b) compute linear scores z
                c) compute probabilities y_hat = sigmoid(z)
                d) compute full-data loss for tracking
                e) compute gradient on chosen batch
                f) update beta
                g) check convergence using gradient norm

        PARAMETERS
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        y : np.ndarray of shape (n_samples,)
            Binary labels 0 or 1.

        RETURNS
        -------
        self
            Trained model instance.
        """

        n, p = X.shape

        # -----------------------------------------------------
        # Add intercept column
        # -----------------------------------------------------
        # We convert X from shape (n, p) to (n, p+1) by adding
        # a first column of ones.
        #
        # This makes the model equation:
        #     z = X_aug @ beta
        #
        # where beta[0] acts as the intercept.
        X_aug = np.column_stack([np.ones(n), X])

        # -----------------------------------------------------
        # Initialize parameters
        # -----------------------------------------------------
        # beta_ has length p+1 because of the intercept.
        # Starting at zero is common and works fine for
        # logistic regression
        self.beta_ = np.zeros(p + 1)

        # Reset training history
        self.loss_history_ = []


        # -----------------------------------------------------
        # Main optimization loop
        # -----------------------------------------------------
        for iteration in range(1, self.n_iter + 1):

            # -------------------------------------------------
            # Choose optimization batch
            # -------------------------------------------------
            # If batch_size is None:
            #   use all training samples -> full batch GD
            #
            # If batch_size = 1:
            #   use one random sample -> SGD
            #
            # If batch_size = k:
            #   use k random samples -> mini-batch GD
            if self.batch_size is None:
                # Full batch gradient descent — uses all n samples
                X_batch, y_batch = X_aug, y
            else:
                # Mini-batch / stochastic: randomly sample batch_size examples
                idx = np.random.choice(n, size=self.batch_size, replace=False)
                X_batch, y_batch = X_aug[idx], y[idx]

            # -------------------------------------------------
            # Forward pass on the selected batch
            # -------------------------------------------------
            # 1. Compute linear scores
            z      = self._linear_combination(X_batch)   # z = X_tilta β

            # 2. Convert them into probabilities
            y_hat  = self.sigmoid(z)                      # y_cap = \sigma(z)

            # -------------------------------------------------
            # Compute full-data loss for monitoring
            # -------------------------------------------------
            # Important detail:
            # even when training uses mini-batches, the script
            # records the loss on the full dataset each iteration.
            #
            # This makes the loss curve easier to interpret.
            z_full      = self._linear_combination(X_aug)
            y_hat_full  = self.sigmoid(z_full)
            loss        = self._compute_loss(y, y_hat_full)
            self.loss_history_.append(loss)

            # -------------------------------------------------
            # Backward pass: gradient computation
            # -------------------------------------------------
            grad = self._compute_gradient(X_batch, y_batch, y_hat)

            grad = np.clip(grad, -1, 1) # Prevents the "explosion"
            # -------------------------------------------------
            # Parameter update
            # -------------------------------------------------
            # Standard gradient descent step:
            #
            #     beta <- beta - lr * grad
            #
            # If grad points toward increasing loss, subtracting it
            # moves beta toward decreasing loss.
            self.beta_ -= self.lr * grad

            # -------------------------------------------------
            # Convergence check
            # -------------------------------------------------
            # The norm of the gradient tells us how steep the loss
            # surface currently is.
            #
            # If ||grad|| is very small, we are near a stationary point.
            grad_norm = np.linalg.norm(grad)

            if grad_norm < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {iteration} (||grad|| = {grad_norm:.2e})")
                break

            # -------------------------------------------------
            # Optional logging
            # -------------------------------------------------
            if self.verbose and iteration % self.verbose == 0:
                print(f"  Iter {iteration:5d} | Loss: {loss:.6f} | ||grad||: {grad_norm:.4f}")

        return self

    # =========================================================
    # PREDICTION
    # =========================================================

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        OUTPUT FORMAT
        -------------
        Returns shape (n_samples, 2):
            column 0 = P(y=0 | x)
            column 1 = P(y=1 | x)

        This matches scikit-learn convention.

        STEPS
        -----
        1. Add intercept column to X
        2. Compute z = X_aug @ beta
        3. Compute p1 = sigmoid(z)
        4. Return [1-p1, p1]
        """

        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        z = self._linear_combination(X_aug)
        prob_class1 = self.sigmoid(z)
        return np.column_stack([1 - prob_class1, prob_class1])

    def predict(self, X, threshold=0.5):
        """
        Convert predicted probabilities into class labels.

        RULE
        ----
        Predict class 1 if:
            P(y=1|x) >= threshold

        else predict class 0.
        """

        prob_class1 = self.predict_proba(X)[:, 1]
        return (prob_class1 >= threshold).astype(int)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Accuracy = (# correct predictions) / (total predictions)

        This is a simple label-based metric.
        It does not care about confidence/probability quality.
        """

        return accuracy_score(y, self.predict(X))

    # =========================================================
    # DIAGNOSTIC PLOTTING
    # =========================================================

    def plot_loss_curve(self, ax=None):
        """
        Plot training loss vs iteration.

        WHY THIS PLOT IS USEFUL
        -----------------------
        It shows optimization behavior:
        - decreasing smoothly -> stable convergence
        - very noisy -> typical in SGD / small batches
        - flat very early -> possible convergence or bad learning rate
        - diverging / oscillating -> learning rate may be too large
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.loss_history_, color='#2196F3', lw=2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Training Loss Curve"); ax.grid(True, alpha=0.3)
        return ax

    def plot_decision_boundary(self, X, y, ax=None, title="Decision Boundary"):
        """
        Plot 2D probability surface and decision boundary.

        IMPORTANT
        ---------
        This only works when X has exactly 2 features, because
        the plot lives in a 2D feature plane.

        WHAT IS DRAWN
        -------------
        1. A dense grid in feature space
        2. Predicted P(y=1|x) on that grid
        3. A colored probability heatmap
        4. The contour where P(y=1|x)=0.5
           -> this is the decision boundary
        5. The actual data points

        WHY P=0.5?
        ----------
        Because the default classifier predicts class 1 when
        probability >= 0.5, so that contour is exactly the
        classification separator.
        """

        if X.shape[1] != 2:
            print(f"Skipping decision boundary plot: X has {X.shape[1]} features, but 2 are required.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        # -----------------------------------------------------
        # Build a plotting grid spanning the data range
        # -----------------------------------------------------
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        xx1, xx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, 300),
            np.linspace(x2_min, x2_max, 300)
        )

        # Flatten grid points into a matrix with shape (N_grid, 2)
        grid = np.column_stack([xx1.ravel(), xx2.ravel()])

        # -----------------------------------------------------
        # Predict probability of class 1 on each grid point
        # -----------------------------------------------------
        Z = self.predict_proba(grid)[:, 1].reshape(xx1.shape)

        # Plot probability heatmap
        contourf = ax.contourf(xx1, xx2, Z, levels=50,
                                cmap='RdBu_r', alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(contourf, ax=ax, label='P(y=1|x)')

        # Plot decision boundary (P=0.5 contour)
        ax.contour(xx1, xx2, Z, levels=[0.5],
                   colors='black', linewidths=2, linestyles='--')

        # Plot data points
        for cls, color, label in zip([0, 1], ['#2196F3', '#F44336'],
                                      ['Class 0', 'Class 1']):
            mask = (y == cls)
            ax.scatter(X[mask, 0], X[mask, 1], c=color, label=label,
                       edgecolors='white', linewidths=0.5, s=50, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.legend()
        return ax


# ============================================================
# MAIN SCRIPT: DEMONSTRATION OF FUNCTIONALITY
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # EXPERIMENT 1
    # Simple 2D linearly separable-ish dataset
    # Train logistic regression with L2 regularization
    # using full-batch gradient descent
    # ========================================================
    print("=" * 60)
    print("EXPERIMENT 1: Linear Dataset, Full Batch GD")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Generate a synthetic binary classification dataset
    # --------------------------------------------------------
    # n_samples=600      -> total number of data points
    # n_features=2       -> only 2 features so we can visualize boundary
    # n_informative=2    -> both features carry class information
    # n_redundant=0      -> no redundant linear combinations
    # class_sep=1.5      -> moderate separation between classes
    X, y = make_classification(
        n_samples=600, n_features=2, n_informative=2, n_redundant=0,
        class_sep=1.5, random_state=SEED
    )

    # --------------------------------------------------------
    # Step 2: Standardize features
    # --------------------------------------------------------
    # Logistic regression itself does not require scaling mathematically,
    # but gradient descent optimization behaves much better when feature
    # scales are similar.
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # --------------------------------------------------------
    # Step 3: Train/test split
    # --------------------------------------------------------
    # test_size=0.25 -> 25% of data reserved for testing
    # stratify=y     -> preserve class proportions in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # --------------------------------------------------------
    # Step 4: Create model
    # --------------------------------------------------------
    # lr=0.5          -> fairly aggressive learning rate
    # n_iter=500      -> maximum of 500 gradient steps
    # penalty='l2'    -> ridge-type shrinkage
    # C=1.0           -> moderate regularization strength
    # batch_size=None -> full-batch gradient descent
    # tol=1e-6        -> stop if gradient norm becomes tiny
    # verbose=100     -> print progress every 100 iterations
    model = LogisticRegressionScratch(
        lr=0.5, n_iter=500, penalty='l2', C=100.0,
        batch_size=None, tol=1e-6, verbose=100
    )

    # --------------------------------------------------------
    # Step 5: Train model
    # --------------------------------------------------------
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Step 6: Report learned parameters
    # --------------------------------------------------------
    print(f"\nLearned beta_0 (intercept): {model.beta_[0]:.4f}")
    print("Learned feature coefficients:")
    for j, coef in enumerate(model.beta_[1:], start=1):
        print(f"  beta{j}: {coef:.4f}")


    # --------------------------------------------------------
    # Step 7: Report metrics
    # --------------------------------------------------------
    # Accuracy measures label correctness.
    # Log loss measures probability quality.
    print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test  Accuracy: {model.score(X_test,  y_test):.4f}")
    print(f"Test  Log Loss: {log_loss(y_test, model.predict_proba(X_test)[:, 1]):.4f}")


    # --------------------------------------------------------
    # Step 8: Plot diagnostics
    # --------------------------------------------------------
    if X_test.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        model.plot_loss_curve(axes[0])
        model.plot_decision_boundary(
            X_test, y_test, axes[1],
            title="Decision Boundary (L2, C=1.0)"
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        model.plot_loss_curve(ax)
        print(f"Decision boundary visualization skipped because n_features = {X_test.shape[1]}.")

    plt.tight_layout()
    plt.savefig("scratch_linear.png", dpi=150, bbox_inches='tight')
    # plt.show()

    # ========================================================
    # EXPERIMENT 2
    # Compare regularization choices over a broad range of C and learning rates
    # ========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Compare L1 vs L2 vs ElasticNet over C and learning rates")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Create a higher-dimensional dataset
    # --------------------------------------------------------
    from sklearn.datasets import make_classification
    X_hd, y_hd = make_classification(
        n_samples=400,
        n_features=20,
        n_informative=4,
        n_redundant=6,
        random_state=SEED
    )

    scaler_hd = StandardScaler()
    X_hd_sc = scaler_hd.fit_transform(X_hd)

    Xtr_h, Xte_h, ytr_h, yte_h = train_test_split(
        X_hd_sc, y_hd, test_size=0.25, random_state=SEED
    )

    # --------------------------------------------------------
    # Step 2: Define parameter ranges for a systematic sweep
    # --------------------------------------------------------
    C_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    penalties = [None, 'l2', 'l1', 'elasticnet']
    elasticnet_l1_ratios = [0.2, 0.5, 0.8]

    configs = []

    for penalty in penalties:
        for C in C_values:
            for lr in lr_values:
                if penalty is None:
                    if C != 1.0:
                        continue
                    configs.append({
                        'penalty': None,
                        'C': 1.0,
                        'lr': lr,
                        'l1_ratio': 0.5,
                        'label': f'None | lr={lr}'
                    })
                elif penalty == 'elasticnet':
                    for rho in elasticnet_l1_ratios:
                        configs.append({
                            'penalty': 'elasticnet',
                            'C': C,
                            'lr': lr,
                            'l1_ratio': rho,
                            'label': f'elasticnet | C={C} | lr={lr} | rho={rho}'
                        })
                else:
                    configs.append({
                        'penalty': penalty,
                        'C': C,
                        'lr': lr,
                        'l1_ratio': 0.5,
                        'label': f'{penalty} | C={C} | lr={lr}'
                    })

    print(f"Total number of configs = {len(configs)}")

    # --------------------------------------------------------
    # Step 3: Train one model per configuration
    # --------------------------------------------------------
    results = []

    for cfg in configs:
        m = LogisticRegressionScratch(
            lr=cfg['lr'],
            n_iter=5000,
            verbose=0,
            penalty=cfg['penalty'],
            C=cfg['C'],
            l1_ratio=cfg['l1_ratio']
        )

        m.fit(Xtr_h, ytr_h)

        train_acc = m.score(Xtr_h, ytr_h)
        test_acc = m.score(Xte_h, yte_h)
        test_logloss = log_loss(yte_h, m.predict_proba(Xte_h)[:, 1])
        n_zeros = np.sum(np.abs(m.beta_[1:]) < 1e-4)

        results.append({
            'penalty': cfg['penalty'],
            'C': cfg['C'],
            'lr': cfg['lr'],
            'l1_ratio': cfg['l1_ratio'],
            'label': cfg['label'],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_logloss': test_logloss,
            'n_zeros': n_zeros,
            'beta': m.beta_.copy()
        })

        print(f"{cfg['label']:45s} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"LogLoss: {test_logloss:.4f} | "
            f"Zero coeffs: {n_zeros}/{len(m.beta_)-1}")

    # --------------------------------------------------------
    # Step 4: Print best-performing configurations
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("TOP 10 CONFIGS BY TEST ACCURACY")
    print("-" * 80)

    results_sorted_acc = sorted(results, key=lambda d: d['test_acc'], reverse=True)
    for r in results_sorted_acc[:10]:
        print(f"{r['label']:45s} | "
            f"Test Acc: {r['test_acc']:.4f} | "
            f"LogLoss: {r['test_logloss']:.4f} | "
            f"Zeros: {r['n_zeros']}")

    print("\n" + "-" * 80)
    print("TOP 10 CONFIGS BY LOWEST TEST LOG LOSS")
    print("-" * 80)

    results_sorted_loss = sorted(results, key=lambda d: d['test_logloss'])
    for r in results_sorted_loss[:10]:
        print(f"{r['label']:45s} | "
            f"Test Acc: {r['test_acc']:.4f} | "
            f"LogLoss: {r['test_logloss']:.4f} | "
            f"Zeros: {r['n_zeros']}")

    # --------------------------------------------------------
    # Step 5: Plot trends vs C
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    grouped = {}

    for r in results:
        if r['penalty'] == 'elasticnet':
            key = f"elasticnet | lr={r['lr']} | rho={r['l1_ratio']}"
        else:
            key = f"{r['penalty']} | lr={r['lr']}"
        grouped.setdefault(key, []).append(r)

    for key, vals in grouped.items():
        vals = sorted(vals, key=lambda d: d['C'])

        C_plot = [v['C'] for v in vals]
        acc_plot = [v['test_acc'] for v in vals]
        loss_plot = [v['test_logloss'] for v in vals]
        zero_plot = [v['n_zeros'] for v in vals]

        if 'None' in key or 'none' in key:
            axes[0].axhline(acc_plot[0], linestyle='--', linewidth=1.2, label=key)
            axes[1].axhline(loss_plot[0], linestyle='--', linewidth=1.2, label=key)
            axes[2].axhline(zero_plot[0], linestyle='--', linewidth=1.2, label=key)
        else:
            axes[0].plot(C_plot, acc_plot, marker='o', linewidth=1.5, label=key)
            axes[1].plot(C_plot, loss_plot, marker='o', linewidth=1.5, label=key)
            axes[2].plot(C_plot, zero_plot, marker='o', linewidth=1.5, label=key)

    for ax in axes:
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel("C (log scale)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy vs C")

    axes[1].set_xlabel("C (log scale)")
    axes[1].set_ylabel("Test Log Loss")
    axes[1].set_title("Test Log Loss vs C")

    axes[2].set_xlabel("C (log scale)")
    axes[2].set_ylabel("# Near-Zero Coefficients")
    axes[2].set_title("Sparsity vs C")

    axes[0].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("scratch_regularization_sweep.png", dpi=150, bbox_inches='tight')
    # plt.show()

    # ========================================================
    # EXPERIMENT 3
    # Compare optimization behavior under different batch sizes
    # ========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Batch Size Comparison")
    print("=" * 60)

    # --------------------------------------------------------
    # Different optimization modes:
    # None -> full batch
    # 64   -> medium mini-batch
    # 16   -> smaller mini-batch
    # 1    -> SGD
    # --------------------------------------------------------
    batch_configs = [
        (None,  'Full Batch GD'),
        (64,    'Mini-batch (n=64)'),
        (16,    'Mini-batch (n=16)'),
        (1,     'SGD (n=1)'),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Just for visually distinct curves
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']

    # --------------------------------------------------------
    # Train one model per batch size
    # --------------------------------------------------------
    for (bs, label), color in zip(batch_configs, colors):
        m = LogisticRegressionScratch(
            lr=0.2, n_iter=300, penalty='l2', C=1.0,
            batch_size=bs, tol=1e-8, verbose=0
        )

        m.fit(Xtr_h, ytr_h)

        # Plot the full-data loss recorded during training
        ax.plot(m.loss_history_, label=label, color=color,
                lw=1.5, alpha=0.85)

    
    # --------------------------------------------------------
    # Finalize plot
    # --------------------------------------------------------
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Convergence: Different Batch Sizes\n"
                 "(Note SGD noise vs smooth full-batch descent)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scratch_batch_convergence.png", dpi=150, bbox_inches='tight')
    # plt.show()