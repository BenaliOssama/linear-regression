# Linear Regression — Scikit-learn

A structured introduction to supervised learning and linear regression using Scikit-learn, NumPy, and Matplotlib. Built from first principles: from fitting a line through three points, to forecasting diabetes progression across 10 features.

---

## Exercises

### Exercise 0: Environment Setup
Set up the virtual environment with all required dependencies.

### Exercise 1: Scikit-learn Estimator
First contact with the Scikit-learn API. Fit a `LinearRegression` model on three data points, extract the coefficient and intercept, and predict for a new value.

**Core pattern:**
```python
model = LinearRegression()
model.fit(X, y)
model.predict([[4]])
```

### Exercise 2: Linear Regression in 1D
Generate synthetic data using `make_regression`, fit a line, visualise the result, and measure error using MSE. Repeated with two noise levels to observe the effect on fit quality.

- Noise = 10 → MSE ≈ 114
- Noise = 50 → MSE ≈ 2854

**MSE formula:**
```python
mse = np.mean((y_pred - y_true) ** 2)
```

### Exercise 3: Train/Test Split
Split a dataset into training and test sets using `train_test_split`. 80% for training, 20% for testing. Establishes the core principle: never evaluate a model on data it was trained on.

### Exercise 4: Forecast Diabetes Progression
Apply linear regression to a real-world medical dataset. 442 patients, 10 baseline features (age, BMI, blood pressure, serum measurements), target is disease progression after one year.

- Train MSE ≈ 2888
- Test MSE ≈ 2858

Similar train and test MSE indicates the model generalised well — no overfitting.

---

## Key Concepts

**Linear equation (multidimensional):**
```
y = a1*x1 + a2*x2 + ... + a10*x10 + b
```
Scikit-learn finds the coefficients `a1...a10` and intercept `b` that minimise MSE across all training samples.

**MSE (Mean Squared Error):** Average of squared differences between predicted and true values. Lower is better. Squaring ensures positive and negative errors don't cancel, and penalises large errors more heavily.

**Train/Test split:** The model learns on the train set and is evaluated on the test set — data it has never seen. This gives an honest measure of generalisation.

**R² (Score):** Proportion of variance in y explained by the model. Ranges from 0 (explains nothing) to 1 (perfect fit).

**Overfitting:** When train MSE is much lower than test MSE — the model memorised the training data instead of learning general patterns.

---

## Setup

```bash
python -m venv ex00
source ex00/bin/activate
pip install pandas numpy jupyter matplotlib scikit-learn
```

**Requirements:**
```
pandas
numpy
jupyter
matplotlib
scikit-learn
```

---

## Project Structure

```
linear-regression/
├── ex00/
│   ├── requirements.txt
│   └── Notebook_ex00.ipynb
├── ex01/
│   └── ex01.py
├── ex02/
│   └── ex02.py
├── ex03/
│   └── ex03.py
├── ex04/
│   └── ex04.py
└── ex05/
    └── ex05.py
```
