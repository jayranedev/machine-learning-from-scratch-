# Linear Regression From Scratch

This project demonstrates simple linear regression without using a machine-learning framework. The latest notebook predicts salary from years of professional experience using gradient descent implemented with NumPy.

## Project Structure

```text
01-linear-regression/
├── data/
│   ├── House_Price_Data.csv
│   ├── Salary_Data.csv
│   └── study_hours_scores_regression.csv
├── notebooks/
│   ├── linear_regression.ipynb
│   ├── linear_regression_2.ipynb
│   └── linear_regression_3.ipynb
└── readme.md
```

## Current Dataset

The current notebook, `linear_regression_3.ipynb`, uses `Salary_Data.csv` with:

- `YearsExperience`: Years of professional experience
- `Salary`: Salary in rupees

The model uses `YearsExperience` as the input feature and `Salary` as the target. The study-hours and house-price datasets remain in the `data/` directory as earlier exercises and possible future experiments.

## Method

The current notebook walks through these steps:

1. Load and inspect the salary dataset with pandas.
2. Convert the `YearsExperience` and `Salary` columns to NumPy arrays.
3. Visualize the data with a scatter plot.
4. Plot an initial linear prediction alongside the observed scores.
5. Define the prediction function, squared-error cost function, and gradient calculation manually.
6. Train the model with gradient descent using:
	- Initial parameters: `w = 0`, `b = 0`
	- Learning rate: `0.01`
	- Iterations: `10,000`
7. Stop early when the change in cost is below `1e-6`.
8. Plot the final regression line against the observed salaries.
9. Ask the user for years of experience and return a salary prediction.

The model is:

```text
prediction = w * years_experience + b
```

## Requirements

- Python 3.9 or later
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook or the VS Code Jupyter extension

Install the Python dependencies with:

```bash
pip install numpy pandas matplotlib jupyter
```

## Run the Notebook

Open `notebooks/linear_regression_3.ipynb` in Jupyter or VS Code and run the cells from top to bottom. Because the CSV path is relative to the notebook, run the notebook with `notebooks/` as its working directory, as VS Code and Jupyter normally do.

The notebook produces scatter plots, prints optimization progress, reports the learned parameters, displays the fitted regression line, and prompts for years of experience to generate a salary estimate.

## Learning Goals

This exercise focuses on the mechanics behind linear regression:

- Representing a prediction function
- Measuring prediction error with squared-error cost
- Computing gradients for the weight and bias
- Updating parameters with gradient descent
- Comparing predictions with observed values
- Using a stopping tolerance to end optimization early
- Applying the trained model to user-provided input

## Notes

This is an educational, single-feature model. It does not split the data into training and test sets, and it does not include model validation or regularization. Salary predictions are estimates based only on years of experience and should not be used as production compensation advice.
