# Linear Regression From Scratch

This project demonstrates simple linear regression without using a machine-learning framework. The current notebook predicts a student's final score from the number of hours studied using gradient descent implemented with NumPy.

## Project Structure

```text
01-linear-regression/
├── data/
│   ├── House_Price_Data.csv
│   └── study_hours_scores_regression.csv
├── notebooks/
│   ├── linear_regression.ipynb
│   └── linear_regression_2.ipynb
└── readme.md
```

## Current Dataset

The current notebook, `linear_regression_2.ipynb`, uses `study_hours_scores_regression.csv` with:

- `Hours`: Study time in hours
- `Scores`: Final score

The model uses `Hours` as the input feature and `Scores` as the target. The original house-price dataset remains in the `data/` directory for future experiments.

## Method

The current notebook walks through these steps:

1. Load and inspect the study-hours dataset with pandas.
2. Convert the `Hours` and `Scores` columns to NumPy arrays.
3. Visualize the data with a scatter plot.
4. Plot an initial linear prediction alongside the observed scores.
5. Define the prediction function, squared-error cost function, and gradient calculation manually.
6. Train the model with gradient descent using:
	- Initial parameters: `w = 0`, `b = 0`
	- Learning rate: `0.01`
	- Iterations: `10,000`
7. Stop early when the change in cost is below `1e-6`.
8. Inspect sample values, feature and target means, and the correlation between study hours and scores.

The model is:

```text
prediction = w * study_hours + b
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

Open `notebooks/linear_regression_2.ipynb` in Jupyter or VS Code and run the cells from top to bottom. Because the CSV path is relative to the notebook, run the notebook with `notebooks/` as its working directory, as VS Code and Jupyter normally do.

The notebook produces scatter plots, prints optimization progress, reports the learned parameters, and displays basic correlation information.

## Learning Goals

This exercise focuses on the mechanics behind linear regression:

- Representing a prediction function
- Measuring prediction error with squared-error cost
- Computing gradients for the weight and bias
- Updating parameters with gradient descent
- Comparing predictions with observed values
- Using a stopping tolerance to end optimization early

## Notes

This is an educational, single-feature model. It does not split the data into training and test sets, and it does not include model validation or regularization. The notebook notes a possible future transition to a Years of Experience versus Salary dataset, but that model has not been added yet.
