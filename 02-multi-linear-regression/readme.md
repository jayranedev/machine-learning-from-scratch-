# Multiple Linear Regression From Scratch

This chapter explores multiple linear regression with several input features. It includes a scikit-learn dataset exploration and a from-scratch gradient descent implementation using NumPy.

## Project Structure

```text
02-multi-linear-regression/
├── data/
│   ├── Advertising.csv
│   └── House_Price_Data.csv
├── notebooks/
│   ├── linear_regression.ipynb
│   └── linear_regression2.ipynb
└── readme.md
```

## Notebooks

### `linear_regression.ipynb`

This notebook uses the California housing dataset from `sklearn.datasets.fetch_california_housing`. It loads the data into a pandas DataFrame, explores the feature columns, creates visualizations, and examines correlations with the median house value.

### `linear_regression2.ipynb`

This notebook uses `Advertising.csv` to predict sales from advertising budgets:

- `TV`: TV advertising budget
- `Radio`: Radio advertising budget
- `Newspaper`: Newspaper advertising budget
- `Sales`: Target value

The model is trained from scratch with NumPy using a linear prediction function, squared-error cost, manually calculated gradients, and gradient descent.

The prediction function is:

```text
prediction = w_tv * TV + w_radio * Radio + w_newspaper * Newspaper + b
```

The `House_Price_Data.csv` file is included for future multiple-regression exercises.

## Requirements

- Python 3.9 or later
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook or the VS Code Jupyter extension

Install the dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Run the Notebooks

Open either notebook in Jupyter or VS Code and run the cells from top to bottom. The CSV paths in `linear_regression2.ipynb` are relative to the notebook directory, so run the notebook with `notebooks/` as its working directory, as VS Code and Jupyter normally do.

The notebooks currently focus on data exploration, visualization, model-cost calculations, gradient calculations, and parameter updates. Additional evaluation and user-input prediction features will be added as the chapter develops.

## Learning Goals

- Understand how multiple features contribute to one prediction
- Implement a multivariable prediction function with NumPy
- Calculate squared-error cost for multiple inputs
- Compute gradients for several weights and a bias
- Train a model with gradient descent
- Explore feature relationships with plots and correlation matrices

## Notes

This is an educational project. The current notebooks do not provide a complete production workflow, and they do not yet include a formal train/test split, cross-validation, feature scaling, or regularization.
