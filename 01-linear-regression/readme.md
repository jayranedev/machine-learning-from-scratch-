# Linear Regression From Scratch

This project demonstrates simple linear regression without using a machine-learning framework. The notebook estimates a house's total price from its area in square feet using gradient descent implemented with NumPy.

## Project Structure

```text
01-linear-regression/
├── data/
│   └── House_Price_Data.csv
├── notebooks/
│   └── linear_regression.ipynb
└── readme.md
```

## Dataset

The dataset contains house listings with the following columns:

- `bhk`: Number of bedrooms, halls, and kitchens
- `propertytype`: Property type
- `location`: Property location
- `sqft`: Area in square feet
- `pricepersqft`: Price per square foot
- `totalprice`: Total property price in rupees

The notebook uses only `sqft` as the input feature and `totalprice` as the target. Prices are divided by `100000` so the target is represented in lakhs.

## Method

The notebook walks through these steps:

1. Load and inspect the CSV data with pandas.
2. Visualize the relationship between house area and price.
3. Remove observations with an area of `10,000` square feet or more, then keep observations below `4,500` square feet for the final model.
4. Convert the selected columns to NumPy arrays.
5. Standardize the area feature:

	```python
	x_scaled = (X - X.mean()) / X.std()
	```

6. Define the prediction function, squared-error cost function, and gradient calculation manually.
7. Train the model with gradient descent using:
	- Initial parameters: `w = 0`, `b = 0`
	- Learning rate: `0.01`
	- Iterations: `10,000`
8. Print the learned weight, bias, cost history, and parameter history.

The model is:

```text
prediction = w * standardized_sqft + b
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

Open `notebooks/linear_regression.ipynb` in Jupyter or VS Code and run the cells from top to bottom. Because the CSV path is relative to the notebook, run the notebook with `notebooks/` as its working directory, as VS Code and Jupyter normally do.

The notebook produces scatter plots before and after filtering and prints the model parameters found by gradient descent.

## Learning Goals

This exercise focuses on the mechanics behind linear regression:

- Representing a prediction function
- Measuring prediction error with squared-error cost
- Computing gradients for the weight and bias
- Updating parameters with gradient descent
- Improving optimization stability through feature scaling

## Notes

This is an educational, single-feature model. It does not split the data into training and test sets and does not use the other available dataset columns, so its results should not be treated as a production house-price estimator.
