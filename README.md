le# Regression Analysis with Custom Linear Regression and Polynomial Regression

This repository contains a Python script that demonstrates the implementation of a custom linear regression class, as well as the application of linear and polynomial regression techniques on different datasets. The script uses the `LinearRegression` class for linear regression and `PolynomialFeatures` for polynomial regression, both from the scikit-learn library.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Datasets](#datasets)
- [Custom Linear Regression](#custom-linear-regression)
- [Linear Regression on California Housing Dataset](#linear-regression-on-california-housing-dataset)
- [Polynomial Regression on Students on Mars Dataset](#polynomial-regression-on-students-on-mars-dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To use the provided script, follow the steps below:

1. Clone this repository to your local machine:

```sh
git clone https://github.com/keren5005/ml-linear-regression.git
```

2. Navigate to the repository's directory:

```sh
cd regression-analysis
```

## Prerequisites

Make sure you have Python (>=3.6) installed on your system. The required libraries can be installed using the instructions provided in the [Getting Started](#getting-started) section.

```sh
pip install -r requirements.txt
```

## Usage

The script `regression_analysis.py` demonstrates various regression techniques on different datasets. To execute the script, run the following command:

```sh
python regression_analysis.py
```

## Datasets

The script uses the following datasets for regression analysis:

1. **Simple Regression Dataset (`sr.csv`):** A simple dataset containing two columns, `x` and `y`. This dataset is used to demonstrate basic linear regression.

2. **California Housing Dataset:** A real-world dataset from scikit-learn's sample datasets. It contains features related to housing in California, used for linear regression analysis.

3. **Students on Mars Dataset (`Students_on_Mars.csv`):** A synthetic dataset simulating student test scores on Mars. This dataset is used to demonstrate polynomial regression.

## Custom Linear Regression

The script defines a custom linear regression class named `LinearRegression`, which is used for both simple and linear regression analyses. The class is instantiated with parameters like `learning_rate`, `n_epochs`, and `threshold`.

## Linear Regression on California Housing Dataset

The script performs linear regression on the California Housing Dataset. It splits the dataset into training and testing sets and evaluates the model's performance using R-squared scores.

## Polynomial Regression on Students on Mars Dataset

Polynomial regression is demonstrated on the Students on Mars Dataset. The script uses scikit-learn's `PolynomialFeatures` to transform the input features into a polynomial basis. Different polynomial degrees are tested, and their corresponding R-squared scores are plotted.

## Results

The script provides results for each regression analysis, including R-squared scores, learned weights, and the degree of polynomial for the best polynomial regression model.

## Contributing

Contributions to this repository are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
