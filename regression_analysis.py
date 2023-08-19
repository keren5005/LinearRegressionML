from sklearn.metrics import r2_score
from linear_regression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

def simple_regression_dataset():
    
    # Load data
    df = pd.read_csv('simple_regression.csv')

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2)

    # Reshape the data
    x_train = x_train.values.reshape(-1, 1)
    y_train = y_train.values
    x_test = x_test.values.reshape(-1, 1)
    y_test = y_test.values

    # Fit the model to the training data
    lr = LinearRegression(learning_rate=0.01, n_epochs=10000, threshold=0.0001)
    lr.fit(x_train, y_train)

    # Evaluate the model on the testing data
    y_pred = lr.predict(x_test)
    r2 = lr.score(y_test, y_pred)

    print(f"Weights: {lr.weights_}")
    print(f"R2 score: {r2}")

#performs Linear regression on a dataset of california housing with test and train
def run_california():
    # Load the data
    x, y = fetch_california_housing(return_X_y=True)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit the model on the training set
    lr = LinearRegression(learning_rate=1 / 1e+12, n_epochs=1000000, threshold=0.0001)
    lr.fit(x_train, y_train)

    # Evaluate the model on the test set
    print(lr.weights_)
    print(f'Training R2 = {lr.score(y_train, lr.predict(x_train))}')
    print(f'Test R2 = {lr.score(y_test, lr.predict(x_test))}')

#performs polynomial regression on a dataset of student test scores on Mars
def run_students_on_mars():
    np.seterr(all="ignore")
    df = pd.read_csv('Students_on_Mars.csv')
    y = df['y'].values
    x = df.drop(['y'], axis=1).values

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression(learning_rate=1 / 1e+12, n_epochs=1000000, threshold=0.0001)
    powers = []
    train_scores = []
    test_scores = []
    for n in range(1, 4):
        #transform the input features into a polynomial basis. The resulting features are then fitted to the training
        # data using fit_transform, and the model is trained on the transformed features using lr.fit
        poly = PolynomialFeatures((1, n), include_bias=False)
        x2_train = poly.fit_transform(x_train)
        lr.fit(x2_train, y_train)
        yhat_train = lr.predict(x2_train)
        r2_train = r2_score(y_train, yhat_train)

        # Evaluate on the testing set
        x2_test = poly.fit_transform(x_test)
        yhat_test = lr.predict(x2_test)
        r2_test = r2_score(y_test, yhat_test)

        #The polynomial degree and corresponding R-squared scores on the training and testing sets
        powers.append(n)
        train_scores.append(1- r2_train)
        test_scores.append(1- r2_test)

    plt.scatter(powers,  train_scores,marker='X', label='Train R-squared')
    plt.scatter(powers, test_scores,marker='*', label='Test R-squared')
    plt.xlabel('Degree')
    plt.ylabel('R-squared')
    plt.legend()
    plt.show()

    n_idx = np.argmin(test_scores)
    n = powers[n_idx]
    poly = PolynomialFeatures((1, n), include_bias=False)
    x2_train = poly.fit_transform(x_train)
    lr.fit(x2_train, y_train)
    yhat_train = lr.predict(x2_train)
    r2_train = r2_score(y_train, yhat_train)

    # Evaluate on the testing set
    x2_test = poly.fit_transform(x_test)
    yhat_test = lr.predict(x2_test)
    r2_test = r2_score(y_test, yhat_test)

    print(f'Degree: {n}')
    print(f'Training Set - R2: {r2_train}')
    print(f'Testing Set  - R2: {r2_test}')
    print(f'Learned weights  - w: {lr.weights_}')

if __name__ == '__main__':
    print("Activation of custom linear regression class - 1st dataset")
    simple_regression_dataset()
    print("=======================================================")
    print("Activation of custom linear regression class - 2nd dataset")
    run_california()
    print("=======================================================")
    print("Using Polynomial Features to change the linear regression to polinomial regression")
    run_students_on_mars()

