import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read a file and returns the Matrix of independent variables (X) and the dependent variable y.
def fileread(filename):

    col_names = [ "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    data = pd.read_csv(filename,names=col_names)
    n, p = data.shape
    data = pd.DataFrame(np.c_[np.ones(n), data])    # adds a column with value 1 as the first column corresponding the intercept
    y = data.iloc[:, -1]
    X = data.iloc[:, 0:-1]

    return X, y


# returns the coefficient vector b which includes the intercept as the first value.
def LR_coefficient(X, y):

    XT = X.transpose()

    XTX = np.matmul(XT, X)

    XTXI = np.linalg.inv(XTX)

    XTy = np.matmul(XT, y)

    b = np.matmul(XTXI, XTy)

    print("Coefficients b:")
    temp = ["{:0.3f}".format(x) for x in b]
    print(b)
    print(temp)
    return b


# Predict the value of the dependent variable using the independent variable and their coefficient (including intercept)
def LR_predict(X, b):

    return np.matmul(X, b)


# Calculate the RMSE of the LR equation using the observed(y) and predicted(y_pred) values.
def RMSE(y, y_pred):

    return np.sqrt(((y - y_pred)**2).mean())


def graph_LR(y_test, y_pred):
    plt.plot(y_pred, y_test, 'o')
    plt.title("Fig. 1 Prediction vs Ground Truth", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Ground-truth")
    plt.show()
    return

if __name__ == '__main__':

    train_file = "housing_training.csv"
    X_train, y_train = fileread(train_file)

    b = LR_coefficient(X_train, y_train)

    test_file = "housing_test.csv"
    X_test, y_test = fileread(test_file)

    y_pred = LR_predict(X_test, b)
    print("\n Predicted y:")
    print(y_pred)

    print("\n RMSE = ", RMSE(y_test, y_pred))

    graph_LR(y_test, y_pred)






