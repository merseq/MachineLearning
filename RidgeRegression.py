
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold      # import KFold
import csv


# returns the coefficient vector b given my
def RR_coefficient(X, y, l):
    XT = X.transpose()
    XTX = np.matmul(XT, X)

    n,p = XTX.shape
    lIMat = l * np.identity(n)

    XTXLIMat = np.linalg.inv(XTX + lIMat)
    XTy = np.matmul(XT, y)
    b = np.matmul(XTXLIMat, XTy)

    return b


# Predict the value of the dependent variable using the independent variable and their coefficient (including intercept)
def RR_predict(X, b):

    return np.matmul(X, b)


# Calculates RMSE of the whole dataset
def RMSE(y, y_pred):

    return np.sqrt(((y - y_pred)**2).mean())


# Perform coefficient determination of the test data multiple time with different lambda values and return the optimal one
def get_opt_lmda_RR(X, y, k):

    kf = KFold(n_splits=k)

    curr_RMSE, last_RMSE = 0, 0
    lmda = 0.0
    lmda_inc = 0.4

    # For each value of λ from .2 we calculate the RMSE and stop when RMSE increases again.
    while (curr_RMSE <= last_RMSE) or lmda < lmda_inc*2:
        sum_RMSE = 0
        lmda += lmda_inc

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            b = RR_coefficient(X_train, y_train, lmda)
            y_pred = RR_predict(X_test, b)

            sum_RMSE += RMSE(y_test, y_pred)

    #    print("sum :", sum_RMSE)
        last_RMSE = curr_RMSE
        curr_RMSE = sum_RMSE/k_val          #average RMSE for the λ of all the k-folds
    #    print("RMSE:", curr_RMSE)

    print("\nLowest RMSE :", last_RMSE)
    opt_lmda = lmda - lmda_inc

    return opt_lmda


# Calculate TPR, FPR and Accuracy for y_pred with given threshold value.
def get_eval_values(y_test, y_pred, theta):

    n = y_test.size
    true_pos = 0
    true_neg = 0

    cond_pos = (y_test == 1).sum()            # label = 1 (true value 5)
    cond_neg = (y_test == -1).sum()
    y_temp = y_pred.copy()                    # using temp so y_pred is not updated.
    y_temp[y_temp >= theta] = 1               # classifying as 1  (which is 5)
    y_temp[y_temp < theta] = -1               # classifying as -1 (which is 6)

    for each in range(n):
        if y_test[each] == 1:
            if y_temp[each] == 1:
                true_pos += 1
        else:
            if y_temp[each] == -1:
                true_neg += 1

    TPR = true_pos/cond_pos
    FPR = (cond_neg - true_neg)/cond_neg
    acc = (true_pos + true_neg)/n

    return TPR, FPR, acc


# Perform Ridge Regression on the k-folds of the dataset using the optimal lambda and different thresholds
# to classify the data and return the evaluation metrics

def generate_RR_eval(X, y, opt_lmda, k, theta_range):

    kf = KFold(n_splits=k)

    TPR_list = list()
    FPR_list = list()
    Acc_list = list()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        b = RR_coefficient(X_train, y_train, opt_lmda)
        y_pred = RR_predict(X_test, b)

        TPR_col_list = list()
        FPR_col_list = list()
        Acc_col_list = list()

        for theta in theta_range:
            temp = get_eval_values(y_test, y_pred, theta)
            TPR_col_list.append(temp[0])
            FPR_col_list.append(temp[1])
            Acc_col_list.append(temp[2])

        TPR_list.append(TPR_col_list)
        FPR_list.append(FPR_col_list)
        Acc_list.append(Acc_col_list)

    return np.array(TPR_list), np.array(FPR_list), np.array(Acc_list)

# Normalize the data
def normalize_data(X_data):

    norm_data = (X_data - X_data.min())/(X_data.max() - X_data.min())

    return norm_data

# Substitute the labels 5 as 1 and 6 as -1
def prep_labels(labels):

    labels.replace(to_replace=[5, 6], value=[1, -1], inplace=True)

    return labels.values


# Draw the ROC curve for the different folds
def graph_ROCs(x_values, y_values ):

    plt.figure("Figure 1")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)

    plt.title("ROC Curves for each k-fold", fontsize=14)

    x_rows = (x_values.shape)[0]

    labels = ['fold' + str(i+1) for i in range(0, x_rows)]

    for i in range(0, x_rows):
        plt.plot(x_values[i], y_values[i], label=labels[i])
    plt.legend(loc="lower right")
  #  plt.show()
    plt.savefig("ROCs_Merlyn.png", transparent=True)
    print("\n The ROC curve for all folds is saved in ROCs_Merlyn.png")

    return

# Draw the ROC curve for the whole dataset
def graph_ROC(x_values, y_values ):

    plt.figure("Figure 2")
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.title("ROC Curve of Ridge Regression", fontsize=14)

    plt.plot(x_values, y_values, color='blue', linewidth=2)
 #   plt.show()
    plt.savefig("ROC_Merlyn.png", transparent=True)
    print("\n The ROC curve for entire data is saved in ROC_Merlyn.png")
    return


if __name__ == '__main__':

    X_df = pd.read_csv('MNIST_15_15.csv', header=None)
    n = X_df.shape[0]
    X_df = X_df.loc[:, X_df.any()]      # Remove columns/features that have no value
    X_df = normalize_data(X_df)         # Normalize the columns
    X_df = pd.DataFrame(np.c_[np.ones(n), X_df])     # adding the intercept column
    X = X_df.values                     # Convert to 2d arrays

    label_df = pd.read_csv('MNIST_LABEL.csv', header=None)
    y = prep_labels(label_df)       # Convert labels 5, 6 to -1, 1

    k_val = 10           # K value for K-fold validation
    opt_lmda = get_opt_lmda_RR(X, y, k_val)     # get optimal lambda

    print("Optimal lmda:", opt_lmda)

    # Perform Ridge Regression on the k-folds of the dataset using the optimal lambda and different thresholds
    # to classify the data and return the evaluation metrics

    theta_range = [-10, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 10]

    # Get TPR and FPR values for different theta values
    TPR_array, FPR_array, Acc_array = generate_RR_eval(X, y, opt_lmda, k_val, theta_range)

    # Print the True Positive Rate & False Positive Rate

    folds = ['fold' + str(i+1) for i in range(0, k_val)]

    print("\n True Positive Rate for the k folds and average (last row) for different thresholds\n")
    TPR_df = pd.DataFrame(TPR_array, columns=theta_range, index=folds)
    TPR_df.loc['average'] = TPR_df.mean()
    print(TPR_df)

    print("\n False Positive Rate for the k folds and average (last row) for different thresholds")
    FPR_df = pd.DataFrame(FPR_array , columns=theta_range, index=folds)
    FPR_df.loc['average'] = FPR_df.mean()
    print(FPR_df)

    print("\n Accuracy for the k folds and average (last row) for different thresholds")
    Acc_df = pd.DataFrame(Acc_array , columns=theta_range, index=folds)
    Acc_df.loc['average'] = Acc_df.mean()
    print(Acc_df)
    print(Acc_df.loc['average'].values)
    # ROC curve for each of the folds
    graph_ROCs(FPR_array, TPR_array)

    # ROC curve for the entire dataset
    graph_ROC(FPR_df.loc['average'].values, TPR_df.loc['average'].values)


    # Write the TPR and FPR values to a csv file to make it easier to read
    print("\n The TPR and FPR values to a csv file, RR_Eval_Metrics.csv")
    csv_data = list()
    TPR_arrayT = TPR_array.transpose()
    FPR_arrayT = FPR_array.transpose()

    for i in range(TPR_arrayT.shape[0]):
        csv_data.append(TPR_arrayT[i])
        csv_data.append(FPR_arrayT[i])

    #  Convert rows to columns the TPR and FPR values
    csv_dataT = zip(*csv_data)
    header2 = ['TPR','FPR'] * len(theta_range)

    with open('RR_Eval_Metrics.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(theta_range)
        writer.writerow(header2)
        writer.writerows(csv_dataT)
        csvFile.close()
