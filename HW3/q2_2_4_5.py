import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def GradientDescent(train, test, r):
    probabilities = []
    length = len(train.drop("Prediction", axis=1).iloc[0]) + 1
    theta = np.zeros(length)
    theta[0] = 1
    true_pve = 0
    true_nve = 0
    false_pve = 0
    false_nve = 0
    for i in range(20):
        for index, row in train.iterrows():
            x = pd.concat([pd.Series([1]), row[:-1]])
            y = row[-1]
            sigmoid = 1 / (1 + np.exp(- (theta.T @ x)))
            gradient = x * float(sigmoid - y)
            theta = theta - r * gradient

    for index, row in test.iterrows():
        x = pd.concat([pd.Series([1]), row[:-1]])
        y = row[-1]
        sigmoid = 1 / (1 + np.exp(- (theta.T @ x)))
        probabilities.append(sigmoid)
        if sigmoid < 0.5:
            if y != 1:
                true_nve += 1
            else:
                false_nve += 1
        else:
            if y != 1:
                false_pve += 1
            else:
                true_pve += 1
    accuracy = (true_pve + true_nve) / (true_pve + true_nve + false_pve + false_nve)
    precision = true_pve / (true_pve + false_pve)
    recall = true_pve / (true_pve + false_nve)

    return probabilities, accuracy, precision, recall


def kNN(fold_train, fold_test, k):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    column = "Prediction"

    knn.fit(fold_train.drop(column, axis=1), fold_train[column])
    prediction = knn.predict(fold_test.drop(column, axis=1))
    accuracy = metrics.accuracy_score(fold_test[column], prediction)
    precision = metrics.precision_score(fold_test[column], prediction)
    recall = metrics.recall_score(fold_test[column], prediction)

    return accuracy, precision, recall


def OneNN(fold_train, fold_test, n):
    fold = kNN(fold_train, fold_test, 1)
    print("\nfold{}:\naccuracy: {}\nprecision: {}\nrecall: {}".format(n, fold[0], fold[1], fold[2]))


df = pd.read_csv("data/emails.csv")

column = "Email No."
single_train = df[:4000].drop(column, axis=1)
single_test = df[4000:].drop(column, axis=1)

fold1_train = df[1000:].drop(column, axis=1)
fold1_test = df[:1000].drop(column, axis=1)

fold2_train = df.iloc[np.r_[0:1000, 2000:5000]].drop(column, axis=1)
fold2_test = df[1000:2000].drop(column, axis=1)

fold3_train = df.iloc[np.r_[0:2000, 3000:5000]].drop(column, axis=1)
fold3_test = df[2000:3000].drop(column, axis=1)

fold4_train = df.iloc[np.r_[0:3000, 4000:5000]].drop(column, axis=1)
fold4_test = df[3000:4000].drop(column, axis=1)

fold5_train = df[:4000].drop(column, axis=1)
fold5_test = df[4000:].drop(column, axis=1)


OneNN(fold1_train, fold1_test, 1)
OneNN(fold2_train, fold2_test, 2)
OneNN(fold3_train, fold3_test, 3)
OneNN(fold4_train, fold4_test, 4)
OneNN(fold5_train, fold5_test, 5)

arr = [1, 3, 5, 7, 10]
mean_accuracy = []

for k in arr:
    fold1 = kNN(fold1_train, fold1_test, k)
    fold2 = kNN(fold2_train, fold2_test, k)
    fold3 = kNN(fold3_train, fold3_test, k)
    fold4 = kNN(fold4_train, fold4_test, k)
    fold5 = kNN(fold5_train, fold5_test, k)
    avg = np.mean([fold1[0], fold2[0], fold3[0], fold4[0], fold5[0]])
    mean_accuracy.append(avg)
    print("\nk = {}\nAverage accuracy:{}".format(k, avg))


plt.plot(arr, mean_accuracy, marker="o", color="red")
plt.title("kNN 5-Fold Cross validation")
plt.xlabel("k")
plt.ylabel("Average accuracy")
plt.grid()

plt.savefig("kNN.png")

rate = 0.1
column = "Prediction"

pred_logistic_reg = GradientDescent(single_train, single_test, rate)[0]
ROC_logistic_reg = metrics.roc_curve(single_test[column], pred_logistic_reg)
AUC_logistic_regression = round(metrics.roc_auc_score(single_test[column], pred_logistic_reg), 2)
txt1 = "LogisticRegression (AUC={})".format(AUC_logistic_regression)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(single_train.drop(column, axis=1), single_train[column])
prediction_knn = knn.predict_proba(single_test.drop(column, axis=1))
ROC_knn = metrics.roc_curve(single_test[column], prediction_knn[:, 1])
AUC_knn = round(metrics.roc_auc_score(single_test[column], prediction_knn[:, 1]), 2)
txt2 = "KNeighborsClassifier (AUC={})".format(AUC_knn)


plt.plot(ROC_logistic_reg[0], ROC_logistic_reg[1], color="red", label=txt1)
plt.plot(ROC_knn[0], ROC_knn[1], color="blue", label=txt2)

plt.xlabel("False Positive Rate (Positive label: 1)")
plt.ylabel("True Positive Rate (Positive label: 1)")
plt.legend()
plt.grid()

plt.savefig("Q5.png")


fold1 = GradientDescent(fold1_train, fold1_test, rate)
print("\nfold{}:\naccuracy: {}\nprecision: {}\nrecall: {}".format(1, fold1[1], fold1[2], fold1[3]))
fold2 = GradientDescent(fold2_train, fold2_test, rate)
print("\nfold{}:\naccuracy: {}\nprecision: {}\nrecall: {}".format(2, fold2[1], fold2[2], fold2[3]))
fold3 = GradientDescent(fold3_train, fold3_test, rate)
print("\nfold{}:\naccuracy: {}\nprecision: {}\nrecall: {}".format(3, fold3[1], fold3[2], fold3[3]))
fold4 = GradientDescent(fold4_train, fold4_test, rate)
print("\nfold{}:\naccuracy: {}\nprecision: {}\nrecall: {}".format(4, fold4[1], fold4[2], fold4[3]))
fold5 = GradientDescent(fold5_train, fold5_test, rate)
print("\nfold{}:\naccuracy: {}\nprecision: {}\nrecall: {}".format(5, fold5[1], fold5[2], fold5[3]))







