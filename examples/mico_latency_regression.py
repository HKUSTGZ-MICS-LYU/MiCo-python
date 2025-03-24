import csv 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

TEST_SIZE = 0.8
RANDOM_SEED = 42

if __name__ == '__main__':
    
    with open('benchmark_results/mico_small.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    
    # [N,M,K,QA,QW,Time]
    data = np.array(data)
    print("Total Num of Data:", len(data))

    N = data[:, 0]
    M = data[:, 1]
    K = data[:, 2]

    QA = data[:, 3]
    QW = data[:, 4]

    latency = data[:, -1]

    # Direct Regression
    X = np.column_stack((N, M, K, QA, QW))
    y = latency

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("Direct Regression")
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # Regression with BOPs
    MACS = N * M * K
    BOPS = MACS * QA * QW
    X = BOPS.reshape(-1, 1)
    y = latency

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("Regression with BOPs")
    print("Coefficent:", reg.coef_.tolist())
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # Regression with Max BOPs
    MACS = N * M * K
    BOPS = MACS * np.max([QA, QW], axis=0)

    X = BOPS.reshape(-1, 1)
    y = latency

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("Regression with Max BOPs")
    print("Coefficent:", reg.coef_.tolist())
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # Regression with Composite Max BOPs
    MACS = N * M * K
    BOPS = MACS * np.max([QA, QW], axis=0)
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    X = np.column_stack((BOPS, W_LOADS, A_LOADS))
    y = latency

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("Regression with Composite Max BOPs")
    print("Coefficent:", reg.coef_.tolist())
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))