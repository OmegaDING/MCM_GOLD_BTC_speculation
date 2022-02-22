import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gold_probability(start_day = 0 ,current_day = 1264):
    pd_reader = pd.read_csv("./LBMA-GOLD.csv")
    x = [x for x in range(start_day, current_day+1)]
    y = pd_reader['USD (PM)'][start_day:current_day+1]
    if pd_reader['USD (PM)'][start_day].astype(int) == -2147483648:
        pd_reader['USD (PM)'][start_day] = pd_reader['USD (PM)'][start_day-1]
    null = -2147483648
    for i in range(start_day, current_day + 1):
        if y[i] <= 0 or y[i] > 10000 or y[i].astype(int) == null:
            y[i] = y[i - 1]

    y_fit = np.polyfit(x, y, 40)
    y_fit_1d = np.poly1d(y_fit)  # 将多项式系数转换为多项式
    der = np.polyder(y_fit_1d, 1)
    der1 = der(x)
    trade = 0
    for i in der1:
        if abs(i) <= 0.3:
            trade += 1
    print(trade)
    print(trade/(current_day - start_day))

    # plot
    plt.plot(x, der1, 'c', label='der1')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.title('trend fitting')
    plt.show()
    return trade/(current_day - start_day)



def bit_probability(start_day = 0 ,current_day = 1825):
    pd_reader = pd.read_csv("./BCHAIN-MKPRU.csv")
    x = [x for x in range(start_day, current_day+1)]
    y = pd_reader['Value'][start_day:current_day+1]
    if pd_reader['Value'][start_day].astype(int) == -2147483648:
        pd_reader['Value'][start_day] = pd_reader['Value'][start_day-1]
    null = -2147483648
    for i in range(start_day, current_day + 1):
        if y[i] <= 0 or y[i] > 10000 or y[i].astype(int) == null:
            y[i] = y[i - 1]

    y_fit = np.polyfit(x, y, 40)
    y_fit_1d = np.poly1d(y_fit)  # 将多项式系数转换为多项式
    der = np.polyder(y_fit_1d, 1)
    der1 = der(x)
    trade = 0
    for i in der1:
        if abs(i) <= 1:
            trade += 1
    print(trade)
    print(trade/(current_day - start_day))

    # plot
    plt.plot(x, der1, 'c', label='Fitting Curve')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.title('trend fitting')
    plt.show()
    return trade/(current_day - start_day)

