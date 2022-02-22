from scipy import optimize
import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import cvxpy as cp

def maxProfit( prices: list[int]) -> int:
    n = len(prices)
    dp = [[0, -prices[0]]] + [[0, 0] for _ in range(n)]
    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[n - 1][0]

# 多项式拟合数据后返回目标天数预测值 黄金
def gold_cal_fit(start_day = 1 ,current_day = 1263,target_day = 1264):
    pd_reader = pd.read_csv("./LBMA-GOLD.csv")
    pd_reader.interpolate('linear')
    pd_reader['USD (PM)'].interpolate('linear')

    if pd_reader['USD (PM)'][start_day].astype(int) == -2147483648:
        pd_reader['USD (PM)'][start_day] = pd_reader['USD (PM)'][start_day-1]

    x = [x for x in range(start_day, current_day+1)]
    y = pd_reader['USD (PM)'][start_day:current_day+1]

    # null = y[579].astype(int)
    null = -2147483648
    # -2147483648 # float32 在nan时转化为int的值

    for i in range(start_day, current_day+1):
        if y[i] <=0 or y[i] > 10000 or y[i].astype(int) == null:
            y[i] = y[i-1]

    y_fit = np.polyfit(x, y, 40)

    # print('y_fit:', y_fit) # 多项式系数
    y_fit_1d = np.poly1d(y_fit) # 将多项式系数转换为多项式
    # print('y_fit_1d:\n', y_fit_1d) # 拟合多项式

    y_hat = np.polyval(y_fit, x) #计算结果
    # This function is also OK: y_hat = y_fit_1d(x)
    # print('y_hat:', y_hat)

    # print('Correlation coefficients:')
    # print(np.corrcoef(y_hat, y))

    # plot
    # plot1 = plt.plot(x, y, 'o', label='Original Values')
    # plot2 = plt.plot(x, y_hat, 'r', label='Fitting Curve')
    # plt.xlabel('Date')
    # plt.ylabel('USD')
    # plt.legend()
    # plt.title('trend fitting')
    # plt.show()

    # print("fitting:",y_fit_1d(target_day))
    # y = []

    y = pd_reader['USD (PM)']
    for i in range(start_day, current_day+3):
        if y[i] <=0 or y[i] > 10000 or y[i].astype(int) == null:
            y[i] = y[i-1]

    # print("original",y[target_day])
    # print("差值", y[target_day] - y_fit_1d(target_day) )
    return y_fit_1d(target_day), y[target_day] #返回预测值与目标值


# 黄金
def gold_cal_action(day=350):
    total = 0.6722 +0.789+0.8356+0.8759+0.8348
    fit, target = gold_cal_fit(start_day=0, current_day= day, target_day= day+1)
    fit60, target60 = gold_cal_fit(start_day=day-60, current_day=day, target_day=day + 1)
    fit90, target90 = gold_cal_fit(start_day=day-90, current_day=day, target_day=day + 1)
    fit150, target150 = gold_cal_fit(start_day=day - 150, current_day=day, target_day=day + 1)
    fit200, target200 = gold_cal_fit(start_day=day-200, current_day=day, target_day=day + 1)
    predict = abs(fit) *0.6722 + abs(fit60) *0.789 + abs(fit90)* 0.8356 + abs(fit200) *0.8759 + abs(fit150) * 0.8348
    predict = predict/total
    # print("pre", predict)
    # print("target", target)
    # print("acc:", 1 - abs(predict - target) / target)

    return predict, target


# 多项式拟合数据后返回目标天数预测值 比特币
def BTC_cal_fit(start_day = 1 ,current_day = 1824,target_day = 1825):
    pd_reader = pd.read_csv("./BCHAIN-MKPRU.csv")
    x = [x for x in range(start_day, current_day+1)]
    y = pd_reader['Value'][start_day:current_day+1]
    if pd_reader['Value'][start_day].astype(int) == -2147483648:
        pd_reader['Value'][start_day] = pd_reader['Value'][start_day-1]

    # null = y[579].astype(int)
    null = -2147483648
    # -2147483648 # float32 在nan时转化为int的值

    for i in range(start_day, current_day+1):
        if y[i] <= 0 or y[i].astype(int) == null:
            y[i] = y[i-1]

    y_fit = np.polyfit(x, y, 40)

    # print('y_fit:', y_fit) # 多项式系数
    y_fit_1d = np.poly1d(y_fit) # 将多项式系数转换为多项式
    # print('y_fit_1d:\n', y_fit_1d) # 拟合多项式

    y_hat = np.polyval(y_fit, x) #计算结果
    # This function is also OK: y_hat = y_fit_1d(x)
    # print('y_hat:', y_hat)

    # print('Correlation coefficients:')
    # print(np.corrcoef(y_hat, y))

    # plot
    # plot1 = plt.plot(x, y, 'o', label='Original Values')
    # plot2 = plt.plot(x, y_hat, 'r', label='Fitting Curve')
    # plt.xlabel('Date')
    # plt.ylabel('USD')
    # plt.legend()
    # plt.title('trend fitting')
    # plt.show()

    # print("fitting:",y_fit_1d(target_day))
    y = []
    for usd in pd_reader['Value']:
        y.append(usd)

    # print("original",y[target_day])
    # print("差值", y[target_day] - y_fit_1d(target_day) )
    return y_fit_1d(target_day), y[target_day] #返回预测值与目标值


# 比特币
def BTC_cal_action(day=350):
    total = 0.728 +0.8719546742209632+0.8778097982708933+0.9187692307692308+0.9158208955223881
    fit, target = BTC_cal_fit(start_day=0, current_day= day, target_day= day+1)
    fit60, target60 = BTC_cal_fit(start_day=day-60, current_day=day, target_day=day + 1)
    fit90, target90 = BTC_cal_fit(start_day=day-90, current_day=day, target_day=day + 1)
    fit150, target150 = BTC_cal_fit(start_day=day - 150, current_day=day, target_day=day + 1)
    fit200, target200 = BTC_cal_fit(start_day=day-200, current_day=day, target_day=day + 1)
    predict = abs(fit) * 0.728 + abs(fit60) *0.8719546742209632 + abs(fit90)* 0.8778097982708933 + abs(fit200) *0.9187692307692308 + abs(fit150) * 0.9158208955223881
    predict = predict/total
    # print("pre", predict)
    # print("target", target)
    # print("acc:", 1 - abs(predict - target) / target)

    return predict, target


def main():
    # 初始持有量
    cash = 1000  # 现金
    gold = 0  # 黄金
    BTC = 0  # 比特币
    # 第200天时的价格
    gold_price = 1249.55
    BTC_price = 1040.5755
    gold_brokerage = 0.01
    BTC_brokerage = 0.02
    n = 1  # 倍数
    # 概率密度函数参数
    mu_gold = 1.1641823385530574
    var_gold = 32.26779842376339
    mu_BTC = 26.10223110207152
    var_BTC = 907.0981884495216


    total_curve=[]
    for i in range(200, 1264):
        # 预测黄金、比特币价格，获得下一天的价格
        gold_p, gold_next_price = gold_cal_action(day=i)
        BTC_p, BTC_next_price = BTC_cal_action(day=i)
        # 可以加入马尔可夫模型
        try:
            print("gold next price",gold_next_price)
            # 分布函数
            x = sympy.Symbol('x')
            f1 = sympy.E ** (-(x + mu_gold - gold_p) ** 2 / (var_gold ** 2) * 2) / (var_gold * math.sqrt(2 * math.pi))
            f1_x = (sympy.E ** (-(x + mu_gold - gold_p) ** 2 / (var_gold ** 2) * 2) / (
                    var_gold * math.sqrt(2 * math.pi))) * x
            gold_return = sympy.integrate(f1_x, (x, gold_p, sympy.oo)).evalf() / \
                          sympy.integrate(f1, (x, gold_p, sympy.oo)).evalf() - gold_p
            gold_risk = sympy.integrate(f1_x, (x, -sympy.oo, gold_p)).evalf() / \
                        sympy.integrate(f1, (x, -sympy.oo, gold_p)).evalf() - gold_p
            print("gold return:",gold_return)
            print("gold risk:",gold_risk)
            f2 = sympy.E ** (-(x + mu_BTC - BTC_p) ** 2 / (var_BTC ** 2) * 2) / (var_BTC * math.sqrt(2 * math.pi))
            f2_x = (sympy.E ** (-(x + mu_BTC - BTC_p) ** 2 / (var_BTC ** 2) * 2) / (var_BTC * math.sqrt(2 * math.pi))) * x
            BTC_return = sympy.integrate(f2_x, (x, BTC_p, sympy.oo)).evalf() / \
                         sympy.integrate(f2, (x, BTC_p, sympy.oo)).evalf() - BTC_p
            BTC_risk = sympy.integrate(f2_x, (x, -sympy.oo, BTC_p)).evalf() / \
                       sympy.integrate(f2, (x, -sympy.oo, BTC_p)).evalf() - BTC_p
            print("btc return:",BTC_return)
            print("btc risk:",BTC_risk)

            # 分四种情况 预测的目标函数为求最大收益、最小风险函数的一部分 预测的holding为持有量
            result = []
            holding = []
            # c = np.array([-(1-gold_brokerage)*gold_p, -(1-BTC_brokerage)*BTC_p, -1])  # 目标函数
            # print("c:",c)
            # # 第一种：买入黄金 买入比特币
            # A_ub = np.array([[gold_price * (1 + gold_brokerage), BTC_price * (1 + BTC_brokerage), 1],
            #                  [gold_price * (1 + gold_brokerage), BTC_price * (1 + BTC_brokerage), -1],
            #                  [-(-gold_risk + n * gold_return), -(-BTC_risk + n * BTC_return), 0]])
            # b_ub = np.array([-1000,
            #                  gold_price * (1 + gold_brokerage) * gold + BTC_price * (1 + BTC_brokerage) * BTC,
            #                  0])
            # bounds = ([0, None], [0, None],[0, None])
            # res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
            # earnings = -res.fun - gold*(gold_return+gold_risk) - BTC*(BTC_return+BTC_risk)
            # holding.append(res.x)
            # result.append(earnings)
            # print("res:",res)
            # print("result:",result)

            ##################################################################################################
            ##################################################################################################
            # import cvxpy
            A_ub = np.array([[gold_price * (1 + gold_brokerage), BTC_price * (1 + BTC_brokerage)],
                             [-(-gold_risk + n * gold_return), -(-BTC_risk + n * BTC_return)]])
            b_ub = np.array([cash,
                             0])
            now = np.array([gold,BTC])
            x = cp.Variable(2) # 2个变量
            T = np.array([(gold_p - gold_price) * (1 - gold_brokerage), (BTC_p - BTC_price) * (1 - BTC_brokerage)])  # 目标函数
            prob = cp.Problem(cp.Maximize(T @ x), [A_ub@x <= b_ub, x + now >= 0])
            prob.solve()
            print("value:", prob.value)
            print("prob:",prob)
            print("gold:",gold_price," next: ", gold_next_price)
            print("i:",i)
            if(type(x) == 'NoneType'):
                x=np.array([0,0])
            print("x:",x.value)

            cash -= x.value[0] * gold_price*(1+gold_brokerage) + x.value[1] * BTC_price*(1+BTC_brokerage)
            gold += x.value[0]
            BTC += x.value[1]
        except:
            print("no fit value")
        print("gold:",gold)
        print("BTC:",BTC)
        print("cash:",cash)
        print("total", gold * gold_price+BTC*BTC_price+cash)
        total_curve.append(gold * gold_price+BTC*BTC_price+cash)

        #
        # c = np.array([-(gold_p - gold_price)*(1 - gold_brokerage), -(BTC_p - BTC_price) * (1-BTC_brokerage)])  # 目标函数
        # print("c:", c)
        # # 第一种：买入黄金 买入比特币
        # A_ub = np.array([[gold_price * (1 + gold_brokerage), BTC_price * (1 + BTC_brokerage)],
        #                  [-(-gold_risk + n * gold_return), -(-BTC_risk + n * BTC_return)]])
        # b_ub = np.array([cash,
        #                  0])
        # bounds = ([0, None], [0, None])
        # res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        # earnings = -res.fun
        #
        # print("res:",res)
        # print("result:",earnings)
        #


        gold_price = gold_next_price
        BTC_price = BTC_next_price

    total = cash + gold * gold_p + BTC * BTC_p  # 总资产
    print("The final result: ")
    print("cash:",cash)
    print("gold",gold)
    print("btc:",BTC)
    print("total",total)


    # plot
    plot1 = plt.plot([x for x in range(0, 1264)], total_curve, 'o', label='Original Values')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.title('total money')
    plt.show()


if __name__ == '__main__':
    main()
