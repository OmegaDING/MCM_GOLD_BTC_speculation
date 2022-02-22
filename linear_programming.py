from scipy import optimize
import numpy as np
import sympy
import math

cash = 1000  # 现金
gold = 0  # 黄金
BTC = 0  # 比特币
gold_price = 1324.6
BTC_price = 621.65
gold_pre = 1323.65  # 预测黄金价格
BTC_pre = 609.67
total = cash + gold*gold_price + BTC*BTC_price  # 总资产
gold_brokerage = 0.01
BTC_brokerage = 0.02
n = 0.9

# 分布函数
x = sympy.Symbol('x')
f1 = (sympy.E**(-(x-1.164-gold_pre)**2/(32.267798**2)*2) / (32.267798*math.sqrt(2*math.pi))) * x
gold_return = sympy.integrate(f1, (x, gold_price, sympy.oo)).evalf()
gold_risk = sympy.integrate(f1, (x, -sympy.oo, gold_price)).evalf()
print(gold_return)
print(gold_risk)
f2 = (sympy.E**(-(x-26.1-BTC_pre)**2/(907.098**2)*2) / (907.098*math.sqrt(2*math.pi))) * x
BTC_return = sympy.integrate(f2, (x, BTC_price, sympy.oo)).evalf()
BTC_risk = sympy.integrate(f2, (x, -sympy.oo, BTC_price)).evalf()
print(BTC_return)
print(BTC_risk)

# 分四种情况 预测的目标函数为求最大收益、最小风险函数的一部分 预测的holding为持有量
result = []
holding = []
c = np.array([-(gold_return+gold_risk), -(BTC_return+BTC_risk)])  # 约束条件

# 第一种：买入黄金 买入比特币
A_ub = np.array([[gold_price/(1-gold_brokerage), BTC_price/(1-BTC_brokerage)], [-(gold_risk+n*gold_return), -(BTC_risk+n*BTC_return)]])
b_ub = np.array([cash+gold*gold_price/(1-gold_brokerage)+BTC*BTC_price/(1-BTC_brokerage), -(gold_risk*gold+BTC_risk*BTC+n*gold_return*gold+n*BTC_return*BTC)])
bounds = ((gold, None), (BTC, None))
res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
earnings = -res.fun - gold*(gold_return+gold_risk) - BTC*(BTC_return+BTC_risk)
holding.append(res.x)
result.append(earnings)

# 第二种：买入黄金 卖出比特币
A_ub = np.array([[gold_price/(1-gold_brokerage), BTC_price*(1-BTC_brokerage)], [-(gold_risk+n*gold_return), -(n*BTC_risk+BTC_return)]])
b_ub = np.array([cash+gold*gold_price/(1-gold_brokerage)+BTC*BTC_price*(1-BTC_brokerage), -(gold_risk*gold+n*BTC_risk*BTC+n*gold_return*gold+BTC_return*BTC)])
bounds = ((gold, None), (0, BTC))
res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
earnings = -res.fun - gold*(gold_return+gold_risk) - BTC*(BTC_return+BTC_risk)
holding.append(res.x)
result.append(earnings)

# 第三种：卖出黄金 买入比特币
A_ub = np.array([[gold_price*(1-gold_brokerage), BTC_price/(1-BTC_brokerage)], [-(n*gold_risk+gold_return), -(BTC_risk+n*BTC_return)]])
b_ub = np.array([cash+gold*gold_price*(1-gold_brokerage)+BTC*BTC_price/(1-BTC_brokerage), -(n*gold_risk*gold+BTC_risk*BTC+gold_return*gold+n*BTC_return*BTC)])
bounds = ((0, gold), (BTC, None))
res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
earnings = -res.fun - gold*(gold_return+gold_risk) - BTC*(BTC_return+BTC_risk)
holding.append(res.x)
result.append(earnings)

# 第四种：卖出黄金 卖出比特币
A_ub = np.array([[gold_price*(1-gold_brokerage), BTC_price*(1-BTC_brokerage)], [-(n*gold_risk+gold_return), -(n*BTC_risk+BTC_return)]])
b_ub = np.array([cash+gold*gold_price*(1-gold_brokerage)+BTC*BTC_price*(1-BTC_brokerage), -(n*gold_risk*gold+n*BTC_risk*BTC+gold_return*gold+BTC_return*BTC)])
bounds = ((0, gold), (0, BTC))
res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
earnings = -res.fun - gold*(gold_return+gold_risk) - BTC*(BTC_return+BTC_risk)
holding.append(res.x)
result.append(earnings)

print(result)  # 选择值最大的方案
print(holding)  # 顺序同上，根据result得知黄金及比特币持有量，进行更新
