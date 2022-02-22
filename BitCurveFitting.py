import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 多项式拟合数据后返回目标天数预测值
def cal_fit(start_day = 1 ,current_day = 1824,target_day = 1825):
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
    plot1 = plt.plot(x, y, 'o', label='Original Values')
    plot2 = plt.plot(x, y_hat, 'r', label='Fitting Curve')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.title('trend fitting')
    plt.show()

    # print("fitting:",y_fit_1d(target_day))
    y = []
    for usd in pd_reader['Value']:
        y.append(usd)
    # print("original",y[target_day])
    # print("差值", y[target_day] - y_fit_1d(target_day) )
    return y_fit_1d(target_day), y[target_day] #返回预测值与目标值



# 用于计算分段均线可靠度
def cal_acc(thereshold = 0.30, start = 400, end = 1825):
    # 计算5，10，30日均线可靠程度
    result = []
    currect = 0
    for i in range(start, end):
        fit, target = cal_fit(i-400, i, i+1) #调整第一个参数以调整区间大小
        print("day:", i, "fit: ", fit, "target: ", target)
        if(abs(fit - target) > target * thereshold):
            result.append(False)
        else:
            result.append(True)
            currect += 1

    acc = currect / result.__len__()
    print("\nacc: ", acc)

# 整体拟合准确度
#  5日分段拟合 0.4
# 10日分段拟合 0.5427
# 30日分段拟合 0.8122562674094708
# 60日分段拟合 0.8719546742209632
# 90日分段拟合 0.8778097982708933
# 150日分段拟合 0.9158208955223881
# 200日分段拟合 0.9187692307692308
# 400日分段拟合 0.8421052631578947
# 从头整体拟合 0.728

def cal_action(threshold = 0.3, day=350):
    total = 0.728 +0.8719546742209632+0.8778097982708933+0.9187692307692308+0.9158208955223881
    fit, target = cal_fit(start_day=0, current_day= day, target_day= day+1)
    fit60, target60 = cal_fit(start_day=day-60, current_day=day, target_day=day + 1)
    fit90, target90 = cal_fit(start_day=day-90, current_day=day, target_day=day + 1)
    fit150, target150 = cal_fit(start_day=day - 150, current_day=day, target_day=day + 1)
    fit200, target200 = cal_fit(start_day=day-200, current_day=day, target_day=day + 1)
    predict = abs(fit) * 0.728 + abs(fit60) *0.8719546742209632 + abs(fit90)* 0.8778097982708933 + abs(fit200) *0.9187692307692308 + abs(fit150) * 0.9158208955223881
    predict = predict/total
    print("pre", predict)
    print("target", target)
    print("acc:", 1 - abs(predict - target) / target)

    if (abs(target-predict) > target * threshold):
        print("NO, not in threshold")
        result = False
    else:
        print("YES, in threshold")
        result = True

    return predict, result



# 测试从200 到1264数据集上误差5%以内的准确度 可以达到0.981203007518797
def acc_test():
    list = []
    corr=0
    for i in range(200,1264):
        _, result = cal_action(day=i)
        if result == True:
            list.append(True)
            corr += 1
        else:
            list.append(False)

    print("acc", corr/ list.__len__())

acc_test()