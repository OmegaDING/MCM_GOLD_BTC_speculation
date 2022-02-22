import pandas as pd
import pandas_datareader  # 用于从雅虎财经获取股票数据
import datetime
import scipy
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm

# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

stockFile = './LBMA-GOLD.csv'
stock = pd.read_csv(stockFile, index_col=0, parse_dates=[0])#将索引index设置为时间，parse_dates对日期格式处理为标准格式。


stock_train = stock['USD (PM)'].resample('3D').mean()
input(stock_train)

# 画原始数据图
stock_train.plot()
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("price")
sns.despine()

stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()

# 一阶差分图
plt.figure()
plt.plot(stock_diff)
plt.title('一阶差分')
plt.show()

acf = sm.graphics.tsa.plot_acf(stock_diff, lags=20)
plt.title("ACF")
acf.show()

pacf = sm.graphics.tsa.plot_pacf(stock_diff, lags=20)
plt.title("PACF")
pacf.show()

model = ARIMA(stock_train, order=(1, 1, 1),freq='')
result = model.fit()

pred = result.predict('20140609', '20160701',dynamic=True, typ='levels')#预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要


plt.figure(figsize=(6, 6))
plt.xticks(rotation=45)
plt.plot(pred)
plt.plot(stock_train)#[<matplotlib.lines.Line2D at 0x28025665278>]