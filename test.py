import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)
y = np.loadtxt('reapers_win_pro.csv').tolist()
length = len(y)
x = range(length)
plt.xlabel('局数', fontproperties=font)
plt.ylabel('胜率', fontproperties=font)
plt.plot(x, y, label='second line')

plt.show()