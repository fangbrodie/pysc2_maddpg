import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)
y = csv.reader(open('reapers_loss.csv'))
loss_n = []
for loss in y:
    loss = [float(i) for i in loss]
    loss_n.append(loss)

loss_n = np.array(loss_n)

length = len(loss_n)
x = range(length)

plt.subplot(611)
plt.plot(x, loss_n[:, 0], 'r-') #红色实线
plt.title('q_loss')

plt.subplot(612)
plt.plot(x, loss_n[:, 1], 'b^')#蓝色三角形
plt.title('p_loss')

plt.subplot(613)
plt.plot(x, loss_n[:, 2], 'go')#绿色圆形
plt.title('mean_target_q')

plt.subplot(614)
plt.plot(x, loss_n[:, 3], 'r-') #红色实线
plt.title('mean_rew')

plt.subplot(615)
plt.plot(x, loss_n[:, 4], 'b^')#蓝色三角形
plt.title('mean_target_q_next')

plt.subplot(616)
plt.plot(x, loss_n[:, 5], 'go')#绿色圆形
plt.title('std_target_q')

plt.xlabel('百局数', fontproperties=font)

plt.show()