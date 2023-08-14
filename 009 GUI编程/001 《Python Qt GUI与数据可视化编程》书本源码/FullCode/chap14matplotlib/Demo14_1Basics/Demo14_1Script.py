## 程序文件： Demo14_1Script.py
## 使用matplotlib.pyplot 指令式绘图

import numpy as np

import matplotlib.pyplot as plt

plt.suptitle("Plot by pyplot API script")    #总的标题

t = np.linspace(0, 10, 40)   #[0,10]之间平均分布的40个数
y1=np.sin(t)   
y2=np.cos(2*t) 

plt.subplot(1,2,1)         #1行2列，第1个子图
plt.plot(t,y1,'r-o',label="sin", linewidth=1, markersize=5)
plt.plot(t,y2,'b:',label="cos",linewidth=2)    
plt.xlabel('time(sec)')    # X轴标题
plt.ylabel('value')        # Y轴标题
plt.title("plot of functions")  ##子图标题
plt.xlim([0,10])   # X轴坐标范围
plt.ylim([-2,2])   # Y轴坐标范围
plt.legend()       # 生成图例

plt.subplot(1,2,2)         #1行2列，第2个子图
week=["Mon","Tue","Wed","Thur","Fri","Sat","Sun"] 
sales=np.random.randint(20,70,7)    #生成7个[20,70)之间的整数
plt.bar(week,sales)     #柱状图
plt.ylabel('sales')     # Y轴标题
plt.title("Bar Chart")  #子图标题

plt.show()
