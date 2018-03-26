# coding=gbk

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def sin():
    import numpy as np
    x = np.linspace(0, 2*np.pi, 50)  # x坐标输入
    y = np.sin(x)
    plt.plot(x,y,'bp--')
    plt.show()


def pie():
    labels = ['Fogs', 'Hogs', 'Dogs', 'Logs']
    sizes = [15,25,45,10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = [0,0.1,0,0]
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 显示为圆
    plt.show()


def hist():
    import numpy as np
    x = np.random.randn(1000)
    plt.hist(x,10)  # 分成10组
    plt.show()
