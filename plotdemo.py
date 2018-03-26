# coding=gbk

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # ����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus'] = False  # ������ʾ����


def sin():
    import numpy as np
    x = np.linspace(0, 2*np.pi, 50)  # x��������
    y = np.sin(x)
    plt.plot(x,y,'bp--')
    plt.show()


def pie():
    labels = ['Fogs', 'Hogs', 'Dogs', 'Logs']
    sizes = [15,25,45,10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = [0,0.1,0,0]
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # ��ʾΪԲ
    plt.show()


def hist():
    import numpy as np
    x = np.random.randn(1000)
    plt.hist(x,10)  # �ֳ�10��
    plt.show()
