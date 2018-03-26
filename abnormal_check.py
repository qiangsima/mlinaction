#coding=gbk
import pandas as pd
catering_sale = './data/catering_sale.xls' #��������
data = pd.read_excel(catering_sale, index_col=u'����') #��ȡ���ݣ�ָ��������Ϊ������

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus'] = False #������ʾ����

plt.figure() #����ͼ��
p = data.boxplot(return_type='dict') #������ͼ
x = p['fliers'][0].get_xdata() #fliers��Ϊ�쳣ֵ�ı�ǩ
y = p['fliers'][0].get_ydata()
y.sort() #�÷���ֱ�Ӹı�ԭ����

#��annotate���ע��
#������Щ����ĵ㣬ע�������ص����Կ��壬��ҪһЩ����
for i in range(len(x)):
    if i > 0:
        plt.annotate(y[i], xy = (x[i],y[i]), xytext = (x[i]+0.05 -0.8/(y[i]-y[i-1]), y[i]))
    else:
        plt.annotate(y[i], xy = (x[i],y[i]), xytext = (x[i]+0.08,y[i]))


plt.show()
