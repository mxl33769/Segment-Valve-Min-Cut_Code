# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 00:25:13 2021

@author: mxl33
"""


import pandas as pd
import numpy as np

data = pd.read_excel('N.xlsx')
# print(type(data))
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
print(excel_list)

Result_K = pd.value_counts(excel_list)

s_1 = Result_K 
s_2 = s_1[s_1.values>3]

##############################################################################

#Visualization of results

import matplotlib.pyplot as plt

#设置输出的图片大小
figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)


#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 2,
}
 
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
 
#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

plt.xlabel('Segment',font2)

plt.ylabel('Number of times segment appears in minimum cuts',font2)
 
data = s_2
 
data. plot( kind='bar') 
