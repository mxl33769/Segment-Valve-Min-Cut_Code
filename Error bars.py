# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:16:44 2021

@author: mxl33
"""

import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(1,10,20)
dy=0.6
y=np.sin(x)*3

plt.errorbar(x,y,yerr=dy,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
plt.show()



x=np.linspace(1,10,20)
dy=np.random.rand(20)
y=np.sin(x)*3

plt.errorbar(x,y,yerr=dy,fmt='+',ecolor='r',color='b',elinewidth=2,capsize=4)
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
plt.show()
