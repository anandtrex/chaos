# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:35:17 2012

@author: anand
"""

import matplotlib.pyplot as plt
import numpy as np

def testplot(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()
    
def testpassing(a):
    a[0] = 10
    
if __name__=="__main__":
    a = np.array([3,4,5])
    print a
    testpassing(a)
    print a