# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:35:17 2012

@author: anand
"""

import matplotlib.pyplot as plt
from numpy import sin, arange
import math

def generateF1(amp, freq, simtime):
    f1t = (amp/1.0)*sin(1.0*math.pi*freq*simtime) + \
         (amp/2.0)*sin(2.0*math.pi*freq*simtime) + \
         (amp/6.0)*sin(3.0*math.pi*freq*simtime)
    f1t = f1t / 1.5
    return f1t
    
def generateF2(amp, freq, simtime):
    f2t = (amp/1.0)*sin(1.0*math.pi*freq*simtime) + \
         (amp/6.0)*sin(3.0*math.pi*freq*simtime) + \
         (amp/3.0)*sin(4.0*math.pi*freq*simtime)
    f2t = f2t / 1.5
    return f2t
    
def generateF3(amp, freq, simtime):
    f3t = (amp/1.0)*sin(1.0*math.pi*freq*simtime) + \
         (amp/2.0)*sin(2.0*math.pi*freq*simtime) + \
         (amp/3.0)*sin(4.0*math.pi*freq*simtime)
    f3t = f3t / 1.5
    return f3t
    
def testplot(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()
    
def testpassing(a):
    a[0] = 10
    
if __name__=="__main__":
    amp = 1.3
    freq = 1/60.0
    nsecs = 1440
    dt = 0.1
    simtime = arange(0,nsecs-dt,dt)
    
    plt.plot(simtime, generateF1(amp, freq, simtime))
    plt.xlabel("time")
    plt.ylabel("f")
    plt.title("f1")
    plt.savefig("f1.png", bbox_inches='tight')
    
    plt.figure()
    plt.plot(simtime, generateF2(amp, freq, simtime))
    plt.xlabel("time")
    plt.ylabel("f")
    plt.title("f2")
    plt.savefig("f2.png", bbox_inches='tight')
    
    plt.figure()
    plt.plot(simtime, generateF3(amp, freq, simtime))
    plt.xlabel("time")
    plt.ylabel("f")
    plt.title("f3")
    plt.savefig("f3.png", bbox_inches='tight')