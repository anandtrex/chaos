# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:49:06 2012

@author: anand
"""


# FORCE internal all to all (no feedback loop) with input that selects between 
# multiple outputs

import math
from sprandn import sprandn
from numpy import mat, zeros, random, sin, arange, tanh, eye, sqrt, tile
import matplotlib
matplotlib.use('GTkAgg')
import matplotlib.pyplot as plt
import time
plt.ion()


def generateInputWeights(N, M, mu = 0, sigma = 1):
    mat = zeros((N, M))
    for i in xrange(0, N):
       mat[i][random.randint(M)] = sigma * random.randn() + mu
       
    return mat
    
def generateMask(N):
    m = zeros(N)
    for i in xrange(0,N):
        if random.rand() < 0.2:
            m[i] = 1
    return m
    
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
    
if __name__ == "__main__":
    N = 1200
    Nin = 100
    Nselect = 3
    p = 0.8
    # pz = 0.8 Not sparse for now
    g = 2.0 # 1.5
    alpha = 80.0
    nsecs = 1440
    nsecspre = 500
    dt = 0.1
    learn_every = 2
    
    scale = 1.0/math.sqrt(p*N)
    M = sprandn(N,N,p)*g*scale
    M = mat(M.todense())
    
    inweights = random.rand(Nin)
    Min = generateInputWeights(N, Nin)

    nRec2Out = N;
    wo = mat(zeros((nRec2Out, 1)))
    dw = mat(zeros((nRec2Out, 1)))
    
    simtimepre = arange(0, nsecspre-dt, dt)
    simtimepre_len = len(simtimepre)
    simtime = arange(0,nsecs-dt,dt)
    simtime_len = len(simtime)
    simtime2 = arange(1*nsecs, 2*nsecs-dt, dt)
    
    amp = 2.0
    freq = 1/60.0
    f1t = generateF1(amp, freq, simtime)
    f2t = generateF2(amp, freq, simtime)
    f3t = generateF3(amp, freq, simtime)
    
    # Blend together three curves for initial learning
    fb = f1t + f2t + f3t
    
    wo_len = zeros(simtime_len)
    zt = zeros(simtime_len)
    zpt = zeros(simtime_len)
    
    x0 = 0.5 * mat(random.randn(N,1))
    z0 = 0.5 * mat(random.randn(1,1))
    
    x = x0
    r = tanh(x)
    z = z0
    
    # Indexing starts from 0!
    ti = -1
    P = (1.0/alpha)*mat(eye(nRec2Out))
    
    # train for f1
    I1 = mat(1 * random.rand(Nin,1) - 0.5)
    I2 = mat(1 * random.rand(Nin,1) - 0.5)
    I3 = mat(1 * random.rand(Nin,1) - 0.5)
    
    Ib = I1 + I2 + I3
    
    fig = plt.figure()
    plt.subplot(2,1,1)
    p1, = plt.plot(simtime, f1t)
    plt.hold(True)
    p2, = plt.plot(simtime, zt)
    plt.title('training')
    plt.legend(('f','z'));
    plt.xlabel('time')
    plt.ylabel('f and z')
    plt.hold(False)
    
    plt.subplot (2,1,2);
    p3, = plt.plot(simtime, wo_len)
    plt.xlabel('time')
    plt.ylabel('|w|')
    plt.legend('|w|')
    plt.draw()
    
    dynplot = True
    
    for t in simtime:
        ti += 1
        if ti % (nsecs/2) == 0:
            print "time:",str(t)
            if dynplot == True:
                p1.set_ydata(f1t)
                p2.set_ydata(zt)
                p3.set_ydata(wo_len)
                plt.draw()
    
        # sim, so x(t) and r(t) are created.
        x = (1.0 - dt) * x + M * (r * dt) + Min * I1
        r = tanh(x)
        z = wo.T * r
    
        if ti % learn_every == 0:
            # update inverse correlation matrix
            k = P * r
            rPr = r.T * k
            c = 1.0/(1.0 + rPr)
            c = c.A[0][0]
            P = P - k * (k.T * c)
    
            # update the error for the linear readout
            e = z - f1t[ti]
            e = e.A[0][0]
    
            # update the output weights
            dw = -e * k * c # k * c or P * r ??
            wo = wo + dw
            
            # update the internal weight matrix using the output's error
            M = M + tile(dw.T, (N, 1))
    
        zt[ti] = z
        wo_len[ti] = sqrt(wo.T*wo)
    
    error_avg = sum(abs(zt-f1t))/simtime_len;
    print "Training MAE:",str(error_avg)
    train_error_avg = error_avg
    
    plt.show()
    plt.figure()
    
    # For testing
    f1t2 = generateF1(amp, freq, simtime2)
    f2t2 = generateF2(amp, freq, simtime2)
    f3t2 = generateF3(amp, freq, simtime2)
    
    print "Now testing... please wait."
    ti = -1
    for t in simtime:
        ti+=1
        
        x = (1.0 - dt) * x + M * (r * dt) + Min  * I1
        r = tanh(x)
        z = wo.T * r
    
        zpt[ti] = z
    
    error_avg = sum(abs(zpt-f1t2))/simtime_len
    print "Testing MAE:",str(error_avg)
    test_error_avg = error_avg
    
    plt.subplot(211)
    plt.plot(simtime, f1t)
    plt.hold(True)
    plt.plot(simtime, zt)
    plt.title('training')
    plt.xlabel('time')
    plt.hold(True)
    plt.ylabel('f and z')
    plt.legend(('f', 'z'))
    
    
    plt.subplot(212)
    plt.hold(True)
    plt.plot(simtime2, f1t2)
    plt.axis('tight')
    plt.plot(simtime2, zpt)
    plt.axis('tight')
    plt.title('simulation')
    plt.xlabel('time')
    plt.ylabel('f and z')
    plt.legend(('f', 'z'))
    plt.show()
    tag = str(time.clock())
    
    # Write final errors and graphs to file
    plt.savefig("testing-"+tag+".png")
    f = open("testing-"+tag+".txt",'w')
    f.write("Training: "+str(train_error_avg)+"\n")
    f.write("Training: "+str(test_error_avg))
    f.close()
