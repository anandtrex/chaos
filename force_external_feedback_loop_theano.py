# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:54:48 2012

@author: anand
"""

# FORCE external feedback loop

import math
from sprandn import sprandn
from numpy import mat, zeros, random, sin, arange, tanh, eye, sqrt, float32, array
#import matplotlib
# matplotlib.use('GTkAgg')
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import theano.tensor as T
from theano import function, shared



plt.ion()

N = 1000
p = float32(0.1)
g = float32(1.5)
alpha = float32(1.0)
nsecs = 1440
dt = float32(0.1)
learn_every = 2

scale = 1.0/math.sqrt(p*N)
M = sprandn(N,N,p)*g*scale
M = mat(M.todense(), dtype=float32)

nRec2Out = N;
wo = mat(zeros((nRec2Out, 1), dtype=float32))
dw = mat(zeros((nRec2Out, 1), dtype=float32))
wf = float32(2.0*(mat(random.rand(N,1))-0.5))

simtime = float32(arange(0,nsecs-dt,dt))
simtime_len = len(simtime)
simtime2 = float32(arange(1*nsecs, 2*nsecs-dt, dt))

amp = 1.3
freq = 1/60.0
ft = (amp/1.0)*sin(1.0*math.pi*freq*simtime) + \
     (amp/2.0)*sin(2.0*math.pi*freq*simtime) + \
     (amp/6.0)*sin(3.0*math.pi*freq*simtime) + \
     (amp/3.0)*sin(4.0*math.pi*freq*simtime)
ft = ft / 1.5

ft2 = (amp/1.0)*sin(1.0*math.pi*freq*simtime2) + \
      (amp/2.0)*sin(2.0*math.pi*freq*simtime2) + \
      (amp/6.0)*sin(3.0*math.pi*freq*simtime2) + \
      (amp/3.0)*sin(4.0*math.pi*freq*simtime2)
ft2 = ft2 / 1.5

#plt.plot(ft2)

wo_len = float32(zeros(simtime_len))
zt = float32(zeros(simtime_len))
zpt = float32(zeros(simtime_len))
x0 = float32(0.5*mat(random.randn(N,1)))
z0 = float32(0.5*mat(random.randn(1,1)))

x = x0
r = tanh(x)
z = z0

P = float32((1.0/alpha)*eye(nRec2Out, dtype=float32))
s = array([a.A[0][0] for a in r], dtype=float32)
tr = shared(s)
trg = T.vector()
tp = shared(P)
tpg = T.matrix()
trPr = T.dot(tr, T.dot(tp, tr))
frPr = function([trg, tpg], trPr, givens=[(tr, trg), (tp, tpg)])
tdP = T.dot(T.dot(tp, tr), T.dot(tr, tp))
fdP = function([trg, tpg], tdP, givens=[(tr, trg), (tp, tpg)]) 

# Indexing starts from 0!
ti = -1


for t in simtime:
    ti += 1
    if ti % (nsecs/2) == 0:
        print "time:",str(t)
        
    # sim, so x(t) and r(t) are created.
    x = (1.0 - dt) * x + M * ( r * dt ) + wf * ( z * dt )
    r = tanh(x)
    z = wo.T * r
    
    if ti % learn_every == 0:
        # update inverse correlation matrix
        s = array([a.A[0][0] for a in r], dtype=float32)
        rPr = mat(frPr(s, P))
        rPr = rPr.A[0][0]
        c = float32(1.0/(1.0 + rPr))
        P = P - fdP(s, P) * c # k*(k.T*c)
        
        # update the error for the linear readout
        e = z - ft[ti]
        e = e.A[0][0]
        
        # update the output weights
        dw = - e * P * r # k * c or P * r ??
        wo = wo + dw
         
    zt[ti] = z
    wo_len[ti] = sqrt(wo.T * wo)
    
error_avg = sum(abs(zt-ft))/simtime_len;
print "Training MAE:",str(error_avg)
train_error_avg = error_avg

plt.show()
plt.figure()

print "Now testing... please wait."
ti = -1
for t in simtime:
    ti+=1
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt)
    r = tanh(x)
    z = wo.T * r
    
    zpt[ti] = z
    
error_avg = sum(abs(zpt-ft2))/simtime_len
print "Testing MAE:",str(error_avg)
test_error_avg = error_avg

plt.subplot(211)
plt.plot(simtime, ft)
plt.hold(True)
plt.plot(simtime, zt)
plt.title('training')
plt.xlabel('time')
plt.hold(True)
plt.ylabel('f and z')
plt.legend(('f', 'z'))


plt.subplot(212)
plt.hold(True)
plt.plot(simtime2, ft2) 
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
