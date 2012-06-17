# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:54:48 2012

@author: anand
"""

# FORCE external feedback loop

import math
from sprandn import sprandn
from numpy import mat, zeros, random, sin, arange, tanh, eye, sqrt, float32, asarray
#import matplotlib
# matplotlib.use('GTkAgg')
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
from theano import function, shared, Out
import theano.tensor as T

N = 1000
p = float32(0.1)
g = float32(1.5)
alpha = float32(1.0)
nsecs = 100 #1440
dt = float32(0.1)
learn_every = 2

scale = 1.0/math.sqrt(p*N)
M = sprandn(N,N,p)*g*scale
M = float32(M.todense())

nRec2Out = N;

simtime = float32(arange(0,nsecs-dt,dt))
simtime_len = len(simtime)
simtime2 = float32(arange(1*nsecs, 2*nsecs-dt, dt))

wo_len = float32(zeros(simtime_len))
zt = float32(zeros(simtime_len))
zpt = float32(zeros(simtime_len))
x0 = float32(0.5*random.randn(N, 1))
z0 = float32(0.5*random.randn(1, 1))

P0 = float32((1.0/alpha)*eye(nRec2Out))

x = shared(x0)
r = shared(tanh(x0))
z = shared(z0)
P = shared(P0)
dts = shared(dt)
wo = shared(zeros((nRec2Out, 1), dtype=float32))
wf = shared(float32(2.0*(random.rand(N, 1))-0.5))
fti = T.scalar('fti')
Ms = shared(M)

xnew = (1.0 - dts) * x + T.dot(Ms, r * dts) + wf * dts * z[0][0]
znew = T.dot(T.transpose(wo), r)
update = function([],[Out(z, borrow=True)], updates=[(x, xnew), (r, T.tanh(x)), (z, znew)])

print "Update compiled"

k = T.dot(P, r)
rPr = T.dot(T.transpose(r), k)
c = 1.0/(1.0 + rPr)
Pnew = P  - T.dot(k, k.T)  * c[0][0]
wonew = wo - (z[0][0] - fti) * k * c[0][0]
learn = function([fti],[Out(wo, borrow=True)],updates=[(P, Pnew), (wo, wonew)])

print "Learn compiled"

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

# Indexing starts from 0!
ti = -1

for t in simtime:
    ti += 1
    if ti % (nsecs/2) == 0:
        print "time:",str(t)
        
    zc, = update()
    zv = asarray(zc)
    
    if ti % learn_every == 0:
        woc, = learn(ft[ti])
        wov = mat(asarray(woc))
    zt[ti] = zv
    wo_len[ti] = sqrt(wov.T * wov)
    
error_avg = sum(abs(zt-ft))/simtime_len;
print "Training MAE:",str(error_avg)
train_error_avg = error_avg

plt.show()


asdfasd

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
