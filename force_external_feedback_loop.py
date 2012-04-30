# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:30:37 2012

@author: anand
"""

# FORCE external feedback loop

import math
from sprandn import sprandn
from numpy import mat, zeros, random, sin, arange, tanh, eye, sqrt
import matplotlib
matplotlib.use('GTkAgg')
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
plt.ion()

N = 1000
p = 0.1
g = 1.5
alpha = 1.0
nsecs = 1440
dt = 0.1
learn_every = 2

# Same as 1/tau in the paper
scale = 1.0 / math.sqrt( p * N )
M = sprandn(N,N,p) * g * scale
M = mat(M.todense())

nRec2Out = N;
wo = mat(zeros((nRec2Out, 1)))
dw = mat(zeros((nRec2Out, 1)))
wf = 2.0*(mat(random.rand(N,1))-0.5)

simtime = arange(0,nsecs-dt,dt)
simtime_len = len(simtime)
simtime2 = arange(1*nsecs, 2*nsecs-dt, dt)

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

wo_len = zeros(simtime_len)
zt = zeros(simtime_len)
zpt = zeros(simtime_len)
x0 = 0.5*mat(random.randn(N,1))
z0 = 0.5*mat(random.randn(1,1))

x = x0
r = tanh(x)
z = z0

# Indexing starts from 0!
ti = -1
P = ( 1.0 / alpha ) * mat(eye(nRec2Out))
print P.shape
print r.shape

fig = plt.figure()
plt.subplot(2,1,1)
p1, = plt.plot(simtime, ft)
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
        
dynplot = False

for t in simtime:
    ti += 1
    if ti % (nsecs/2) == 0:
        print "time:",str(t)
        if dynplot == True:
            p1.set_ydata(ft)
            p2.set_ydata(zt)
            p3.set_ydata(wo_len)
            plt.draw()
        
    # sim, so x(t) and r(t) are created.
    x = (1.0 - dt) * x + M * ( r * dt ) + wf * ( z * dt )
    r = tanh(x)
    z = wo.T * r
    
    if ti % learn_every == 0:
        # update inverse correlation matrix
        # k = P*r
        # rPr = r.T*k
        c = 1.0/(1.0 + r.T * P * r)
        c = c.A[0][0]
        P = P - (P * r * r.T * P * c) # k*(k.T*c)
        
        # update the error for the linear readout
        e = z - ft[ti]
        e = e.A[0][0]
        	
        # update the output weights
        dw = - e * P * r # -e * k * c # k * c or P * r ??
        wo = wo + dw
         
    zt[ti] = z
    wo_len[ti] = sqrt(wo.T*wo)
    
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
    z = wo.T*r
    
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
f.write("Training: "+str(train_error_avg))
f.write("Training: "+str(test_error_avg))
f.close()
