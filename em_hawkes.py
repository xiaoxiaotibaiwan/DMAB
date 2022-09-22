import sys
sys.path.append("../TPL/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from init_net import init
import math
def lambd(p,events,T): # 强度函数
    mu,alpha,theta = p
    y_0 = mu
    y_1 = 0
    for i in range(len(events)):
        y_1 += alpha*math.exp(-theta*(T - events[i]))
    return y_0 + y_1
def cdf(p,events,T):
    mu,alpha,theta = p
    y0 = mu*(T-events[0])
    if y0 < 0:
        y0 = 0
    y1 = 0
    for i in range(len(events)):
        if T >= events[i]:
            y1 += (1-math.exp(-theta*(T - events[i])))
    return y0+(y1)*(alpha/theta)

def log_likelihood(p,events,T): # 对数极大似然函数
    mu,alpha,theta = p
    y_0 = mu*T
    y_1 = 0
    y_2 = 0
    for i in range(len(events)):
        y_1 += (math.exp(-theta*(T-events[i])) -1)
        y_2 += math.log(lambd(p,events[:i],events[i]))
    y_1 = (alpha/theta)*y_1
    return y_2 - (y_1 + y_0)

def em_fit(events,T,maxiter=800,reltol=1e-7):
    t0 = events[0]
    N = len(events) 
    mu = N*0.8*(1+(np.random.random()-0.5)/10)/T
    alpha = 0.2 + ((np.random.random()-0.5)/10)
    theta = mu*(1+(np.random.random()-0.5)/10)
    odll_p = log_likelihood([mu,alpha,theta],events,T)
    for _ in range(maxiter): #不需要使用循环变量的情况
        # print(mu,alpha,theta)
        # E-setp
        E1 = 1/mu
        E2 = 0
        E3 = 0
        C1 = 0
        C2 = 0
        phi = 0
        ga = 0
        C1 += 1-math.exp(-theta*(T-t0))
        C2 += (T -t0) * math.exp(-theta*(T -t0))
        for i in range(1,N):
            d = events[i] - events[i-1]
            r = T - events[i]
            ed = math.exp(-theta*d)
            er = math.exp(-theta*r)
            ga = ed*(d*(1+phi)+ga)
            phi = ed*(1+phi)
            Z = mu + alpha*theta*phi
            atz = alpha*theta/Z
            E1 += 1/Z
            E2 += atz*phi
            E3 += atz*ga
            C1 += 1-er
            C2 += r*er
        mu = mu*E1/T
        theta = E2/(alpha*C2+E3)
        alpha = E2/C1

        odll = log_likelihood([mu,alpha,theta],events,T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < reltol:
            break
        odll_p = odll
    return mu,alpha,theta