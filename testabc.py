import numpy as np
from abcsmc.abcsmc import ABCSMC
import scipy
### Simple test problem with a bivariate Gaussian density

means= np.array([0.037579, 0.573537])
cov = np.array([[.01,.005],[.005,.01]])
data = np.random.multivariate_normal(means, cov, size=1000)

def simulation(param):
    cov =np.array([[.01,.005],[.005,.01]])
    return np.random.multivariate_normal(param, cov, size=1000)

def dist_metric(d,x):
    return np.sum(np.abs(np.mean(x,axis=0) - np.mean(d,axis=0)))

model_sim = simulation
priors =  [('normal', [0.03,.5]), ('uniform', [0.1, 0.9])]
sampler = ABCSMC(2,100,data,15,priors,model_sim,dist_metric,[0.5,0.002])
chains = sampler.sample()
