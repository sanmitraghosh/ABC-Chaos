import numpy as np
from abcsmc.abcsmc import ABCSMC
import scipy


means= np.array([0.037579, 0.573537])
cov = np.array([[.01,.005],[.005,.01]])
data = np.random.multivariate_normal(means, cov, size=1000)

def simulation(param):
    cov =np.array([[.01,.005],[.005,.01]])
    #Ideally do something with the pool here
    return np.random.multivariate_normal(param, cov, size=1000)

def dist_metric(d,x):
    return np.sum(np.abs(np.mean(x,axis=0) - np.mean(d,axis=0)))
"""
class test_ABCSMC:
    def __init__(self, data, simulation, dist_metric):

        self.data = data
        self.simulation = simulation
        self.dist_metric = dist_metric
        self.priors_norm_unif = [('normal', [0.03,.5]), ('uniform', [0.1, 0.9])]
        self.priors_norm_gam = priors =  [('normal', [0.03,.5]), ('gamma', [0.1, 0.9])]
        self.seed = 21
        self.particles = 100
        self.nparam = 2

        self.schedule = None
        self.niter = None

    def test_priors(self):

    def test_kernel(self):

    def test_results(self):

"""


model_sim = simulation



priors =  [('normal', [0.03,.5]), ('uniform', [0.1, 0.9])]
sampler = ABCSMC(2,100,data,15,priors,model_sim,dist_metric,[0.5,0.002])
chains = sampler.sample()
print(sampler.priors.dimension)