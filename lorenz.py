from scipy.integrate import odeint
import numpy as np
from abcsmc.abcsmc import ABCSMC
from gp.smoothGP import smooth_gp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
# Define the Lorenz chaotic attractor
#
def model_deriv(y, t, param):
    #Define parameters
    a,b,c=param
    #define states
    X,Y,Z=y
    #define derivatives
    dX_dt = a*(Y-X)
    dY_dt = X*(b-Z)-Y
    dZ_dt = X*Y -c*Z   
    return dX_dt,dY_dt,dZ_dt
    
def model_sol(param):
    y0 = [1.,1.,1.]
    time = np.linspace(0, 100, 1000)
    solution = odeint(model_deriv, y0, time, args=(param,))
    return solution
#
# Define the r.h.s of dx/dt = f(x,theta)
#
def simulation_rhs_f(trial):
    stateLen = 200
    sample = 10
   
    xbar_X, xbar_Y, xbar_Z = xbar
    rhsf_X = np.ones((stateLen,sample))
    rhsf_Y = np.ones((stateLen,sample))
    rhsf_Z = np.ones((stateLen,sample))
    for t in range(stateLen):
        for j in range(sample):
            rhsf_X[t,j] = (xbar_Y[t,j] - xbar_X[t,j])*trial[0]
            rhsf_Y[t,j] = xbar_X[t,j]*(trial[1] - xbar_Z[t,j]) - xbar_Y[t,j]
            rhsf_Z[t,j] = xbar_X[t,j]*xbar_Y[t,j] - trial[2]*xbar_Z[t,j]
    
    if np.any(np.array([trial])<0.0):
        rhsf_X[0,:]=-1000.
        rhsf_Y[0,:]=-1000.
        rhsf_Z[0,:]=-1000.
               
    return [rhsf_X.mean(axis=1),rhsf_Y.mean(axis=1),rhsf_Z.mean(axis=1)]
#
# Define a simple distance metric : Todo Use correlation integral
#
def dist_metric(d,x):
    dist=0.0
    for states in xrange(len(x)):
        if np.all(np.array(x[states][0]==-1000.0)):
            dist += np.inf
        else:
            dist += np.sum((d[states]-x[states])**2)    
            
    return dist
#
# fine solver for ppc
#
def model_sol_pos(param,init):
    y0 = init
    time = np.linspace(0, 100, 100000)
    solution = odeint(model_deriv, y0, time, args=(param,))
    return solution
#
# plot the attractors at mean +- 2 sigma
#
def plot_pos(soln_fn, step, sampler, init):
    center = []
    upper = []
    lower = []
    for par in xrange(sampler.nparam):
        center.append(np.mean(sampler.theta[step][:,par]))
        upper.append(np.mean(sampler.theta[step][:,par])+1.96*np.std(sampler.theta[step][:,par]))
        lower.append(np.mean(sampler.theta[step][:,par])-1.96*np.std(sampler.theta[step][:,par]))
    mean_soln = soln_fn(center,init)
    lower_soln = soln_fn(lower,init)
    upper_soln = soln_fn(upper,init)
    
    return mean_soln, lower_soln, upper_soln

if __name__ == '__main__':
#
# generate fake data with little noise
#
    time = np.linspace(0, 100, 1000)
    true_params = [10.,28.,8./3.]
    sol=model_sol(true_params)
    Y=np.ones((len(time),3))
    for states in range(3):
        Y[:,states] = sol[:,states] + np.random.randn(len(time),)*(0.025*np.std(sol[:,states]))
#
# GP smoothing
#
    input_time=time[-200:].reshape(200,1)
    xbar, dxbar, gps = smooth_gp(Y, input_time, sol)
#
# setup and run abcsmc with GP distance 
#
    priors =  [('uniform', [5.0,50.]), ('uniform', [0.0,100.]), ('uniform', [0.0,50.0])]
    model_sim = simulation_rhs_f
    sampler = ABCSMC(3,100,dxbar,10,priors,model_sim,dist_metric,[2.5e6,100],quantile=75)
    samples = sampler.sample()
#
# plot GP
#
    start = 80
    stop = 100
    window = [start,stop]
    for s in xrange(3):
        gps[s].plot_noiseless(window)
        plt.plot(input_time[-200:],sol[-200:,s],
                        color='#ff7f0e',lw=1.5,label='state '+str(s+1))
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
    plt.show()    
        
#
# plot attractor reconstructions
#
    init=[xbar[0][0,:].mean(axis=0),xbar[1][0,:].mean(axis=0),xbar[2][0,:].mean(axis=0)]

    msol, lsol, usol = plot_pos(model_sol_pos, 9, sampler, init)
    true_init = [sol[0,0], sol[0,1], sol[0,2]]
    true_soln = model_sol_pos(true_params,true_init)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(25, 17.5))
    ax = fig.add_subplot(221, projection='3d')
    plt.plot(true_soln[:,0],true_soln[:,1],true_soln[:,2], label ='True Solution')
    plt.legend()
    ax = fig.add_subplot(222, projection='3d')
    plt.plot(msol[:,0],msol[:,1],msol[:,2], label ='Center Solution')
    plt.legend()
    ax = fig.add_subplot(223, projection='3d')
    plt.plot(lsol[:,0],lsol[:,1],lsol[:,2], label ='Lower bnd Solution')
    plt.legend()
    ax = fig.add_subplot(224, projection='3d')
    plt.plot(usol[:,0],usol[:,1],usol[:,2], label ='Upper bnd Solution')
    plt.legend()
    plt.show()



