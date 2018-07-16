import numpy as np
import matplotlib.pyplot as plt
import GPy

def smooth_gp(traces, input_time, true_sol=None, optim_restarts = 1):
    time_len = len(input_time)
    gp_models = []
    state = []
    velocity = []
    for states in xrange(traces.shape[1]):
        kern = GPy.kern.Matern32(1) + GPy.kern.Bias(1) 
        gp_models.append(GPy.models.GPRegression(input_time, 
                                    traces[-time_len:,states].reshape(len(input_time),1), kern))
    start = 400
    stop = 500
    window = [start,stop]
    for states in xrange(traces.shape[1]):
        gp_models[states].optimize_restarts(optimizer='lbfgs',messages=False,
                                           num_restarts = optim_restarts)
        state.append(gp_models[states].posterior_samples_f(input_time,size=10))
        velocity.append(gp_models[states].predictive_gradients(input_time)
                       [0].reshape(len(input_time),))
        gp_models[states].plot_noiseless(window)
        if not(true_sol) == None:
            plt.plot(input_time[-500:],true_sol[-500:,states],
                     color='#ff7f0e',lw=1.5,label='state '+str(states+1))
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
    return state, velocity, gp_models