import numpy as np
import scipy
from sklearn.covariance import ledoit_wolf
import sys
from .priors import *
from .kernels import *

class ABCSMC(object):

	def __init__(self, nparam, npart, data, niter, priors, simulator, dfunc, schedule, quantile=None):

		self.data = data

		if npart < nparam + 1:
			raise ValueError('Not enough particles')
		else:
			self.npart = npart

		self.nparam = nparam
		
		if not(quantile == None) and niter < 2:
			raise ValueError('Requires more iterations')

		elif not(quantile == None):
			quantile = float(quantile)
			self.adapt = True
			self.quantile = quantile
			self.niter = niter
			self.epsilon = np.tile(schedule[0],niter)
			if len(schedule) > 1:
				self.tmin = schedule[-1]#give some other name
			else:
				self.tmin = 0.0

		if (quantile == None) and len(schedule) < 2:
			raise ValueError('Atleast two tolerances required')

		elif (quantile == None) and len(schedule) > 2:
			self.epsilon = np.asarray(schedule)
			self.adapt = False
			self.niter = len(schedule)

		elif (quantile == None):
			if schedule[1] >= schedule[0]:
				raise ValueError('Schedule endpoint must be smaller than start')
			
			if niter < 2:
				raise ValueError('Requires more iterations')

			self.epsilon = np.linspace(schedule[0], schedule[1], num=niter)
			self.adapt = False
			self.niter = niter

		if not(self.adapt):
				self.tmin = 0.0
		self.simulator = simulator
		self.dfunc = dfunc


		self.theta=np.zeros([self.niter,self.npart,self.nparam])
		self.wt=np.zeros([self.niter,self.npart])
		self.delta=np.zeros([self.niter,self.npart])

		self.pert_kernel = FilippiOCM(nparam,npart)

		self.verbose = True

		if not(len(priors)==nparam):
			raise ValueError('Priors must be specified for all parameters')
		else:
			self.priors = Priors(priors)


		self.end_sampling = False


	def dist(self, x):
		#if np.any(x.shape != self.data.shape):
		#	raise ValueError('Simulated and observed data is of different shape')

		return self.dfunc(self.data,x)


	def next_epsilon(self,t):

		new_epsilon = np.percentile(self.delta[t], self.quantile)

		if new_epsilon < self.tmin:
			new_epsilon = self.tmin

		return new_epsilon


	def calculate_weight(self, t, Pid, covariance):

		kernelPdf = scipy.stats.multivariate_normal(
					mean=self.theta[t][Pid],cov=covariance).pdf(self.theta[t-1])

		if  np.any(self.wt[t-1]) ==0 or np.any(kernelPdf)==0:
			print ("Kernel or weights error", kernelPdf, self.wt[t-1])
			sys.exit(1)

		priorproduct = self.priors.priorproduct(self.theta[t][Pid])

		return priorproduct/(np.sum(self.wt[t-1]*kernelPdf))


	def calculate_covariance(self, t):

		covariance = self.pert_kernel.covariance(t, self.theta[t-1], self.delta[t-1],
														 self.epsilon[t], self.wt[t-1])

		if np.linalg.det(covariance) <1.E-15:
			covariance  =  ledoit_wolf(self.theta[t-1])[0]

		return covariance


	def sample(self):

		t = 0

		while self.end_sampling == False:

			if  t == self.niter or self.epsilon[t] == self.tmin:
				self.end_sampling = True
				return self.theta[t-1]

			if t==0:
				for p in range(self.npart):
					self.theta[t][p], self.delta[t][p] = self.stepper(t, p)

				self.wt[t] =1./self.npart
				if self.verbose:
						print ("\t Stage:",t,"\t tol:",self.epsilon[t],"\t Params:",[np.mean(self.theta[t][:,i]) for i in range(self.nparam)])#change this later

				if self.adapt:
					self.epsilon[t+1] = self.next_epsilon(t)

				t += 1

			else:
				covariance = self.calculate_covariance(t)

				for p in range(self.npart):
					self.theta[t][p], self.delta[t][p] = self.stepper(t, p, covariance)
					self.wt[t][p] = self.calculate_weight(t, p, covariance)

				self.wt[t] = self.wt[t]/np.sum(self.wt[t])

				if self.verbose:
					print( "\t Step:",t,"\t epsilon_t:",self.epsilon[t],"\t Params:",[np.mean(self.theta[t][:,i]) for i in range(self.nparam)])#change this later

				if self.adapt and t <self.niter-1:
						self.epsilon[t+1] = self.next_epsilon(t)

				t += 1

	def stepper(self, t, Pid, covariance=None):

		while True:

			if t ==0: 
				
				theta_star = self.priors.sample()
				x = self.simulator(theta_star)
				rho = self.dist(x)

			else:
	            
				ispart = int(np.random.choice(self.npart,size=1,p=self.wt[t-1]))
				theta_old = self.theta[t-1][ispart]
				
				theta_star = np.atleast_1d(scipy.stats.multivariate_normal.rvs(mean= theta_old,cov=covariance,size=1))
				x = self.simulator(theta_star)
				rho = self.dist(x)

			if rho <= self.epsilon[t]:
				break

		return theta_star, rho
