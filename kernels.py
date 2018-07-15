import numpy as np

class FilippiOCM(object):
	'''Filippi et al 2012, eq 12 & 13'''
	def __init__(self,nparam,npart):

		self.nparam = nparam
		self.npart = npart

	def covariance(self, t, prev_pop, prev_delta, current_epsilon, prev_wt):
		
		ind = np.where(prev_delta <= current_epsilon)[0]
		n0 = len(ind)
		theta_bar = prev_pop[ind]
		wt_bar = prev_wt[ind]
		covariance = np.diag(np.zeros(self.nparam))
		if n0 ==0:
			return 2.*np.cov(prev_pop.T)
		else:
			wt_bar = wt_bar/np.sum(wt_bar) # normalise

			for i in range(self.nparam):
				for j in range(self.nparam):
					covariance[i,j] = np.sum([np.sum(prev_wt[p]*wt_bar*(theta_bar[:,i]-prev_pop[p,i])*(theta_bar[:,j]-prev_pop[p,j]) ) for p in range(self.npart)])
		
			
		return covariance
