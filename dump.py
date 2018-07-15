		self.simulator = simulator


        self.theta=np.zeros([self.niter,self.npart,self.nparam])
        self.wt=np.zeros([self.niter,self.npart])
        self.delta=np.zeros([self.niter,self.npart])



        self.pert_kernel = OptimalLocalCovarianceMatrix(nparam,npart)

        self.verbose = True

        if not(len(prios)==len(nparams):

            raise ValueError('Priors must be specified for all parameters')

    	self.priors = Priors(priors)


		self.end_sampling = False


	def dist(self,x):

			return self.dfunc(self.data,x)


	def next_epsilon(self,t):

		new_epsilon = np.percentile(self.delta[t], self.quantile)

		if new_epsilon < self.tmin:

			new_epsilon = self.tmin

		return new_epsilon


	def calculate_weight(self, t, Pid, covariance):

        kernelPdf = scipy.stats.multivariate_normal(mean=self.theta[t][Pid],cov=covariance,allow_singular=True).pdf(self.theta[t-1])

		if  np.any(self.wt[t-1]) ==0 or np.any(kernels)==0:

            print "Error computing Kernel or weights...", kernels, self.wt[t-1]
            sys.exit(1)

        priorproduct = self.prior.priorproduct(self.theta[t][Pid])

		return priorproduct/(np.sum(self.wt[t-1]*kernelPdf))


    def calculate_covariance(self, t, theta):

        return self.pert_kernel.get_covariance(theta[t-1], wt[t-1], theta)#pop_prev, w_prev, theta


    def sample(self):

        t = 0

        while self.end_sampling == False:

            if  t+1 == self.niter or self.epsilon[t] == self.tmin:

                self.end_sampling = True

				return theta[t]mean(axis=0)

    		if t==0:

                for p in range(self.npart):

                    self.theta[t][p], self.delta[t][p], _ = self.stepper(t, p)

                self.wgt[t] =1./self.npart
                t += 1

    		else:

                for p in range(self.npart):

                    self.theta[t][p], self.delta[t][p], covariance = self.stepper(t, p)
    				self.wt[t][p] = self.calculate_weight(t, p, covariance)

                self.wt[t] = self.wt[t]/np.sum(self.wt[t])

                if self.verbose:

                    print "\t Step:",t,"\t tol:",self.epsilon[t],"\t Params:",[np.mean(self.theta[t][:,i]) for i in range(self.nparam)]#change this later

                if self.adapt_t and t <self.niter-1:

                    self.epsilon[t+1]=self.next_epsilon(t)

                t += 1


    def stepper(self, t, Pid):


        while True:

                if t ==0: #draw from prior


                        theta_star = self.prior.sample()
                        x = self.simulator(theta_star)
                        rho = self.dist(x)

                else:

                        #np.random.seed()
                        ispart = int(np.random.choice(self.npart,size=1,p=self.wgt[t-1]))
                        theta_old = self.theta[t-1][ispart]
                        covariance = self.calculate_covariance(t, theta_old)
                        theta_star = np.atleast_1d(scipy.stats.multivariate_normal.rvs(mean= theta_old,cov=covariance,size=1))
                        x = self.model(theta_star)
                        rho = self.dist(x)

                if rho <= self.epsilon[t]:

                        break

        return theta_star, rho, covariance
