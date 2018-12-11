
# Common definitions
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd
import matplotlib.pyplot as plt
npr.seed(42)

"""
Parameters:
theta0: starting values for the simulation
delta: target acceptance probability
Likelihood: function of theta
nSamples: number of samples including the burn-in period
nSamplesAdapt: number of samples used to tune the sampler
testRun: print debugging info, default False
"""

def NoUTurn(theta0, delta, Likelihood, nSamples, nSamplesAdapt, testRun=False):
	"""
	Find first epsilon using the heuristic suggested in the paper. (Algorithm 4)
	"""
	#epsilon0=FindReasonableEpsilon(theta0, Likelihood, testRun)
	epsilon0=0.05
	if testRun: print('First epsilon: ', epsilon0)
	"""
	mu: use the recommended value for mu. Mu is used in tuning epsilon as are 
	parameters epsilon0bar, h0bar and gamma
	"""
	mu=np.log(10*epsilon0)
	epsilon0bar=1.0
	h0bar=0.0
	gamma=0.05
	t0=10.0
	kappa=0.75
	# initialize variables that store parameters
	thetaHist=np.empty([nSamples+1, len(theta0)])
	thetaHist[0]=theta0
	rHist=np.empty([nSamples, len(theta0)])
	# Parameters for tuning epsilon
	epsilonHist=np.empty(nSamples+1)
	epsilonHist[0]=epsilon0
	logEpsilonBarHist=np.empty(nSamplesAdapt)
	logEpsilonBarHist[0]=np.log(epsilon0bar)
	
	hBarHist=np.empty(nSamplesAdapt)
	hBarHist[0]=h0bar
			  
	gradLikelihood=autograd.grad(Likelihood)
	nAccepted=0	
	for i in range(1,nSamples+1):
		# Simulate momentum variable
		r0 = npr.normal(size=len(theta0))
		# Calculate slice variable u, that renders the conditional distribution p(theta, r | u)
		highBound=np.exp(Likelihood(thetaHist[i-1])-1/2*r0@r0)
		if testRun: 
			print('high bound: ', highBound)
			print('Likelihood: ', Likelihood(thetaHist[i-1]), ' , 1/2*r0@r0: ', 1/2*r0@r0)
		u=npr.uniform(low=0, high=highBound)
		# initialize values; in the beginning left (minus) and right (plus) sides are the same.
		thetaMinus=thetaHist[i-1]
		thetaPlus=thetaHist[i-1]
		rMinus=r0
		rPlus=r0
		j=0.0
		thetaHist[i]=thetaHist[i-1]
		# number of theta candidates
		n=1.0
		# s indicates whether the sample continues to trace new values or has made an U-Turn (then s=0)
		s=1.0
		if testRun: print('***Starting iteration, round: ', i)
		while s==1:
			# Decide the direction, -1 or 1
			vj=-1+2*npr.binomial(n=1, p=0.5)
			if testRun: print('Going to direction: ', vj)
			if vj == -1:
				thetaMinus,rMinus,_,_,thetaCur,nCur,sCur,alpha,nAlpha=BuildTree(thetaMinus,rMinus,u,vj,j,epsilonHist[i-1],thetaHist[i-1],r0,Likelihood,testRun)
			elif vj==1:
				_,_,thetaPlus,rPlus,thetaCur,nCur,sCur,alpha,nAlpha=BuildTree(thetaPlus,rPlus,u,vj,j,epsilonHist[i-1],thetaHist[i-1],r0,Likelihood, testRun)
			# Accept the move from old state to new state (thetaCur)
			if sCur==1.0:
				if npr.uniform() < nCur/n:
					if testRun: 
						print('Accepted new theta value, round: ', i)
						print('Accepted likelihood: ', Likelihood(thetaCur))
					thetaHist[i]=thetaCur
			n=n+nCur
			"""
			Loop until the sampler makes an U-Turn:
			s=0, if the distance between the leftmost (thetaMinus) and rightmost (thetaPlus) nodes reduces
			"""
			s=sCur*((thetaPlus-thetaMinus)@rMinus>=0)*((thetaPlus-thetaMinus)@rPlus>=0)
			j+=1.0
		# Tune epsilon during burn-in period
		if i < nSamplesAdapt:
			hBarHist[i]=(1-1/(i+t0))*hBarHist[i-1]+1/(i+t0)*(delta-alpha/nAlpha)
			logEpsilon=mu-i**(1/2)/gamma*hBarHist[i]
			logEpsilonBarHist[i]=i**(-kappa)*logEpsilon+(1-i**(-kappa))*logEpsilonBarHist[i-1]
			epsilonHist[i]=np.exp(logEpsilon)
		# For the actual sample always use same epsilon.
		else:
			epsilonHist[i]=epsilonHist[i-1]
	return thetaHist[nSamplesAdapt+1:], epsilonHist
"""
Parameters:
theta: parameters of interest
r: momentum variable
u: slice variable that restricts the sample generation
v: [-1,1] defines whether to explore the distribution backwards or forwards
j: height of the binary tree; number of nodes in the tree is 2**j
epsilon: step size
theta0: latest accepted theta value (from the round before the current one)
r0: momentum variable simulated in the 'main program' NoUTurn
Likelihood: likelihoodfunction of theta
deltaMax: If the error in the simulation becomes extremely large, then the process is stopped. The paper suggests
		that a large number - e.g 1000 - should be selected so that it doesn't interfere with the simulation.
testRun: True/False, print debugging info?
"""
def BuildTree(theta,r,u,v,j,epsilon,theta0,r0,Likelihood, testRun=False, deltaMax=1000):
	gradLikelihood=autograd.grad(Likelihood)
	if j==0:
		thetaCur,rCur=Leapfrog(theta,r,v*epsilon,gradLikelihood)
		if testRun: 
			print('BuildTree: j==0, Likelihood(thetaCur)', Likelihood(thetaCur))
			print('compared to Likelihood(theta0): ', Likelihood(theta0))
		nCur=1.0*(np.log(u)<(Likelihood(thetaCur)-1/2*rCur@rCur))
		# Stop simulating if the difference of Likelihood(theta)-0.5r@r - log u < -deltaMax
		sCur=1.0*(np.log(u)<(deltaMax+Likelihood(thetaCur)-1/2*rCur@rCur))
		return thetaCur,rCur,thetaCur,rCur,thetaCur,nCur,sCur,min(1.0,np.exp(Likelihood(thetaCur)-1/2*rCur@rCur-Likelihood(theta0)+1/2*r0@r0)),1.0
	else:
		thetaMinus,rMinus,thetaPlus,rPlus,thetaCur,nCur,sCur,alphaCur,nAlphaCur=BuildTree(theta,r,u,v,j-1,epsilon,theta0,r0,Likelihood)
		if sCur==1:
			if v==-1:
				thetaMinus,rMinus,_,_,thetaCur2,nCur2,sCur2,alphaCur2,nAlphaCur2=BuildTree(thetaMinus,rMinus,u,v,j-1,epsilon,theta0,r0,Likelihood)
			else:
				_,_,thetaPlus,rPlus,thetaCur2,nCur2,sCur2,alphaCur2,nAlphaCur2=BuildTree(thetaPlus,rPlus,u,v,j-1,epsilon,theta0,r0,Likelihood)
			if npr.uniform()<nCur2/(nCur+nCur2):
				if testRun: print('Accepted thetaCur2, Likelihood: ', Likelihood(thetaCur2))
				thetaCur=thetaCur2
			alphaCur=alphaCur+alphaCur2;nAlphaCur=nAlphaCur+nAlphaCur2
			sCur=sCur2*((thetaPlus-thetaMinus)@rMinus>=0.0)*((thetaPlus-thetaMinus)@rPlus>=0.0)
			nCur=nCur+nCur2
		if testRun: print('BuildTree: j!=1, Likelihood(thetaCur)', Likelihood(thetaCur))
		return thetaMinus,rMinus,thetaPlus,rPlus,thetaCur,nCur,sCur,alphaCur,nAlphaCur

   
# def Leapfrog(parTheta, parR, parEpsilon, thetaGradient), return parTheta0, parR0
def FindReasonableEpsilon(parTheta, fLikelihood, testRun=False):
	if testRun: print('Started FindReasonableEpsilon')
	# Initialize values
	epsilon=1.0
	r=npr.normal(size=len(parTheta))
	gradLikelihood=autograd.grad(fLikelihood)
	# Generate first theta and momentum r
	thetaCur, rCur=Leapfrog(parTheta, r, epsilon, gradLikelihood)
	# a gets values [-1, 1], and epsilon is halved or doubled accordingly
	condition=(fLikelihood(thetaCur)-1/2*rCur@rCur-(fLikelihood(parTheta)-1/2*r@r))
	a=2.0*((condition)>np.log(0.5))-1

	while condition**a > -a*np.log(2.0):
		thetaOld=thetaCur; rOld=rCur
		epsilon=(2**a)*epsilon
		thetaCur, rCur=Leapfrog(thetaOld, rOld, epsilon, gradLikelihood)
		condition=(fLikelihood(thetaCur)-1/2*rCur@rCur-(fLikelihood(thetaOld)-1/2*rOld@rOld))
		a=2.0*(condition>0.5)-1
	if testRun: print('Ending FindReasonableEpsilon')
	return epsilon

# Leadfrog function is the same for HMC and NUTS
def Leapfrog(parTheta, parR, parEpsilon, thetaGradient):
	parR0 = parR +(parEpsilon/2)*thetaGradient(parTheta)
	parTheta0 = parTheta+parEpsilon*parR0
	parR0 = parR0+(parEpsilon/2)*thetaGradient(parTheta0)
	return parTheta0, parR0