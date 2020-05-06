import numpy as np

def forward(prior, evidence):
	sprobs = np.array([[0.7, 0.3],[0.3, 0.7]])
	eprobs = np.array([[0.9,0.1],[0.2, 0.8]])

	transition = sprobs[0:2, 0] * prior[0] + sprobs[0:2, 1] * prior[1]
	if evidence==1:
		posterior = np.multiply(eprobs[0:2,0],transition)
	else:
		posterior = np.multiply(eprobs[0:2, 1], transition)
	norm = 1/np.sum(posterior)
	posterior = norm*posterior
	return posterior

def backward(backMsg, evidence):
	sprobs = np.array([[0.7, 0.3], [0.3, 0.7]])
	if evidence:
		eprobs = np.array([[0.9, 0],[0,0.2]])
	else:
		eprobs = np.array([[0.1, 0], [0, 0.8]])

	b = np.matmul(sprobs,eprobs)
	b = np.matmul(b,backMsg)
	norm = 1/np.sum(b)
	return norm*b

def normalize(forwardMsg,backMsg):
	normalized = np.multiply(forwardMsg,backMsg)
	norm = 1/(np.sum(normalized))
	return norm*normalized

def BackwardForwardFilter(day, evidence, initial_state):
	# evidence is vector of ones and zeros representing True and False respectively
	#initial_state given as a 2X1 vector

	fv = np.zeros((day+1,2))
	b = np.ones((day+1,2))
	sv = np.zeros((day+1,2))
	fv[0,:] = initial_state

	for i in range(1,day+1):
		fv[i,:] = forward(fv[i-1,:],evidence[i-1])
	for i in range(day, 0,-1):
		#calculations
		sv[i,:] = normalize(fv[i],b[i])
		b[i-1,:] = backward(b[i,:],evidence[i-1])

		#print in nice(?) format
		print('Posterior at time step',i, '                      ',fv[i])
		print('backwardsMessage at time step ',i, '              ',b[i])
		print('smoothed probabilities at time step',i, '         ',sv[i])
		print(' ')

	sv[0, :] = normalize(fv[0], b[0])
	print('Posterior at time step', 0, '                      ', fv[0])
	print('backwardsMessage at time step ', 0, '              ', b[0])
	print('smoothed probabilities at time step', 0, '         ', sv[0])
	print(' ')

BackwardForwardFilter(5,np.array([1,1,0,1,1]),np.array([0.5, 0.5]))
