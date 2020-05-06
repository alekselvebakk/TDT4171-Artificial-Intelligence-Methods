import numpy as np

def decideSensorModel(evidence):
    sensorModelTrue = np.array([[0.9,    0],
                                [0,      0.2]])
    sensorModelFalse = np.array([[0.1,   0],
                                 [0,     0.8]])
    if evidence == 1:
        return sensorModelTrue
    else:
        return sensorModelFalse


def forward(evidence, priori):
    #Defining transition and sensor model
    transModel = np.array([[0.7, 0.3],
                           [0.3, 0.7]])
    sensorModel = decideSensorModel(evidence)

    #calculating posterior
    posteriori = sensorModel@transModel@priori

    #Normalizing
    norm_inv = 1/np.sum(posteriori)
    posterior = norm_inv*posteriori

    #returning
    return posterior

def backward(evidence, sendback):
    #Defining transition and sensor model
    transModel = np.array([[0.7, 0.3],
                           [0.3, 0.7]])
    sensorModel = decideSensorModel(evidence)

    #calculating further sendback
    b = transModel@sensorModel@sendback


    #normalizing
    norm_inv = 1/np.sum(b)
    b = norm_inv*b

    #returning
    return b

def print_forward(evidence, pri):
    print('\nPrinting only forward: \n')


    #handling input
    evidence = np.array(evidence)
    days = len(evidence)
    p_0 = np.array(pri)

    #going forward
    p = p_0
    for i in range(days):
        print('\nDay: ', i, '\nProbabilities: \n', p)
        p = forward(evidence[i], p)
    print('\nDay: ', days, '\nProbabilities: \n', p, '\n\n')
    return p



def print_backward(evidence_input, sendback):
    print('\nPrinting only backward: \n')

    #handling input


def normalize(p, b):
    product = np.multiply(p, b)
    norm = 1/np.sum(product)
    return norm*product


def forwardBackward(day, evidence, initial_state):
	# evidence is vector of ones and zeros representing True and False respectively
	#initial_state given as a 2X1 vector

	fv = np.zeros((day+1,2))
	b = np.ones((day+1,2))
	sv = np.zeros((day+1,2))
	fv[0,:] = initial_state

	for i in range(1,day+1):
		fv[i,:] = forward(evidence[i-1],fv[i-1,:])
	for i in range(day, 0,-1):
		#calculations
		sv[i,:] = normalize(fv[i],b[i])
		b[i-1,:] = backward(evidence[i-1], b[i,:])

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




print_forward([1, 1, 0, 1, 1], [0.5, 0.5])
forwardBackward(5,np.array([1,1,0,1,1]),np.array([0.5, 0.5]))
