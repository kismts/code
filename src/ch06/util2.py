#from ch05.util import *
r = 4
n = 3
print_matr = False

def oracle(state, predicate):
    for item in range(len(state)):
        if predicate(item):
            state[item] *= -1

def round_matrix(matrix):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            item = matrix[row][col]
            matrix[row][col] = round(item.real,r)+1j*round(item.imag,r)

def round_vector(vector):
    for i in range(len(vector)):
        item = vector[i]
        vector[i] = round(item.real,r)+1j*round(item.imag,r)

def magnitude(vector):
    return [round(abs(item),r) for item in vector]

def probability(magn):
	return [round(pow(item, 2),r) for item in magn]

def print_state(text, state):
    round_vector(state)
    print("\n"+ text)
    print(state)
    print("\nmagnitude:")
    magn = magnitude(state)
    print(magn)
    print("\nprobability:")
    print(probability(magn))

def print_matrix(matrix):
    print("\nA matrix:")
    print(matrix)
    print("\ninverse of A matrix:")
    print(np.conj(matrix.transpose()))

def init_state(n):
    state = [0 for _ in range(2**n)]
    state[0] = 1
    return state

def random_transformation(n):
    import scipy.stats
    U = scipy.stats.unitary_group.rvs(2**n)
    round_matrix(U)
    if print_matr: print_matrix(U)

	def f_direct(state):
    	assert(len(state) == 2**n)
    	s = U @ state
    	for k in range(len(s)):
        	state[k] = s[k]

    def f_inverse(state):
        assert(len(state) == 2**n)
        s = np.conj(U.transpose()) @ state
        for k in range(len(s)):
            state[k] = s[k]

    return f_direct, f_inverse


f = random_transformation(n)
A = f[0]

state = init_state(n)
print("\ninit state:")
print(state)

A(state)
print_state("state after A matrix:", state)

predicate = lambda k: True if k == 3 else False
oracle(state, predicate)

print_state("\nstate after oracle:", state)