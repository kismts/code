import numpy as np

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
    print("\nA-1 matrix:")
    print(np.conj(matrix.transpose()))

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

def init_state(n):
    state = [0 for _ in range(2**n)]
    state[0] = 1
    return state

def inner(v1, v2):
    assert(len(v1) == len(v2))
    return sum(z1*z2.conjugate() for z1, z2 in zip(v1, v2))

def inversion(original, current):
    proj = inner(original, current)
    for k in range(len(current)):
        current[k] = 2*proj*original[k] - current[k]

from math import cos 
def classical_grover(state, predicate, iterations):
    s = state.copy()
    #items = [k for k in range(len(state)) if predicate(k)]
    #p = sum([abs(s[k])**2 for k in items])
    #theta = asin(sqrt(p))
    #assert is_close(inner(s, state), 1)
    for it in range(1, iterations + 1):
        oracle(state, predicate)
        inversion(s, state)
        #assert is_close(inner(s, state), cos(2 * it * theta))
        #p = sum([abs(state[k])**2 for k in items])
        #assert is_close(p, sin((2 * it + 1)*theta)**2) 

from math import log2
def inversion_0_transformation(f, state):
    n = int(log2(len(state)))

    transform = f[0]
    inverse_transform = f[1]

    inverse_transform(state)
    print_state("state after inv_A:", state)

    #assert is_close(state[0].imag, 0)
    for k in range(1, len(state)):
        state[k] = -state[k]
    print_state("state after M0:", state)

    transform(state)
    print_state("state after A matrix:", state)


f = random_transformation(n)
A = f[0]

state = init_state(n)
print("\ninit state:")
print(state)

A(state)
print_state("state after A matrix:", state)
copy = state.copy()

predicate = lambda k: True if k == 3 else False
oracle(state, predicate)
print_state("\nstate after oracle:", state)

classical_grover(copy, predicate,1)
print_state("\ncopy state after inversion", copy)

inversion_0_transformation(f, state)





