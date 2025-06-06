import numpy as np

#from ch05.util import *

def oracle(state, predicate):
    for item in range(len(state)):
        if predicate(item):
            state[item] *= -1


def random_transformation(n):
    import scipy.stats
    U = scipy.stats.unitary_group.rvs(2**n)
    print("\nA matrix:")
    print(U)
    UT = np.conj(U.transpose())
    print("\nA-1 matrix:")
    print(UT)

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

from math import log2
def inversion_0_transformation(f, state):
    n = int(log2(len(state)))

    transform = f[0]
    inverse_transform = f[1]

    inverse_transform(state)
    print("\nstate after inv_A:")
    print(state)

    #assert is_close(state[0].imag, 0)
    for k in range(1, len(state)):
        state[k] = -state[k]
    print("\nstate after M0:")
    print(state)

    transform(state)
    print("\nstate after A matrix (second):")
    print(state)


n = 2
f = random_transformation(n)
A = f[0]

state = init_state(n)
print("\ninit state:")
print(state)

A(state)
print("\nstate after A matrix (first):")
print(state)

predicate = lambda k: True if k == 3 else False
oracle(state, predicate)
print("\nstate after oracle:")
print(state)

inversion_0_transformation(f, state)





