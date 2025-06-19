r = 2
n = 2
col = 2
print_matr = True
print_magn = False
print_prob = False
compl_matr = False

def oracle(state, predicate):
    for item in range(len(state)):
        if predicate(item):
            state[item] *= -1

def round_matrix(matrix):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            item = matrix[row][col]
            if compl_matr:
                matrix[row][col] = round(item.real,r)+1j*round(item.imag,r)
            else:
                matrix[row][col] = round(item, r)

def round_vector(vector):
    for i in range(len(vector)):
        item = vector[i]
        if compl_matr:
            vector[i] = round(item.real,r)+1j*round(item.imag,r)
        else:
            vector[i] = round(item, r)

def magnitude(vector):
    return [round(abs(item),r) for item in vector]

def probability(magn):
    return [round(pow(item, 2),r) for item in magn]

def print_state(text, state):
    round_vector(state)
    print("\n"+ text)
    print(state)
    magn = magnitude(state)
    if print_magn:
        print("\nmagnitude:")
        print(magn)
    if print_prob:
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

def real_matrix(dim):
    A = np.random.rand(dim, dim)
    Q,R = np.linalg.qr(A)
    return Q

def random_transformation(n):
    import scipy.stats
    U = []
    if compl_matr: 
        U = scipy.stats.unitary_group.rvs(2**n)
    else:
        U = real_matrix(2**n)
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
    
    def f_col(state, column):
        assert(len(state) == 2**n)
        m = np.conj(U.transpose())
        col = [arr[column] for arr in m]
        s = [c * s for c,s in zip(col, state)]
        for k in range(len(s)):
            state[k] = s[k]
            

    return f_direct, f_inverse, f_col


f = random_transformation(n)
A = f[0]
A_inv = f[1]
col_inv = f[2]

state = init_state(n)
print("\ninit state:")
print(state)

A(state)
print_state("A_state:", state)

a_state = state.copy()
orig_state = a_state.copy()
A_inv(orig_state)
print_state("\nA_inv X a_state:", orig_state)

col_inv(a_state, col)
print_state("\nA_inv_col X a_state:", a_state)

predicate = lambda k: True if k == col else False
oracle(state, predicate)

print_state("\nstate after oracle:", state)
oracle_state = state.copy()

col_inv(oracle_state, col)
print_state("\nA_inv_col X oracle_state:", oracle_state)

A_inv(state)
print_state("\nA_inv X oracle_state", state)









