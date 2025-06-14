import numpy as np

def random_unitary_matrix(n):
    """
    Generates a random real unitary (orthogonal) matrix of size n x n.
    """
    # Generate a random matrix with real entries
    A = np.random.rand(n, n)

    # Perform Gram-Schmidt orthonormalization
    Q, R = np.linalg.qr(A)
    
    return Q
    
    # Example usage:
    n = 3
    unitary_matrix = random_unitary_matrix(n)
    print(unitary_matrix)
    
    # Verification:  Check if the matrix is close to orthogonal
    print(np.allclose(unitary_matrix @ unitary_matrix.conj().T, np.eye(n)))  # Should be close to True