from numpydoc import setup
import numpy as np


def LUDecomp(mat):
    """
    Perform LU decomposition of a matrix using only elementary matrices.
    """
    n = mat.shape[0]
    EL = []
    L = np.eye(n)
    U = mat
    # Construct all type 3 matricies
    for col in range(0, n):
        for row in range(col + 1, n):
            E = cmultadd(n, row, col, (-U[row, col] / U[col, col]))
            E1 = cmultadd(n, row, col,  U[row, col] / U[col, col])
            U = np.dot(E, U)
            EL.append(E1)

    # Construct all type 1 matrcies.
    for j in range(0, n):
        E = cmult(n, j, 1 / U[j, j])
        E1 = cmult(n, j, U[j, j])
        U = np.dot(E, U)
        EL.append(E1)

    for i in EL:
        L = np.dot(L, i)

    return [L, U]
