import numpy as np


def Gauss(A, mod=0):
    """Compute the Gaussian elimination of the matrix A modulo mod

    Parameters
    ----------
    A : numpy array
        The matrix that will be row-reduced
    mod : int, optional
        integer representing the modulus

    Returns
    -------
    numpy array
        the row-reduced matrix

    """

    if mod:
        A = np.array(A, int)
    else:
        A = np.array(A, float)

    m, n = A.shape

    dim_min = min(m, n)
    row = 0
    for k in range(dim_min):
        if mod:
            column = [[el, np.abs(A[el, k]) % mod]
                      for el in range(row, m)
                      ]
        else:
            column = [[el, np.abs(A[el, k])]
                      for el in range(row, m)
                      ]

        column.sort(key=lambda x: x[1], reverse=True)
        i_max = column[0][0]

        pivot2 = [i_max, k]

        zero_column = k
        while A[pivot2[0], pivot2[1]] == 0:
            zero_column += 1
            pivot2 = [i_max, zero_column]
            if pivot2[1] == n:
                if mod:
                    return np.fmod(A, mod)
                else:
                    return A

            if mod:
                column = [[el, np.abs(A[el, pivot2[1]]) % mod]
                          for el in range(pivot2[0], m)
                          ]
            else:
                column = [[el, np.abs(A[el, pivot2[1]])]
                          for el in range(pivot2[0], m)
                          ]

            column.sort(key=lambda x: x[1], reverse=True)

            i_max = column[0][0]

            pivot2 = [i_max, zero_column]

        A[[row, pivot2[0]], :] = A[[pivot2[0], row], :]

        final_pivot = [row, pivot2[1]]

        for i in range(final_pivot[0] + 1, m):
            f = A[i, final_pivot[1]] / A[final_pivot[0], final_pivot[1]]
            den_f = A[final_pivot[0], final_pivot[1]]
            num_f = A[i, final_pivot[1]]
            for j in range(0, n):
                if mod:
                    A[i, j] = (np.fmod(den_f * A[i, j]
                               - num_f * A[final_pivot[0], j], mod))
                else:
                    A[i, j] = A[i, j] - A[final_pivot[0], j] * f
        row += 1

    if mod:
        return np.fmod(A, mod)
    else:
        return A


def Rank(M):
    """Compute the rank of the matrix M that needs
       to be already in echelon row form.
       See: https://en.wikipedia.org/wiki/Row_echelon_form
    """
    rank = 0
    for row in M:
        if not np.allclose(row, np.zeros(len(row))):
            rank += 1

    return rank


if __name__ == "__main__":
    # np.set_printoptions(suppress=True,precision=3)
    # 9 qubit code X stabilizers: see https://arxiv.org/abs/1404.3747

    L = 9
    Sx = [[0, 1, 3, 4], [1, 2], [4, 5, 7, 8], [6, 7]]
    binary_generators = []

    for s in Sx:
        a = np.zeros(L)
        a[s] = 1
        print(a)
        binary_generators.append(a.tolist())

    binary_generators = np.array(binary_generators, dtype=int)

    print(binary_generators)

    row_reduced_Sx = Gauss(binary_generators, mod=2)

    print(row_reduced_Sx)

    print(Rank(row_reduced_Sx))
