def all_possible_moves(matrix):
    moves = []
    n = matrix.shape[0]

    for i in range(0, n-1):
        for j in range(0, n-2):
            # move right
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i, j + 1]
            new_matrix[i, j + 1] = matrix[i, j]
            moves.append((i, j, 'R', new_matrix))

        for j in range(1, n-1):
            # move left
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i, j - 1]
            new_matrix[i, j - 1] = matrix[i, j]
            moves.append((i, j, 'L', new_matrix))

    for j in range(0, n-1):
        for i in range(0, n-2):
            # move down
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i + 1, j]
            new_matrix[i + 1, j] = matrix[i, j]
            moves.append((i, j, 'D', new_matrix))

        for i in range(1, n-1):
            # move up
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i - 1, j]
            new_matrix[i - 1, j] = matrix[i, j]
            moves.append((i, j, 'U', new_matrix))

    return moves



import numpy as np
n = 6
matrix = (np.random.permutation(n * n) + 1).reshape(n, n)
moves1 = all_possible_moves(matrix)
print(len(moves1))
