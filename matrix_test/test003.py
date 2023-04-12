import numpy as np

def find_index(matrix, num):
    """找到num在矩阵中的位置"""
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == num:
                return (i, j)

def move_left(matrix, num, steps):
    """将num所在行向左移动，步数为steps"""
    i, j = find_index(matrix, num)
    for k in range(steps):
        matrix[i][j-k-1], matrix[i][j-k] = matrix[i][j-k], matrix[i][j-k-1]
    return matrix

def move_right(matrix, num, steps):
    """将num所在行向右移动，步数为steps"""
    i, j = find_index(matrix, num)
    for k in range(steps):
        matrix[i][j+k+1], matrix[i][j+k] = matrix[i][j+k], matrix[i][j+k+1]
    return matrix

def move_up(matrix, num, steps):
    """将num所在列向上移动，步数为steps"""
    i, j = find_index(matrix, num)
    for k in range(steps):
        matrix[i-k-1][j], matrix[i-k][j] = matrix[i-k][j], matrix[i-k-1][j]
    return matrix

def move_down(matrix, num, steps):
    """将num所在列向下移动，步数为steps"""
    i, j = find_index(matrix, num)
    for k in range(steps):
        matrix[i+k+1][j], matrix[i+k][j] = matrix[i+k][j], matrix[i+k+1][j]
    return matrix

def sort_matrix(matrix):
    """将矩阵排序为1到n*n"""
    n = len(matrix)
    res = []
    for num in range(1, n*n+1):
        i, j = find_index(matrix, num)
        target_i, target_j = (num-1) // n, (num-1) % n
        if i != target_i:
            if i < target_i:
                matrix = move_down(matrix, num, target_i - i)
                res += [f"{target_i-i}D"] * (target_i - i)
            else:
                matrix = move_up(matrix, num, i - target_i)
                res += [f"{i-target_i}U"] * (i - target_i)
        if j != target_j:
            if j < target_j:
                matrix = move_right(matrix, num, target_j - j)
                res += [f"{target_j-j}R"] * (target_j - j)
            else:
                matrix = move_left(matrix, num, j - target_j)
                res += [f"{j-target_j}L"] * (j - target_j)
    return res, matrix

def generate_random_matrix(n):
    # 生成从1到n*n的随机排列
    seq = np.random.permutation(n * n) + 1
    # 转换成N阶矩阵
    matrix = seq.reshape(n, n)
    return matrix

if __name__ == '__main__':
    print(sort_matrix(generate_random_matrix(5)))


