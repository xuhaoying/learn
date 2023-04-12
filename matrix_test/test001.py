import numpy as np

# 计算状态的曼哈顿距离
def manhattan_distance(state, n):
    distance = 0
    for i in range(n):
        for j in range(n):
            value = state[i][j]
            if value != 0:
                target_i = (value - 1) // n
                target_j = (value - 1) % n
                distance += abs(i - target_i) + abs(j - target_j)
    return distance

# 计算状态的哈密顿距离
def hamming_distance(state, n):
    distance = 0
    for i in range(n):
        for j in range(n):
            value = state[i][j]
            if value != 0 and value != i * n + j + 1:
                distance += 1
    return distance

# 查找当前状态中最大的值的位置
def find_zero(state):
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i][j] == state.shape[0]**2:
                return (i, j)

# 移动0到新的位置，返回移动后的状态
def move(state, zero_pos, direction):
    i, j = zero_pos
    new_i, new_j = i, j
    if direction == 'U':
        new_i -= 1
    elif direction == 'D':
        new_i += 1
    elif direction == 'L':
        new_j -= 1
    elif direction == 'R':
        new_j += 1
    new_state = state.copy()
    new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]
    return new_state

# 计算当前状态的f值
def f(state, n, h_func):
    h = h_func(state, n)
    g = state.g
    return g + h

# A*算法求解最短路径
def solve_puzzle(start_state, n, h_func):
    open_list = [start_state]
    closed_list = set()
    steps = []  # 保存移动步骤
    while open_list:
        # 选取f值最小的状态
        current_state = min(open_list, key=lambda x: f(x, n, h_func))
        open_list.remove(current_state)
        closed_list.add(current_state)
        # 判断是否达到目标状态
        if np.array_equal(current_state, np.arange(1, n*n+1).reshape(n, n)):
            # 回溯路径
            while current_state.parent is not None:
                steps.insert(0, current_state.move)
                current_state = current_state.parent
            return steps
        # 扩展状态
        zero_pos = find_zero(current_state)
        for direction in ['U', 'D', 'L', 'R']:
            if ((direction == 'U' and zero_pos[0] > 0) or
                (direction == 'D' and zero_pos[0] < n-1) or
                (direction == 'L' and zero_pos[1] > 0) or
                (direction == 'R' and zero_pos[1] < n-1)):
                new_state = move(current_state, zero_pos, direction)
                new_state.g = current_state.g + 1
                new_state.h = h_func
