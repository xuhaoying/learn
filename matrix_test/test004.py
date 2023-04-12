import numpy as np
from heapq import heappush, heappop

# 计算矩阵中每个数字在最终排序后的位置
def get_goal_positions(n):
    goal_positions = {}
    for i in range(n):
        for j in range(n):
            goal_positions[i * n + j + 1] = (i, j)
    return goal_positions

# 计算当前矩阵的状态和代价
def get_state_and_cost(state, goal_positions):
    n = int(np.sqrt(len(state)))
    cost = 0
    for i in range(n):
        for j in range(n):
            if state[i][j] == 0:
                continue
            goal_i, goal_j = goal_positions[state[i][j]]
            cost += abs(i - goal_i) + abs(j - goal_j)
    return state, cost

# 检查状态是否是合法的
def is_valid_state(state):
    n = int(np.sqrt(len(state)))
    flat_state = state.flatten()
    return len(flat_state) == len(set(flat_state)) and set(flat_state) == set(range(n ** 2))

# 获取下一个状态和移动方式
def get_successors(state):
    n = int(np.sqrt(len(state)))
    zero_i, zero_j = np.argwhere(state == 0)[0]
    successors = []
    if zero_i > 0:
        new_state = np.copy(state)
        new_state[zero_i][zero_j], new_state[zero_i - 1][zero_j] = new_state[zero_i - 1][zero_j], new_state[zero_i][zero_j]
        successors.append((new_state, f'{zero_i}{zero_j}U'))
    if zero_i < n - 1:
        new_state = np.copy(state)
        new_state[zero_i][zero_j], new_state[zero_i + 1][zero_j] = new_state[zero_i + 1][zero_j], new_state[zero_i][zero_j]
        successors.append((new_state, f'{zero_i}{zero_j}D'))
    if zero_j > 0:
        new_state = np.copy(state)
        new_state[zero_i][zero_j], new_state[zero_i][zero_j - 1] = new_state[zero_i][zero_j - 1], new_state[zero_i][zero_j]
        successors.append((new_state, f'{zero_i}{zero_j}L'))
    if zero_j < n - 1:
        new_state = np.copy(state)
        new_state[zero_i][zero_j], new_state[zero_i][zero_j + 1] = new_state[zero_i][zero_j + 1], new_state[zero_i][zero_j]
        successors.append((new_state, f'{zero_i}{zero_j}R'))
    return successors

# A*搜索算法
def solve_puzzle(start_state):
    n = int(np.sqrt(len(start_state)))
    goal_positions = get_goal_positions(n)
    start_state, start_cost = get_state_and_cost(start_state, goal_positions)
    visited = set()
    queue = [(start_cost, start_state, '')]
    while queue:
        _, state, moves = heappop
        pass
