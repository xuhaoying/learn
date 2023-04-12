import numpy as np
import heapq


# 定义节点类
class Node:
    def __init__(self, state, g_cost, h_cost, move):
        self.state = state
        self.g_cost = g_cost  # 从初始状态到当前状态的代价
        self.h_cost = h_cost  # 从当前状态到目标状态的估计代价
        self.f_cost = g_cost + h_cost  # f(n) = g(n) + h(n)
        self.move = move  # 从父节点到当前节点的移动方式

    # 定义比较方法，以f(n)为关键字，用于堆排序
    def __lt__(self, other):
        return self.f_cost < other.f_cost


# 定义A*算法类
class AStar:
    def __init__(self, start, end):
        self.start = start  # 起始状态
        self.end = end  # 目标状态
        self.n = start.shape[0]  # 矩阵大小
        self.moves = []  # 移动方式

    # 计算启发函数h(n)，这里使用曼哈顿距离
    def heuristic(self, state):
        h = 0
        for i in range(self.n):
            for j in range(self.n):
                value = state[i, j]
                if value != self.end[i, j]:
                    end_i, end_j = np.where(self.end == value)
                    h += abs(end_i - i) + abs(end_j - j)
        return h

    # 获取当前状态的下一个状态
    def get_successors(self, state):
        successors = []
        zero_i, zero_j = np.where(state == 0)
        # 向上移动
        if zero_i > 0:
            new_state = state.copy()
            new_state[zero_i, zero_j] = new_state[zero_i - 1, zero_j]
            new_state[zero_i - 1, zero_j] = 0
            successors.append(Node(new_state, 1, self.heuristic(new_state), str(state[zero_i, zero_j]) + 'U'))
        # 向下移动
        if zero_i < self.n - 1:
            new_state = state.copy()
            new_state[zero_i, zero_j] = new_state[zero_i + 1, zero_j]
            new_state[zero_i + 1, zero_j] = 0
            successors.append(Node(new_state, 1, self.heuristic(new_state), str(state[zero_i, zero_j]) + 'D'))
        # 向左移动
        if zero_j > 0:
            new_state = state.copy()
            new_state[zero_i, zero_j] = new_state[zero_i, zero_j - 1]
            new_state[zero_i, zero_j - 1] = 0
            successors.append(Node(new_state, 1, self.heuristic(new_state), str(state[zero_i, zero_j]) + 'L'))
        # 向右移动
        if zero_j < self.n - 1:
            new_state = state.copy()
            new_state[zero_i, zero_j] = new_state[zero_i, zero_j + 1]
            new_state[zero_i, zero_j + 1] = 0
            successors.append(Node(new_state))
