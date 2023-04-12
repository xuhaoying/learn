import numpy as np
from queue import PriorityQueue


def generate_random_matrix(n):
    # 生成从1到n*n的乱序N阶矩阵
    return (np.random.permutation(n * n) + 1).reshape(n, n)


class MatrixSort:
    # 定义移动方向
    DIRECTIONS = {'U': "move_up",
                  'D': "move_down",
                  'L': "move_left",
                  'R': "move_right"}

    def __init__(self, n):
        self.matrix = generate_random_matrix(n)
        self.expectation_matrix = self.get_expectation_matrix()

    def get_expectation_matrix(self):
        n = self.matrix.shape[0]
        return np.arange(1, n * n + 1).reshape((n, n))

    def move_left(self, row_index):
        self.matrix[row_index] = np.roll(self.matrix[row_index], -1,)

    def move_right(self, row_index):
        self.matrix[row_index] = np.roll(self.matrix[row_index], 1)

    def move_up(self, col_index):
        self.matrix.T[col_index] = np.roll(self.matrix.T[col_index], -1)

    def move_down(self, col_index):
        self.matrix.T[col_index] = np.roll(self.matrix.T[col_index], 1)

    def move(self, index, direction):
        getattr(self, MatrixSort.DIRECTIONS.get(direction.upper()))(index)

    def manhattan_distance(self):
        # 将两个矩阵相减，得到每个元素的差值矩阵
        diff_matrix = np.subtract(self.matrix, self.expectation_matrix)
        # 取差值矩阵中的每个元素的绝对值
        abs_diff_matrix = np.absolute(diff_matrix)
        # 以行为单位求和
        manhattan_distance = np.sum(abs_diff_matrix, axis=1)
        return manhattan_distance








#
# def astar(start, goal):
#     """
#     使用A*算法搜索从start到goal的最短路径
#     start: 开始状态
#     goal: 目标状态
#     返回值: 从start到goal的最短路径（移动方式）
#     """
#     queue = PriorityQueue()
#     queue.put((0, start, []))
#     visited = {tuple(start.reshape(-1)): 0}
#
#     while not queue.empty():
#         _, state, path = queue.get()
#
#         if np.array_equal(state, goal):
#             return path
#
#         for direction in DIRECTIONS:
#             new_state = move(state, direction)
#             new_path = path + [direction]
#             # 计算启发值（曼哈顿距离）
#             h = manhattan_distance(new_state, goal)
#             g = len(new_path)
#             f = g + h
#             if tuple(new_state.reshape(-1)) not in visited or visited[tuple(new_state.reshape(-1))] > f:
#                 visited[tuple(new_state.reshape(-1))] = f
#                 queue.put((f, new_state, new_path))
#
#     return None








