import numpy as np
import heapq


# 定义状态类
class State:
    def __init__(self, matrix, cost, moves):
        self.matrix = matrix
        self.cost = cost
        self.moves = moves

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)


# 定义启发式函数：曼哈顿距离
def manhattan_distance(state, target):
    distance = 0
    for i in range(state.matrix.shape[0]):
        for j in range(state.matrix.shape[1]):
            if state.matrix[i][j] != target[i][j]:
                target_row, target_col = np.where(target == state.matrix[i][j])
                distance += abs(i - target_row) + abs(j - target_col)
    return distance


# 定义移动函数
def move(state, row, col, move_row, move_col):
    new_matrix = np.copy(state.matrix)
    temp = new_matrix[row + move_row][col + move_col]
    new_matrix[row + move_row][col + move_col] = 0
    new_matrix[row][col] = temp
    new_cost = state.cost + 1
    new_moves = list(state.moves)
    new_moves.append(str(new_matrix[row][col]) + direction_map[(move_row, move_col)])
    return State(new_matrix, new_cost, new_moves)


# 定义A*算法函数
def solve_puzzle(start, target):
    start_state = State(start, 0, [])
    target_state = State(target, 0, [])
    heap = []
    visited = set()
    heapq.heappush(heap, start_state)
    while heap:
        current_state = heapq.heappop(heap)
        if current_state == target_state:
            return current_state.moves
        if tuple(map(tuple, current_state.matrix)) in visited:
            continue
        visited.add(tuple(map(tuple, current_state.matrix)))
        zero_row, zero_col = np.where(current_state.matrix == 0)
        for move_row, move_col in directions:
            new_row, new_col = zero_row + move_row, zero_col + move_col
            if 0 <= new_row < current_state.matrix.shape[0] and 0 <= new_col < current_state.matrix.shape[1]:
                new_state = move(current_state, zero_row, zero_col, move_row, move_col)
                new_state.cost += manhattan_distance(new_state, target_state)
                heapq.heappush(heap, new_state)


# 定义解决方案输出函数
def print_solution(solution):
    print('Total moves: %d' % len(solution))
    for i, move in enumerate(solution):
        print('Move %d: %s' % (i + 1, move))


# 示例
# 定义起始状态和目标状态
start = np.array([[1, 4, 3], [7, 5, 6], [2, 8, 9]])
target = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 定义方向映射字典
# direction_map = {(-1, 0): 'U', (1, 0): 'D', (0, -1): 'L', (0,
