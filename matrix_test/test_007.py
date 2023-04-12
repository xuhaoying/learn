import numpy as np
from typing import List, Tuple

# 定义状态类
class State:
    def __init__(self, matrix: np.ndarray, moves: List[str]):
        self.matrix = matrix  # 矩阵
        self.moves = moves    # 移动方式
        self.cost = 0         # 代价

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def __lt__(self, other):
        return self.cost < other.cost

# 判断状态是否已经在列表中出现过
def is_visited(state: State, visited: List[State]) -> bool:
    for v in visited:
        if state == v:
            return True
    return False

# 生成状态的所有邻居
def generate_neighbors(state: State) -> List[Tuple[np.ndarray, str]]:
    neighbors = []
    matrix = state.matrix
    n = matrix.shape[0]

    # 寻找空格的位置
    empty_pos = np.argwhere(matrix == n * n)

    # 左移
    if empty_pos[1] > 0:
        new_matrix = matrix.copy()
        new_matrix[empty_pos[0], empty_pos[1]] = new_matrix[empty_pos[0], empty_pos[1]-1]
        new_matrix[empty_pos[0], empty_pos[1]-1] = n*n
        neighbors.append((new_matrix, 'L'))

    # 右移
    if empty_pos[1] < n - 1:
        new_matrix = matrix.copy()
        new_matrix[empty_pos[0], empty_pos[1]] = new_matrix[empty_pos[0], empty_pos[1]+1]
        new_matrix[empty_pos[0], empty_pos[1]+1] = n*n
        neighbors.append((new_matrix, 'R'))

    # 上移
    if empty_pos[0] > 0:
        new_matrix = matrix.copy()
        new_matrix[empty_pos[0], empty_pos[1]] = new_matrix[empty_pos[0]-1, empty_pos[1]]
        new_matrix[empty_pos[0]-1, empty_pos[1]] = n*n
        neighbors.append((new_matrix, 'U'))

    # 下移
    if empty_pos[0] < n - 1:
        new_matrix = matrix.copy()
        new_matrix[empty_pos[0], empty_pos[1]] = new_matrix[empty_pos[0]+1, empty_pos[1]]
        new_matrix[empty_pos[0]+1, empty_pos[1]] = n*n
        neighbors.append((new_matrix, 'D'))

    return neighbors

"""
在A*算法中，我们需要维护一个open list和一个closed list。Open list存储所有待扩展的节点，Closed list存储所有已经扩展过的节点。这两个list可以用一个优先队列（priority queue）来实现。

其中，每个节点的状态需要包含以下信息：

当前矩阵的状态
到达当前状态的移动步骤序列
估计的代价f(n)：f(n) = g(n) + h(n)，其中g(n)表示从初始状态到当前状态的实际移动步数，h(n)表示当前状态到目标状态的预估移动步数（曼哈顿距离）。
所以我们可以用一个三元组表示节点的状态：

python
Copy code
(node_state, move_seq, f_value)
其中，node_state表示当前状态的矩阵，move_seq表示从初始状态到当前状态的移动步骤序列，f_value表示当前状态的估价函数值。

open list和closed list可以用一个字典来实现。键为节点状态的哈希值，值为三元组。

python
Copy code
open_list = {}
closed_list = {}
在A*算法中，我们需要不断地从open list中选出f(n)最小的节点进行扩展，并将其加入到closed list中。如果某个节点的状态已经存在于closed list中，说明我们已经处理过该状态，不需要再次处理。

因此，在向open list中添加新节点时，我们需要先判断该节点的状态是否已经存在于closed list中，如果是，则直接跳过。否则，将该节点加入到open list中。
"""