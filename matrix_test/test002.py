import numpy as np

def solve_puzzle(board):
    n = board.shape[0]
    goal_board = np.arange(1, n * n + 1).reshape(n, n)  # 目标矩阵
    start_board = board.copy()  # 复制原矩阵，避免修改原矩阵
    moves = []  # 移动列表

    # 定义曼哈顿距离启发式函数
    def manhattan_distance(board):
        distance = 0
        for i in range(n):
            for j in range(n):
                if board[i][j] != 0:
                    # 当前元素的位置
                    row, col = divmod(board[i][j] - 1, n)
                    # 目标元素的位置
                    distance += abs(row - i) + abs(col - j)
        return distance

    # 定义所有可能的移动函数
    def possible_moves(board):
        moves = []
        for i in range(n):
            for j in range(n):
                if board[i][j] != 0:
                    if i > 0:  # 可以向上移动
                        move = (i, j, i - 1, j)
                        moves.append(move)
                    if i < n - 1:  # 可以向下移动
                        move = (i, j, i + 1, j)
                        moves.append(move)
                    if j > 0:  # 可以向左移动
                        move = (i, j, i, j - 1)
                        moves.append(move)
                    if j < n - 1:  # 可以向右移动
                        move = (i, j, i, j + 1)
                        moves.append(move)
        return moves

    # 定义搜索函数
    def search(board, depth, max_depth, last_move):
        # 如果当前状态已经是目标状态，返回True
        if (board == goal_board).all():
            return True

        # 如果已经到达搜索深度的限制，返回False
        if depth == max_depth:
            return False

        # 计算当前状态的曼哈顿距离
        score = manhattan_distance(board)

        # 对所有可能的移动进行搜索
        for move in possible_moves(board):
            # 避免来回移动同一行或同一列，加快搜索速度
            if move[0] == last_move[2] and move[1] == last_move[3]:
                continue

            # 将空格移动到目标位置，生成新状态
            new_board = board.copy()
            new_board[move[2]][move[3]] = board[move[0]][move[1]]
            new_board[move[0]][move[1]] = 0

            # 如果新状态的曼哈顿距离小于当前状态，进行搜索
            new_score = manhattan_distance(new_board)
            if new_score < score:
                # 添加移动步骤到移动列表
                moves.append((new_board[move[2]][move[3]], last_move[2] - move[2], last_move[3] - move[3]))
                # if search(new_board, depth

def find_best_node(frontier, f_score):
    """
    找到启发式函数值最小的节点
    """
    min_score = float('inf')
    min_node = None
    for node in frontier:
        if f_score[node] < min_score:
            min_score = f_score[node]
            min_node = node
    return min_node


def astar(start, goal):
    """
    A*搜索算法，返回从start到goal的最短路径
    """
    # 记录已经探索过的节点
    closed_set = set()
    # 记录需要探索的节点
    open_set = set([start])
    # 记录每个节点的父节点
    came_from = {}

    # 记录从起点到每个节点的实际代价
    g_score = {start: 0}
    # 记录从起点到每个节点的启发式代价
    f_score = {start: manhattan_distance(start, goal)}

    while open_set:
        # 找到启发式函数值最小的节点
        current = find_best_node(open_set, f_score)

        # 如果已经到达目标节点，则返回路径
        if current == goal:
            return reconstruct_path(came_from, current)

        # 将当前节点标记为已探索过
        open_set.remove(current)
        closed_set.add(current)

        # 遍历所有与当前节点相邻的节点
        for neighbor in get_neighbors(current, goal):
            # 如果相邻节点已经被探索过，则忽略
            if neighbor in closed_set:
                continue

            # 计算从起点到相邻节点的代价
            tentative_g_score = g_score[current] + 1

            # 如果相邻节点不在需要探索的节点中，则将其加入
            if neighbor not in open_set:
                open_set.add(neighbor)

            # 如果已经找到更优的路径，则忽略
            elif tentative_g_score >= g_score[neighbor]:
                continue

            # 保存从起点到相邻节点的路径
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal)

    # 如果无法到达目标节点，则返回None
    return None

# closed_set表示已经探索过的节点集合，open_set表示待探索的节点集合，came_from表示每个节点的父节点，g_score表示从起点到每个节点的实际代价，f_score表示从起点到每个节点的启发式代价。函数中的循环用来遍历所有需要探索的节点，直到找到目标节点或者无法找到可行路径。