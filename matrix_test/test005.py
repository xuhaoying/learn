def A_star(matrix, goal_matrix):
    open_set = PriorityQueue()  # 用优先队列存放开放列表中的状态节点
    closed_set = set()  # 存放已经被考虑的状态节点

    # 以元组的形式记录当前状态，h为启发式函数的值，g为当前状态到起点的实际距离，f为h和g之和
    start_node = (matrix, 0, manhattan_distance(matrix, goal_matrix), None)
    open_set.put(start_node)  # 将起点加入开放列表

    while not open_set.empty():
        current_node = open_set.get()  # 获取开放列表中f值最小的状态节点
        if current_node[0] == goal_matrix:  # 如果当前状态等于目标状态，返回路径
            return get_path(current_node)

        closed_set.add(current_node[0])  # 将当前状态添加到已考虑状态集合中

        # 遍历所有可能移动的状态
        for move in all_possible_moves(current_node[0]):
            new_matrix = apply_move(current_node[0], move)  # 应用移动操作，得到新状态矩阵
            if new_matrix in closed_set:  # 如果新状态已经被考虑过，跳过此次循环
                continue
            # 计算新状态的h、g和f值
            new_g = current_node[1] + 1  # 由于没有空格，每次移动距离为1
            new_h = manhattan_distance(new_matrix, goal_matrix)
            new_f = new_g + new_h
            new_node = (new_matrix, new_g, new_h, current_node)  # 以元组的形式记录新状态
            open_set.put(new_node)  # 将新状态加入开放列表

    return None  # 如果开放列表为空，返回None，表示无法到达目标状态

def all_possible_moves(matrix):
    moves = []
    n = matrix.shape[0]
    m = matrix.shape[1]

    for i from 0 to n-1:
        for j from 0 to m-2:
            # move right
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i, j + 1]
            new_matrix[i, j + 1] = matrix[i, j]
            moves.append((i, j, 'R', new_matrix))

        for j from 1 to m-1:
            # move left
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i, j - 1]
            new_matrix[i, j - 1] = matrix[i, j]
            moves.append((i, j, 'L', new_matrix))

    for j from 0 to m-1:
        for i from 0 to n-2:
            # move down
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i + 1, j]
            new_matrix[i + 1, j] = matrix[i, j]
            moves.append((i, j, 'D', new_matrix))

        for i from 1 to n-1:
            # move up
            new_matrix = matrix.copy()
            new_matrix[i, j] = matrix[i - 1, j]
            new_matrix[i - 1, j] = matrix[i, j]
            moves.append((i, j, 'U', new_matrix))

    return moves


# all_possible_moves() 函数的作用是返回给定矩阵所有可能的移动方式，以及移动后的新矩阵。
# 函数的输入参数是一个 numpy 矩阵 matrix，输出是一个列表 moves，其中每个元素都是一个四元组 (i, j, direction, new_matrix)。其中 (i, j) 表示空格的位置，direction 表示移动方向，new_matrix 表示移动后的新矩阵。

# 为了实现该函数，我们首先得到矩阵的行数和列数。然后，我们可以分别考虑将空格向左、向右、向上、向下移动的情况。
# 具体来说，我们可以使用双重循环遍历每个非边缘的元素，
# 对于每个元素，我们都可以将它与它右边（向左移动）、下面（向上移动）的元素交换位置，从而得到新矩阵。
# 将每种移动方式及其对应的新矩阵都添加到 moves 列表中。最后返回 moves 列表。

def apply_move(matrix, move):
    # 复制矩阵以避免在原始矩阵上进行更改
    new_matrix = copy.deepcopy(matrix)
    # 获取矩阵的形状
    shape = matrix.shape
    # 找到要移动的行或列的索引
    if move[-1] == "U":
        idx = int(move[:-1])
        # 将行向上移动，最后一行变为第一行
        new_matrix = np.roll(new_matrix, -1, axis=0)
        # 将第一行设置为0
        new_matrix[0, idx] = 0
    elif move[-1] == "D":
        idx = int(move[:-1])
        # 将行向下移动，第一行变为最后一行
        new_matrix = np.roll(new_matrix, 1, axis=0)
        # 将最后一行设置为0
        new_matrix[shape[0] - 1, idx] = 0
    elif move[-1] == "L":
        idx = int(move[:-1])
        # 将列向左移动，最后一列变为第一列
        new_matrix = np.roll(new_matrix, -1, axis=1)
        # 将第一列设置为0
        new_matrix[idx, 0] = 0
    elif move[-1] == "R":
        idx = int(move[:-1])
        # 将列向右移动，第一列变为最后一列
        new_matrix = np.roll(new_matrix, 1, axis=1)
        # 将最后一列设置为0
        new_matrix[idx, shape[1] - 1] = 0

    return new_matrix

# 函数接受两个参数：一个矩阵和一个移动字符串，返回一个新的矩阵状态。
# 在函数中，我们首先创建一个新的矩阵副本以避免在原始矩阵上进行更改。
# 然后，根据移动字符串中的指示，将矩阵向上、下、左或右滚动，
# 并将移动的行或列的第一个或最后一个元素设置为0。最后，返回新的矩阵状态。