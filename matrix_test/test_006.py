import numpy as np

def generate_random_matrix(n):
    """
    生成一个值为从1到n*n的乱序N阶矩阵
    """
    arr = np.arange(1, n*n+1)
    np.random.shuffle(arr)
    return arr.reshape(n, n)

def manhattan_distance(start, end):
    """
    计算两个点之间的曼哈顿距离
    """
    return abs(start[0]-end[0]) + abs(start[1]-end[1])

def get_shortest_path(matrix, start, end):
    """
    获取矩阵中两个节点之间的最短路径
    """
    n = matrix.shape[0]
    # 用字典记录每个节点的曼哈顿距离
    distances = {(i, j): manhattan_distance((i, j), end) for i in range(n) for j in range(n)}
    # 用字典记录每个节点的前驱节点
    predecessors = {(i, j): None for i in range(n) for j in range(n)}
    # 记录已经访问过的节点
    visited = set()
    # 用字典记录起点到每个节点的最短距离
    cost_so_far = {start: 0}
    # 用优先队列存储待访问的节点
    queue = [(0, start)]
    while queue:
        _, current = queue.pop(0)
        if current == end:
            # 找到了终点，返回路径和总代价
            path = []
            while current != start:
                path.append(predecessors[current])
                current = predecessors[current]
            path.reverse()
            return path, cost_so_far[end]
        visited.add(current)
        for neighbor in [(current[0]-1, current[1]), (current[0]+1, current[1]),
                         (current[0], current[1]-1), (current[0], current[1]+1)]:
            if neighbor[0] < 0 or neighbor[0] >= n or neighbor[1] < 0 or neighbor[1] >= n:
                # 超出矩阵范围，跳过
                continue
            new_cost = cost_so_far[current] + 1
            if neighbor not in visited and (neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]):
                # 更新前驱节点和代价
                cost_so_far[neighbor] = new_cost
                predecessors[neighbor] = current
                # 加入优先队列
                priority = new_cost + distances[neighbor]
                queue.append((priority, neighbor))
        queue.sort()
    # 没有找到路径
    return None, None

def move_matrix_by_path(matrix, path):
    """
    根据移动路径移动矩阵
    """
    for step in path:
        if step.startswith("row"):
            row, direction = int(step[3:]), step[2]
            if direction == "L":
                matrix[row] = np.roll(matrix[row], -1)
            elif direction ==
