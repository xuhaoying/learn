定义输入参数：n表示矩阵大小，matrix表示输入的n*n矩阵

定义目标矩阵goal_matrix，包含数字1到nn，并用reshape方法将其变为nn的矩阵形式。

定义启发式函数h(state)，用于计算当前状态与目标状态之间的曼哈顿距离。具体实现如下：

初始化h_score为0

循环遍历矩阵中的每一个元素：

用np.where函数找到该元素在当前状态矩阵和目标状态矩阵中的位置

计算该元素在当前状态和目标状态之间的曼哈顿距离，并加入h_score中

返回h_score作为当前状态与目标状态之间的曼哈顿距离

定义状态类State，包含当前状态矩阵state_matrix和移动步骤moves。具体实现如下：

初始化方法__init__(self, state_matrix, moves)：传入state_matrix和moves，并分别赋值给对象的state_matrix和moves属性。

定义方法get_next_states(self)，用于获取所有可能的下一步状态。具体实现如下：

初始化空列表next_states

找到空格在矩阵中的位置，用np.where函数获取其坐标

如果空格不在第一行，则可以上移，将空格和上面的元素互换位置，生成新的状态。将新状态和步骤"U"加入next_states列表中。

如果空格不在最后一行，则可以下移，将空格和下面的元素互换位置，生成新的状态。将新状态和步骤"D"加入next_states列表中。

如果空格不在第一列，则可以左移，将空格和左边的元素互换位置，生成新的状态。将新状态和步骤"L"加入next_states列表中。

如果空格不在最后一列，则可以右移，将空格和右边的元素互换位置，生成新的状态。将新状态和步骤"R"加入next_states列表中。

返回next_states列表，包含所有可能的下一步状态。

定义A*算法函数，用于求解矩阵排序问题。具体实现如下：

初始化变量：

初始状态start_state，为输入矩阵matrix和空字符串""

目标状态goal_state，为目标矩阵goal_matrix和空字符串""

open_list，为初始状态start_state的列表

closed_list，为已经遍历过的状态的列表


function solve_puzzle(puzzle):
    start_state = (puzzle, '')  # 初始状态，移动步骤为空
    goal_state = get_goal_state(puzzle)  # 目标状态
    frontier = PriorityQueue()  # 优先队列
    frontier.put(start_state, 0)  # 将初始状态加入队列
    explored = set()  # 记录已经探索过的状态

    while not frontier.empty():  # 当队列不为空
        current_state, cost = frontier.get()  # 取出当前状态及其代价
        explored.add(current_state[0].tostring())  # 记录已探索状态
        if current_state[0].tolist() == goal_state.tolist():  # 如果当前状态为目标状态
            return current_state[1]  # 返回移动步骤

        for new_state, action in get_successors(current_state[0]):  # 遍历当前状态的所有可能的后继状态
            new_cost = cost + 1 + heuristic(new_state, goal_state)  # 计算新状态的代价
            if new_state.tostring() not in explored:  # 如果新状态没有被探索过
                frontier.put((new_state, current_state[1] + action), new_cost)  # 将新状态加入队列
            elif frontier.get_cost(new_state) > new_cost:  # 如果新状态已经被探索过但代价更小
                frontier.update_priority((new_state, current_state[1] + action), new_cost)  # 更新优先队列

    return None  # 如果找不到解，返回None

function
A * (start_state, goal_state):
// 初始化起始状态
start_node = Node(state=start_state, g=0, h=heuristic(start_state, goal_state), parent=None)
// 初始化open
list和closed
list
open_list = [start_node]
closed_list = []

while open_list is not empty:
    // 从open
    list中选取估价函数最小的节点
    current_node = node in open_list
    with the lowest f = g + h
    // 将当前节点从open
    list移入closed
    list
    open_list.remove(current_node)
    closed_list.append(current_node)

    // 判断当前节点是否为目标状态
    if current_node.state == goal_state:
        return current_node

    // 获取当前节点的所有邻居节点
    for neighbor in get_neighbors(current_node.state):
        // 如果邻居节点在closed
        list中，则跳过
        if neighbor in closed_list:
            continue
        // 计算邻居节点的实际代价g
        tentative_g = current_node.g + 1
        // 如果邻居节点不在open
        list中，则加入open
        list
        if neighbor not in open_list:
            neighbor_node = Node(state=neighbor, g=tentative_g, h=heuristic(neighbor, goal_state), parent=current_node)
            open_list.append(neighbor_node)
        // 如果邻居节点已经在open
        list中，比较新的g值是否更优
        else:
        neighbor_node = find_node_in_list(neighbor, open_list)
        if tentative_g < neighbor_node.g:
            neighbor_node.g = tentative_g
            neighbor_node.parent = current_node

// open
list为空，无解
return None

function
heuristic(state, goal_state):
// 曼哈顿距离启发式函数
distance = 0
for i in range(state.shape[0]):
    for j in range(state.shape[1]):
        if state[i][j] == 0:
            continue
        x, y = np.where(goal_state == state[i][j])
        distance += abs(x - i) + abs(y - j)
return distance


class Node:
    // 表示搜索树中的节点

    def __init__(self, state, g, h, parent):
        self.state = state
        self.g = g
        self.h = h
        self.parent = parent

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)


function
get_neighbors(state):
// 获取当前状态的所有邻居状态
neighbors = []
rows, cols = state.shape
for i in range(rows):
    for j in range(cols):
        // 向左移动
        if j > 0:
            neighbor = np.copy(state)
            neighbor[i][j] = state[i][j - 1]
            neighbor[i][j - 1] = state[i][j]
            neighbors.append(neighbor)
        // 向右移动
        if j < cols - 1:
            neighbor = np.copy(state)
            neighbor[i][j] = state[i][j + 1]
            neighbor[i][j + 1] = state[i][j]
            neighbors.append(neighbor)
        // 向上移动
        if i > 0:
            neighbor = np.copy(state)
            neighbor[i][j] = state
