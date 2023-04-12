import numpy as np
import heapq

def move_tile(board, n, r, c, dr, dc):
    """
    移动棋盘上(r, c)位置的数字，并返回移动后的新位置和移动的操作
    """
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= n or nc < 0 or nc >= n:
        return r, c, None
    board[r, c], board[nr, nc] = board[nr, nc], board[r, c]
    return nr, nc, str(board[r, c]) + ('L' if dc == -1 else 'R' if dc == 1 else 'U' if dr == -1 else 'D')

def board_to_string(board):
    """
    将棋盘转化为字符串
    """
    return ''.join([str(x) for x in board.reshape(-1)])

def h(state, n):
    """
    启发函数，使用曼哈顿距离
    """
    h_score = 0
    for i in range(n):
        for j in range(n):
            if state[i, j] == 0:
                continue
            val = state[i, j] - 1
            r, c = val // n, val % n
            h_score += abs(i - r) + abs(j - c)
    return h_score

def a_star_search(board, n):
    """
    A*算法求解
    """
    # 初始化
    start_state = board.copy()
    goal_state = np.arange(1, n * n + 1).reshape(n, n)
    goal_state[-1, -1] = 0
    open_set = [(0, start_state)]
    closed_set = set()
    g = {board_to_string(start_state): 0}
    f = {board_to_string(start_state): h(start_state, n)}

    while open_set:
        # 取出开启列表中f值最小的节点
        cur_f, cur_state = heapq.heappop(open_set)
        cur_str = board_to_string(cur_state)

        # 如果达到目标状态，返回移动方式
        if np.array_equal(cur_state, goal_state):
            actions = []
            while cur_str != board_to_string(board):
                actions.append(g[cur_str][1])
                cur_str = g[cur_str][0]
            return actions[::-1]

        # 将节点加入到关闭列表中
        closed_set.add(cur_str)

        # 扩展节点
        for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_r, new_c, action = move_tile(cur_state, n, *np.argwhere(cur_state == 0)[0], dr, dc)
            if action is None:
                continue
            new_state_str = board_to_string(cur_state)
            new_g = g[cur_str] + [(new_r, new_c, action)]
            new_f = h(cur_state, n) + len(new_g)
            if new_state_str not in closed_set and (new_f, new_state_str) not in open_set:
                g[new_state_str] = new_g
                f[new_state_str] = new_f
                heapq.heappush(open_set, (new_f, cur_state.copy()))

    #
