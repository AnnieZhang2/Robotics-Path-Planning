import heapq
import math
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, cost, pind, theta):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node 父节点的索引值
        self.theta = theta


class Para:
    def __init__(self, minx, miny, mintheta, maxx, maxy, maxtheta, xw, yw, thetaw, reso, motion, rotate):
        self.minx = minx
        self.miny = miny
        self.mintheta = mintheta
        self.maxx = maxx
        self.maxy = maxy
        self.maxtheta = maxtheta
        self.xw = xw
        self.yw = yw
        self.thetaw = thetaw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set
        self.rotate = rotate #机器人旋转


def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """

    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1,0)
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1,0)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, P)] = n_start

    q_priority = []
    heapq.heappush(q_priority,
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority) #从队列中取出cost最小的索引值
        n_curr = open_set[ind] #从open_set中取出这个节点
        closed_set[ind] = n_curr #把这个节点放入closed_set，表示已经处理过
        open_set.pop(ind) #把这个节点从open_set中删掉
       #这里需要改一下：不仅仅有上下左右斜8个动作，还有增加或减小theta的动作，应该套有两个循环
        for i in range(len(P.motion)):
            for j in range(3):
                node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]) + abs(0.1*P.rotate[j]), ind, n_curr.theta + P.rotate[j])

                if not check_node(node, P, obsmap): #collision check
                    continue

                n_ind = calc_index(node, P) #当前节点的索引值
                if n_ind not in closed_set:
                    if n_ind in open_set:
                        if open_set[n_ind].cost > node.cost:
                            open_set[n_ind].cost = node.cost
                            open_set[n_ind].pind = ind
                    else:#既不在close也不在open：从来没探索过
                        open_set[n_ind] = node
                        heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P))) #把这个节点添加至队列中
    #fvalue,这里存疑 这里就是加上了距离终点的成本，确保我们是逐步逼近终点而没有走的更远，那么角度怎么加上去合适捏？

    pathx, pathy, paththeta = extract_path(closed_set, n_start, n_goal, P)

    return pathx, pathy, paththeta

def check_node(node, P, obsmap):#判断L型这三个点是否在障碍物内即可
    #如果是一个L型
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy or node.theta <P.mintheta or node.theta >P.maxtheta:
        return False
    #这个是中心点判断
    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False
    #直角边点1判断
    x1 = node.x + math.cos(node.theta)
    y1 = node.y + math.sin(node.theta)
    if isinstance(x1,int) and isinstance(y1,int):
        if obsmap[x1-P.minx][y1-P.miny]:
            return False
    #直角边点2判断
    x2 = node.x + math.cos(node.theta + 90)
    y2 = node.y + math.sin(node.theta + 90)
    if isinstance(x2,int) and isinstance(y2,int):
        if obsmap[x2-P.minx][y2-P.miny]:
            return False

    return True


def u_cost(u):
    return math.hypot(u[0], u[1])


def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)


def h(node, n_goal):#当前节点与终点的位移
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y) + 0.1*abs(node.theta - n_goal.theta)


def calc_index(node, P):#索引值需要重新设置
    return (node.y - P.miny) * P.xw * P.thetaw + (node.x - P.minx) * P.thetaw + node.theta


def calc_parameters(ox, oy, rr, reso): #环境参数稍微改一下
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = maxx - minx, maxy - miny
    thetaw = 360
    mintheta = 0
    maxtheta = 359

    rotate = get_rotate()
    motion = get_motion()
    P = Para(minx, miny, mintheta, maxx, maxy, maxtheta, xw, yw, thetaw, reso, motion,rotate)
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap


def calc_obsmap(ox, oy, rr, P): #这个主要说明在机器人范围内的点是否是障碍
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]
    #全地图false

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy): #这个划定了范围，要改一下
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    return obsmap


def extract_path(closed_set, n_start, n_goal, P):
    pathx, pathy ,paththeta= [n_goal.x], [n_goal.y], [n_goal.theta]
    n_ind = calc_index(n_goal, P)

    while True:
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        paththeta.append(node.theta)
        n_ind = node.pind

        if node == n_start:
            break

    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]
    paththeta = [theta for theta in reversed(paththeta)]

    return pathx, pathy, paththeta


def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion

def get_rotate():
    rotate = [10,1,-10]
    return rotate

def get_env():
    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    return ox, oy


def main():
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    robot_radius = 2.0
    grid_resolution = 1.0
    ox, oy = get_env()

    pathx, pathy, paththeta = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

    plt.plot(ox, oy, 'sk')
    plt.plot(pathx, pathy, '-r')
    plt.plot(sx, sy, 'sg')
    plt.plot(gx, gy, 'sb')

# 添加theta值的标签
    for i in range(len(paththeta)):
        plt.text(pathx[i], pathy[i], f"{paththeta[i]:.2f}", fontsize=8, ha='right')

    plt.axis("equal")
    plt.show()



if __name__ == '__main__':
    main()
