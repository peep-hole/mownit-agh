import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import randrange


def read_graph(file_name, s, t, E):

    file = open(file_name, "r")
    lines = file.readlines()
    G = nx.Graph()
    tensioned = False

    for line in lines:
        line = line.strip("\n").split(":")
        if len(line) != 3:
            raise AttributeError

        from_v = int(line[0], 10)
        to_v = int(line[1], 10)
        resist = float(line[2])
        if from_v not in G.nodes:
            G.add_node(from_v, real=True)

        if to_v not in G.nodes:
            G.add_node(to_v, real=True)

        if {from_v, to_v} == {s, t}:
            tensioned = True
            if s < t:
                tension = E
            else:
                tension = -E

            G.add_edge(from_v, to_v, resist=resist, current=0, tension=tension)
        else:
            G.add_edge(from_v, to_v, resist=resist, current=0, tension=0)

    if not tensioned:
        print("there is no edge between ", s, " and ", t, )
        exit(1)

    return G


def to_color(max_tension, tension):
    if tension < 0.3 * max_tension:
        return 'r'
    if tension < 0.6 * max_tension:
        return 'y'
    return 'g'


def does_cycle_contains(s, t, cycle):
    for i, el in enumerate(cycle):
        if el == s and i + 1 < len(cycle) and cycle[i+1] == t:
            return True
        elif el == t and i + 1 < len(cycle) and cycle[i+1] == s:
            return True
    return False


def edge_to_index(G, s, t):
    for i, el in enumerate(G.edges):
        if {s, t} == {el[0], el[1]}:
            return i

    print("there is no edge ", s, " ", t)
    exit(1)


def edge_to_tuple(G, s, t):
    for i, el in enumerate(G.edges):
        if (s, t) == (el[0], el[1]):
            return tuple([s, t])
        elif (t, s) == (el[0], el[1]):
            return tuple([t, s])
    return None


def get_first_max_degree_node_index(G):
    nodes = G.nodes
    deg = [G.degree[i] for i in nodes]
    max_ind = 0
    for i, d in enumerate(deg):
        if d > deg[max_ind]:
            max_ind = i
    return max_ind


def plot_result(G, X, resists, precision):

    def plot_graph(G):

        plt.subplot(111)
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'tension')
        colors = nx.get_edge_attributes(G, 'color').values()
        nx.draw_networkx(G, pos, node_color='g', edge_color=colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    R = nx.DiGraph()
    edge_list = list(G.edges)
    tens = [round(abs(X[i])*resists[edge_list[i]],precision) for i, _ in enumerate(G.edges)]
    max_tens = max(tens)
    for i, edge in enumerate(G.edges):
        if X[i] < 0:
            R.add_edge(max(edge), min(edge), tension=tens[i], color=to_color(max_tens, tens[i]))
        else:
            R.add_edge(min(edge), max(edge), tension=tens[i], color=to_color(max_tens, tens[i]))
    plot_graph(R)


def kirchhoff(file_name, s, t, E, precision):
    G = read_graph(file_name, s, t, E)
    if s not in G.nodes or t not in G.nodes:
        print("s or t not in graph")
        exit(1)

    simple_cycles = nx.cycle_basis(G)

    check_array = [False]*len(simple_cycles)

    A = [[0 for i in range(len(G.edges))] for j in range(len(G.edges))]
    B = [0] * len(G.edges)

    resists = nx.get_edge_attributes(G, 'resist')
    tensions = nx.get_edge_attributes(G, 'tension')

    # finding first max degree node index so it can be omitted in system of equations
    x = 0
    max_ind = get_first_max_degree_node_index(G)
    for i, node in enumerate(G.nodes):
        if i == max_ind:
            x = 1
            continue
        for neigh in nx.neighbors(G, node):
            edge_index = edge_to_index(G, node, neigh)
            if node < neigh:
                A[i - x][edge_index] = -1
            else:
                A[i - x][edge_index] = 1

    last_index = len(G.nodes) - 1

    # adding cycle equations to system prioritizing the ones containing source
    for i, cycle in enumerate(simple_cycles):
        if len(G.edges) <= last_index:
            break
        if does_cycle_contains(s, t, cycle):
            for j in range(len(cycle)):
                from_index = j
                to_index = (j + 1) % (len(cycle))
                edge_index = edge_to_index(G, cycle[from_index], cycle[to_index])
                edge_tuple = edge_to_tuple(G, cycle[from_index], cycle[to_index])
                res = resists[edge_tuple]
                e = tensions[edge_tuple]
                if cycle[from_index] > cycle[to_index]:
                    res *= -1
                    e *= -1
                A[last_index][edge_index] = res
                B[last_index] += e
            last_index += 1
            check_array[i] = True

    # adding to system rest of cycle equations if any left (check_array)
    for i, cycle in enumerate(simple_cycles):
        if len(G.edges) <= last_index:
            break
        if not check_array[i]:
            for j in range(len(cycle)):
                from_index = j
                to_index = (j + 1) % (len(cycle))
                edge_index = edge_to_index(G, cycle[from_index], cycle[to_index])
                edge_tuple = edge_to_tuple(G, cycle[from_index], cycle[to_index])
                res = resists[edge_tuple]
                e = tensions[edge_tuple]
                if cycle[from_index] > cycle[to_index]:
                    res *= -1
                    e *= -1
                A[last_index][edge_index] = res
                B[last_index] += e
            last_index += 1
            check_array[i] = True

    X = np.linalg.solve(A, B)

    plot_result(G, X, resists, precision)
    return X



def node_index_to_matrix_index(G, old_source, new, index):
    sorted_list = sorted(list(G.nodes))
    x = 0
    for i, el in enumerate(sorted_list):
        if el == old_source or el == new:
            x += 1
        elif el == index:
            return i - x
    print("Cold not find node index ", index, " in graph")
    exit(1)


def matrix_index_to_node_index(G, old_source, new, index):
    sorted_listed = sorted(list(G.nodes))
    x = 0
    for i, el in enumerate(sorted_listed):
        if el == old_source or el == new:
            x += 1
        elif index == i - x:
            return el
    print("Cold not find node index ", index, " in matrix")
    exit(1)


def node_res_to_universal_res(G_old, X, E, s, t, new):
    result = [0 for i in range(len(G_old.edges))]
    resists = nx.get_edge_attributes(G_old, 'resist')
    for edge in list(G_old.edges):
        ind = edge_to_index(G_old, edge[0], edge[1])
        res = resists[edge]
        if edge[0] == s or edge[1] == s:
            if {edge[0], edge[1]} == {s, t}:
                v1 = E if edge[0] == s else X[node_index_to_matrix_index(G_old, s, new, edge[0])]
                v2 = E if edge[1] == s else X[node_index_to_matrix_index(G_old, s, new, edge[1])]
                result[ind] = (v1 - v2)/res
            else:
                if edge[0] != s:
                    i = node_index_to_matrix_index(G_old, s, new, edge[0])
                else:
                    i = node_index_to_matrix_index(G_old, s, new, edge[1])

                result[ind] = (-1)*X[i]/res
            continue

        i1 = node_index_to_matrix_index(G_old, s, new, edge[0])
        i2 = node_index_to_matrix_index(G_old, s, new, edge[1])

        v1 = X[i1]
        v2 = X[i2]

        result[ind] = (v1 - v2)/res

    return result


def node_voltage_method(filename, s, t, E, precision):
    G = read_graph(filename, s, t, E)
    old_G = G.copy()
    resist_tmp = nx.get_edge_attributes(G, 'resist')[edge_to_tuple(G, s, t)]
    G.remove_edge(s, t)
    new_node_index = min(list(G.nodes)) - 1
    G.add_node(new_node_index, real=False)
    G.add_edge(s, new_node_index, resist=0, tension=E if s < t else -E)
    G.add_edge(new_node_index, t, resist=resist_tmp, tension=0)

    A = [[0 for i in range(len(G.nodes) - 2)] for j in range(len(G.nodes) - 2)]
    B = [0 for i in range(len(G.nodes) - 2)]

    resists = nx.get_edge_attributes(G, 'resist')

    for i in range(len(A)):
        accumulator = 0
        node_ind = matrix_index_to_node_index(G, s, new_node_index, i)
        for neigh in nx.neighbors(G, node_ind):

            res = resists[edge_to_tuple(G, node_ind, neigh)]
            val = 1.0 / res
            accumulator -= val

            if neigh == s:
                continue
            if neigh == new_node_index:
                val *= E
                B[i] -= val*E
                continue

            neigh_ind = node_index_to_matrix_index(G, s, new_node_index, neigh)

            A[i][neigh_ind] = val

        A[i][i] = accumulator

    X = np.linalg.solve(A, B)
    uni = node_res_to_universal_res(old_G, X, E, s, t, new_node_index)

    plot_result(old_G, uni, nx.get_edge_attributes(old_G, 'resist'), precision)
    return uni


def generate_erdos(n):
    file = open("erdos.txt", "w")
    for i in range(1, n + 1):
        for j in range(i+1, n + 1):
            res = randrange(1, 10000) / 500.0
            file.write(str(i) + ":" + str(j) + ":" + str(res) + '\n')

    file.close()

