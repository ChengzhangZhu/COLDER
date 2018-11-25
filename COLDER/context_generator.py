"""
This code generates context information for social relation and behavior

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""

from graph import SocialGraph, Graph
import numpy as np
from tqdm import tqdm


def weighted_random_walk_generator(g=Graph(), minT=1, maxT=32, p=0.15, max_length=5):  # epoch determines the number of walks
    """

    :param g: Graph class
    :param minT: the minimum number of walk start from a node
    :param maxT: the maximum number of walk start from a node
    :param p: the stop probability of at each step
    :return: walk path
    """
    nodes = g.node.keys()  # the node list
    degree = list()  # record the degree of nodes
    centrality = list()  # the centrality of each node, here use degree centrality
    total_degree = 0
    for node in nodes:
        degree = g.degree[node]
    degree = np.asarray(degree)
    degree = (degree - np.min(degree))/(np.max(degree) - np.min(degree))  # normalize the degree to avoid overflow in exp
    for i in range(len(nodes)):
        total_degree += np.exp(degree[i])
        centrality.append(np.exp(degree[i]))
    centrality = np.asarray(centrality)*1.0/total_degree
    walk_path = []  # walking path
    for i, node in tqdm(enumerate(nodes)):
        t = np.max([centrality[i]*maxT, minT])  # the number of walk for a node
        for step in range(int(t)):
            walk_path.append(weighted_random_walk(g, node, p, max_length))
    return walk_path


def weighted_random_walk(g=Graph(), node=0, p=0.15, max_length=5):  # random work with stop probability and max length (window size of context)
    walk_path = [node]
    walk = True
    while walk:
        candidate = g.node[walk_path[-1]]
        walk_prob = []
        total_weight = 0
        for n in candidate:  # calculate the probability of the next node
            walk_prob.append(g.edge[(walk_path[-1],n)])
            total_weight += g.edge[(walk_path[-1],n)]
        walk_prob = np.asarray(walk_prob)*1.0/total_weight
        n = np.random.choice(candidate,p=walk_prob)
        walk_path.append(n)
        if np.random.rand() < p or len(walk_path) == max_length:
            walk = False
    return walk_path


def path2context(nodes, walk_path):
    context = dict()
    for node in nodes:
        context[node] = []
    for path in tqdm(walk_path):
        context_nodes = np.unique(path)
        for node in context_nodes:
            for c_node in context_nodes:
                if c_node is not node:
                    context[node].append(c_node)
    for node in nodes:
        context[node] = np.unique(context[node])
    return context


def social_implicit_context_generator(g=SocialGraph(), minT=1, maxT=32, p=0.15, max_length=5):
    social_context = dict()
    # Stage 1. Split social graph into user-user and item-item graphs
    user_graph, item_graph = g.split()

    # Stage 2. Random walk in user-user graph
    print('Random walk in user-user graph')
    user_path = weighted_random_walk_generator(user_graph, minT=minT, maxT=maxT, p=p, max_length=max_length)
    social_context['user_context'] = user_path
    # Stage 3. Random walk in item-item graph
    print('Random walk in item-item graph')
    item_path = weighted_random_walk_generator(item_graph, minT=minT, maxT=maxT, p=p, max_length=max_length)
    social_context['item_context'] = item_path
    return social_context
