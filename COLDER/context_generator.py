"""
This code generates context information for social relation and behavior

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""

from .graph import SocialGraph, Graph

def random_walk(g=Graph()):
    walk_path = None
    return walk_path


def social_implicit_context_generator(g=SocialGraph()):
    social_context = {'user_context':[], 'item_context':[]}
    # Stage 1. Split social graph into user-user and item-item graphs
    user_graph, item_graph = g.split()

    # Stage 2. Random walk in user-user graph


    # Stage 3. Random walk in item-item graph

    return social_context
