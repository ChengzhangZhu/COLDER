"""
This code generates training samples

Subsampling Frequent Data + Negative Sampling

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""

from context_generator import social_implicit_context_generator
from graph import SocialGraph, Graph
import numpy as np


def negative_sampling_prob(g=SocialGraph(), context=None):
    init_u = np.zeros(len(g.node_u))
    init_i = np.zeros(len(g.node_i))
    users = g.node_u.keys()
    items = g.node_i.keys()
    user_sample_prob = dict(zip(users,init_u))
    item_sample_prob = dict(zip(items,init_i))
    user_context = context['user_context']
    item_context = context['item_context']
    user_total = 0
    item_total = 0
    for c in user_context:
        for i in c:
            user_sample_prob[i] += 1
            user_total += 1
    for c in item_context:
        for i in c:
            item_sample_prob[i] += 1
            item_total += 1
    for i in user_sample_prob:
        user_sample_prob[i] = user_sample_prob[i]*1.0/user_total
    for i in item_sample_prob:
        item_sample_prob[i] = item_sample_prob[i]*1.0/item_total
    return user_sample_prob, item_sample_prob


def sample_generator(g=SocialGraph()):
    samples = {'user1':[], 'item1':[], 'review1':[], 'rating1':[], 'label1':[], 'mask1':[],
               'user2':[], 'item2':[], 'review2':[], 'rating2':[], 'label2':[], 'mask2':[]}
    context = social_implicit_context_generator(g)
    


    return samples
