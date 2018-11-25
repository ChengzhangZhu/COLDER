"""
This code generates training samples

Subsampling Frequent Data + Negative Sampling

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""

from context_generator import social_implicit_context_generator
from graph import SocialGraph
import numpy as np
import pickle
from tqdm import tqdm
import argparse


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


def negative_sample(sets, conflicts, p=None):
    while True:
        if p is None:
            sample = np.random.choice(sets)
        else:
            sample = np.random.choice(sets,p=p)
        if sample not in conflicts:
            return sample


def shuffle_samples(samples):
    num_samples = len(samples.values()[0])
    index = np.r_[0:num_samples]
    np.random.shuffle(index)
    for col in samples:
        samples[col] = [samples[col][i] for i in index]
    return samples


def sample_generator(g=SocialGraph(), minT=1, maxT=32, p=0.15, max_length=5):
    samples = {'user1':[], 'item1':[], 'review1':[], 'rating1':[], 'label1':[], 'context_u':[], 'success1':[],
               'user2':[], 'item2':[], 'review2':[], 'rating2':[], 'label2':[], 'context_i':[], 'success2':[]}
    context = social_implicit_context_generator(g, minT=minT, maxT=maxT, p=p, max_length=max_length)
    user_context = context['user_context']
    item_context = context['item_context']
    user_nodes = g.node_u.keys()  # the user id
    item_nodes = g.node_i.keys()  # the item id
    user_sample_prob, item_sample_prob = negative_sampling_prob(g, context)
    # Generate user samples
    print('Generate user samples')
    for user in tqdm(user_nodes):
        items = g.node_u[user]
        # item = np.random.choice(items)
        for item in items:
            # # Negative in context success pair
            for other_user in user_context[user]:
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user,item)])
                samples['rating1'].append(g.rating[(user,item)])
                samples['label1'].append(g.label[(user,item)])
                samples['success1'].append(1)
                samples['context_u'].append(1)
                other_items = g.node_u[other_user]
                other_item = np.random.choice(other_items)
                samples['user2'].append(other_user)
                samples['item2'].append(other_item)
                samples['review2'].append(g.review[(other_user,other_item)])
                samples['rating2'].append(g.rating[(other_user,other_item)])
                samples['label2'].append(g.label[(other_user,other_item)])
                samples['success2'].append(1)
                if other_item in item_context[item]:
                    samples['context_i'].append(1)
                else:
                    samples['context_i'].append(0)
            # # Negative in context not success pair
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user, item)])
                samples['rating1'].append(g.rating[(user, item)])
                samples['label1'].append(g.label[(user, item)])
                samples['success1'].append(1)
                samples['context'].append(1)
                negative_item = negative_sample(sets=item_nodes, conflicts=other_items)
                samples['user2'].append(other_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(np.random.choice(g.review.values()))
                samples['rating2'].append(np.random.choice(g.rating.values()))
                samples['label2'].append(np.random.choice(g.label.values()))
                samples['success2'].append(0)
                if negative_item in item_context[item]:
                    samples['context_i'].append(1)
                else:
                    samples['context_i'].append(0)
            # # Not in context sampling success pair
            for i in range(len(user_context[user])):
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user, item)])
                samples['rating1'].append(g.rating[(user, item)])
                samples['label1'].append(g.label[(user, item)])
                samples['success1'].append(1)
                samples['context'].append(0)
                negative_user = negative_sample(sets=user_nodes, conflicts=user_context[user], p=user_sample_prob)
                other_items = g.node_u[negative_user]
                other_item = np.random.choice(other_items)
                samples['user2'].append(negative_user)
                samples['item2'].append(other_item)
                samples['review2'].append(g.review[(negative_user, other_item)])
                samples['rating2'].append(g.rating[(negative_user, other_item)])
                samples['label2'].append(g.label[(negative_user, other_item)])
                samples['success2'].append(1)
                if other_item in item_context[item]:
                    samples['context_i'].append(1)
                else:
                    samples['context_i'].append(0)
            # # Not in context sampling not success pair
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user, item)])
                samples['rating1'].append(g.rating[(user, item)])
                samples['label1'].append(g.label[(user, item)])
                samples['success1'].append(1)
                samples['context'].append(0)
                negative_item = negative_sample(sets=item_nodes, conflicts=other_items)
                samples['user2'].append(negative_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(np.random.choice(g.review.values()))
                samples['rating2'].append(np.random.choice(g.rating.values()))
                samples['label2'].append(np.random.choice(g.label.values()))
                samples['success2'].append(0)
                if negative_item in item_context[item]:
                    samples['context_i'].append(1)
                else:
                    samples['context_i'].append(0)
    # Generate item samples
    print('Generate item samples')
    for item in tqdm(item_nodes):
        users = g.node_i[item]
        # user = np.random.choice(users)
        for user in users:
            # # Negative in context success pair
            for other_item in item_context[item]:
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user,item)])
                samples['rating1'].append(g.rating[(user,item)])
                samples['label1'].append(g.label[(user,item)])
                samples['success1'].append(1)
                samples['context_i'].append(1)
                other_users = g.node_i[other_item]
                other_user = np.random.choice(other_users)
                samples['user2'].append(other_user)
                samples['item2'].append(other_item)
                samples['review2'].append(g.review[(other_user,other_item)])
                samples['rating2'].append(g.rating[(other_user,other_item)])
                samples['label2'].append(g.label[(other_user,other_item)])
                samples['success2'].append(1)
                if other_user in user_context[user]:
                    samples['context_u'].append(1)
                else:
                    samples['context_u'].append(0)
            # # Negative in context not success pair
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user, item)])
                samples['rating1'].append(g.rating[(user, item)])
                samples['label1'].append(g.label[(user, item)])
                samples['success1'].append(1)
                samples['context_i'].append(1)
                negative_user = negative_sample(sets=user_nodes, conflicts=other_users)
                samples['user2'].append(negative_user)
                samples['item2'].append(item)
                samples['review2'].append(np.random.choice(g.review.values()))
                samples['rating2'].append(np.random.choice(g.rating.values()))
                samples['label2'].append(np.random.choice(g.label.values()))
                samples['success2'].append(0)
                if negative_user in user_context[user]:
                    samples['context_u'].append(1)
                else:
                    samples['context_u'].append(0)
            # # Not in context sampling success pair
            for i in range(len(item_context[item])):
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user, item)])
                samples['rating1'].append(g.rating[(user, item)])
                samples['label1'].append(g.label[(user, item)])
                samples['success1'].append(1)
                samples['context_i'].append(0)
                negative_item = negative_sample(sets=item_nodes, conflicts=item_context[item], p=item_sample_prob)
                other_users = g.node_i[negative_item]
                other_user = np.random.choice(other_users)
                samples['user2'].append(other_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(g.review[(other_user, negative_item)])
                samples['rating2'].append(g.rating[(other_user, negative_item)])
                samples['label2'].append(g.label[(other_user, negative_item)])
                samples['success2'].append(1)
                if other_user in user_context[user]:
                    samples['context_u'].append(1)
                else:
                    samples['context_u'].append(0)
            # # Not in context sampling not success pair
                samples['user1'].append(user)
                samples['item1'].append(item)
                samples['review1'].append(g.review[(user, item)])
                samples['rating1'].append(g.rating[(user, item)])
                samples['label1'].append(g.label[(user, item)])
                samples['success1'].append(1)
                samples['context_i'].append(0)
                negative_user = negative_sample(sets=user_nodes, conflicts=other_users)
                samples['user2'].append(negative_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(np.random.choice(g.review.values()))
                samples['rating2'].append(np.random.choice(g.rating.values()))
                samples['label2'].append(np.random.choice(g.label.values()))
                samples['success2'].append(0)
                if negative_user in user_context[user]:
                    samples['context_u'].append(1)
                else:
                    samples['context_u'].append(0)
    return shuffle_samples(samples)

parser = argparse.ArgumentParser(description='Sample Generator')
parser.add_argument('--data_sets', default='/data/qli1/Experiment/Qian/Fraud Detection/Yelp Shibuti Datasets/Data/yelp_Zip_data.csv', help='Specific Data Set',
                    dest='data_sets', type=str)
parser.add_argument('--minT', default=1, help='Set the minimun walk for each node, default is 1',
                    dest='minT', type=int)
parser.add_argument('--maxT', default=32, help='Set the maximun walk for each node, default is 32',
                    dest='maxT', type=int)
parser.add_argument('--max_length', default=5, help='Set the maximun walk length, default is 5',
                    dest='max_length', type=int)
parser.add_argument('--p', default=0.15, help='Set the walk stop probability at each step, default is 0.15',
                    dest='p', type=float)
parser.add_argument('--save_name', default='samples.pkl', help='Set the save name of the generated samples,'
                                                               'default is samples.pkl', dest='save_name', type=str)
args = parser.parse_args()


def main():
    print('Constrcut Graph')
    graph = SocialGraph()
    graph.build(args.data_sets)
    print('Construct Samples')
    samples = sample_generator(graph, minT=args.minT, maxT=args.maxT, p=args.p, max_length=args.max_length)
    pickle.dump(samples,open(args.save_name,'wb'))

if __name__ == "__main__":
    main()