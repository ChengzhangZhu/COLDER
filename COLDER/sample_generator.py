"""
This code generates training samples

Subsampling Frequent Data + Negative Sampling

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""
from context_generator import social_implicit_path_generator, path2context
from graph import SocialGraph
import numpy as np
import pickle
from tqdm import tqdm
import argparse


def negative_sampling_prob(g=SocialGraph(), random_path=None):
    init_u = np.zeros(len(g.node_u))
    init_i = np.zeros(len(g.node_i))
    users = g.node_u.keys()
    items = g.node_i.keys()
    user_sample_prob = dict(zip(users,init_u))
    item_sample_prob = dict(zip(items,init_i))
    user_context = random_path['user_path']
    item_context = random_path['item_path']
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
    return np.asarray(user_sample_prob.values()), np.asarray(item_sample_prob.values())


def negative_sample(sets, conflicts, p=None):
    complete = False
    for i in sets:
        if i not in conflicts:
            complete = True
            break
    while complete:
        if p is None:
            sample = np.random.choice(sets)
        else:
            sample = np.random.choice(sets,p=p)
        if sample not in conflicts:
            return sample
    return None


def fast_negative_sample(random_sets, conflicts, index):
    max_len = len(random_sets)
    count = 0
    while True:
        if count == max_len:
            break
        if random_sets[index] not in conflicts:
            index += 1
            if index == max_len:
                index = 0
            return random_sets[index], index
        index += 1
        if index == max_len:
            index = 0
        count += 1
    return random_sets[index], index


def shuffle_samples(samples):
    num_samples = len(samples.values()[0])
    index = np.r_[0:num_samples]
    np.random.shuffle(index)
    for col in samples:
        samples[col] = [samples[col][i] for i in index]
    return samples


random_sample_index = 0


def update_sample_index(sample_index_set):
    global random_sample_index
    sample_index = sample_index_set[random_sample_index]
    random_sample_index += 1
    if random_sample_index == len(sample_index_set):
        random_sample_index = 0
    return sample_index


def sample_generator(g=SocialGraph(), random_path=None):
    samples = {'user1':[], 'item1':[], 'review1':[], 'rating1':[], 'label1':[], 'context_u':[], 'success1':[],
               'user2':[], 'item2':[], 'review2':[], 'rating2':[], 'label2':[], 'context_i':[], 'success2':[]}
    print('Transfer random path to context...')
    user_context = path2context(g.node_u.keys(), random_path['user_path'])
    item_context = path2context(g.node_i.keys(), random_path['item_path'])
    user_nodes = np.asarray(g.node_u.keys())  # the user id
    item_nodes = np.asarray(g.node_i.keys())  # the item id
    print('Calculate negative sampling probability...')
    user_sample_prob, item_sample_prob = negative_sampling_prob(g, random_path)
    user_sample_index = 0  # the user negative sampling index
    item_sample_index = 0 # the item negative sampling index
    negative_user_set = np.random.choice(user_nodes, size=len(user_nodes)*10, p=user_sample_prob)
    negative_item_set = np.random.choice(item_nodes, size=len(item_nodes)*10, p=item_sample_prob)
    # Generate user samples
    num_reviews = len(g.review)  # number of reviews
    sample_index = np.r_[0:num_reviews]  # the index set used to random selection
    np.random.shuffle(sample_index)
    review_keys = g.review.keys()
    uni_labels = np.unique(g.label.values())  # unique labels
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
                samples['context_u'].append(1)
                negative_item = negative_sample(sets=item_nodes, conflicts=other_items)
                if negative_item is None:
                    negative_item = np.random.choice(item_nodes)
                    samples['success2'].append(1)
                else:
                    samples['success2'].append(0)
                samples['user2'].append(other_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(g.review[review_keys[update_sample_index(sample_index)]])

                samples['rating2'].append(g.rating[review_keys[update_sample_index(sample_index)]])
                samples['label2'].append(np.random.choice(uni_labels))
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
                negative_user, user_sample_index = fast_negative_sample(random_sets=negative_user_set, conflicts=user_context[user], index=user_sample_index)
                if negative_user is None:
                    negative_user = np.random.choice(user_nodes)
                    samples['context_u'].append(1)
                    samples['context_u'].append(1)
                else:
                    samples['context_u'].append(0)
                    samples['context_u'].append(0)
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
                negative_item = negative_sample(sets=item_nodes, conflicts=other_items)
                if negative_item is None:
                    negative_item = np.random.choice(item_nodes)
                    samples['success2'].append(1)
                else:
                    samples['success2'].append(0)
                samples['user2'].append(negative_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(g.review[review_keys[update_sample_index(sample_index)]])
                samples['rating2'].append(g.rating[review_keys[update_sample_index(sample_index)]])
                samples['label2'].append(np.random.choice(uni_labels))
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
                if negative_user is None:
                    negative_user = np.random.choice(user_nodes)
                    samples['success2'].append(1)
                else:
                    samples['success2'].append(0)
                samples['user2'].append(negative_user)
                samples['item2'].append(item)
                samples['review2'].append(g.review[review_keys[update_sample_index(sample_index)]])
                samples['rating2'].append(g.rating[review_keys[update_sample_index(sample_index)]])
                samples['label2'].append(np.random.choice(uni_labels))
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
                negative_item, item_sample_index = fast_negative_sample(random_sets=negative_item_set, conflicts=item_context[item], index=item_sample_index)
                if negative_item is None:
                    negative_item = np.random.choice(item_nodes)
                    samples['context_i'].append(1)
                    samples['context_i'].append(1)
                else:
                    samples['context_i'].append(0)
                    samples['context_i'].append(0)
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
                negative_user = negative_sample(sets=user_nodes, conflicts=other_users)
                if negative_user is None:
                    negative_user = np.random.choice(user_nodes)
                    samples['success2'].append(1)
                else:
                    samples['success2'].append(0)
                samples['user2'].append(negative_user)
                samples['item2'].append(negative_item)
                samples['review2'].append(g.review[review_keys[update_sample_index(sample_index)]])
                samples['rating2'].append(g.rating[review_keys[update_sample_index(sample_index)]])
                samples['label2'].append(np.random.choice(uni_labels))
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
parser.add_argument('--save_name', default='Zip', help='Set the save name of the generated samples,'
                                                               'default is Zip', dest='save_name', type=str)
parser.add_argument('--load_random_path', default='N', help='Load existing random path (Y/N)', dest='load_random_path', type=str)
args = parser.parse_args()


# ## For test
# print('Building Graph...')
# graph = SocialGraph()
# graph.build('test.csv')
# print('Saving Graph...')
# pickle.dump(graph, open('{}_graph.pkl'.format('test'),'wb'))
# print('Generating Random Path...')
# random_path = social_implicit_path_generator(graph, minT=args.minT, maxT=args.maxT, p=args.p, max_length=args.max_length)
# print('Saving Random Path...')
# pickle.dump(random_path, open('{}_path.pkl'.format('test'),'wb'))
# print('Construct Samples...')
# samples = sample_generator(graph, random_path)
# pickle.dump(samples,open('{}_sample.pkl'.format('test'),'wb'))


def main():
    if args.load_random_path != 'Y':
        print('Building Graph...')
        graph = SocialGraph()
        graph.build(args.data_sets)
        print('Saving Graph...')
        pickle.dump(graph, open('{}_graph.pkl'.format(args.save_name),'wb'))
        print('Generating Random Path...')
        random_path = social_implicit_path_generator(graph, minT=args.minT, maxT=args.maxT, p=args.p, max_length=args.max_length)
        print('Saving Random Path...')
        pickle.dump(random_path, open('{}_path.pkl'.format(args.save_name),'wb'))
    else:
        print('Loading Graph...')
        graph = pickle.load(open('{}_graph.pkl'.format(args.save_name),'rb'))
        print('Loading Random Path...')
        random_path = pickle.load(open('{}_path.pkl'.format(args.save_name), 'rb'))
    print('Construct Samples...')
    samples = sample_generator(graph, random_path)
    pickle.dump(samples,open('{}_sample.pkl'.format(args.save_name),'wb'))
    # return None


if __name__ == "__main__":
    main()