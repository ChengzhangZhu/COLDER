"""
This code defines a social graph class, and is to build social graph from a social review data set.

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""
import pandas as pd
import re
from tqdm import tqdm

def clean_str(string):
    try:
        string = string.decode('utf-8', 'ignore').encode('utf-8')
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        string = re.sub(r"<sssss>", "", string)
    except:
        print(string)
        return string
    return str(string).strip().lower()


class Graph:
    def __init__(self):
        self.node = dict()  # node dictionary, node i --> node j
        self.edge = dict()  # edge dictionary, (node i, node j) --> edge weight
        self.degree = dict()  # node degree dictionary, node --> degree

    def build(self, triplets):
        print('Graph building...')
        for triplet in tqdm(triplets):
            self.transform_triplet(triplet)

    def transform_triplet(self, triplet):
        n1, n2, w = triplet
        if n1 not in self.node:
            self.node[n1] = []
            self.degree[n1] = 0
        if n2 not in self.node:
            self.node[n2] = []
            self.degree[n2] = 0
        if (n1, n2) not in self.edge:
            self.edge[(n1, n2)] = 0
            self.edge[(n2, n1)] = 0
        self.node[n1].append(n2)
        self.node[n2].append(n1)
        self.edge[(n1, n2)] += w
        self.edge[(n2, n1)] += w
        self.degree[n1] += w
        self.degree[n2] += w


class SocialGraph:

    def __init__(self):
        self.user = dict()  # user dictionary, user_name --> user_id
        self.item = dict()  # item dictionary, item_name --> item_id
        self.user_reverse = dict()  # user reverse dictionary, user_id --> user_name
        self.item_reverse = dict()  # item reverse dictionary, item_id --> item_name
        self.node_u = dict()  # user node dictionary, user_id --> item_id_list (connected to the user)
        self.node_i = dict()  # item node dictionary, item_id --> user_id_list (connected to the item)
        self.edge = dict()   # edge dictionary, (user_id, item_id) --> edge weight
        self.review = dict()  # review dictionary, (user_id, item_id) --> review content
        self.rating = dict()  # rating dictionary, (user_id, item_id) --> rating
        self.label = dict()   # label dictionary, (user_id, item_id) --> label

    def name_to_id(self, user_name, item_name):
        user_id = [self.user[i] for i in user_name]
        item_id = [self.item[i] for i in item_name]
        return user_id, item_id

    def build(self, filename=None, data=None):
        # Stage 1. Load Data
        if data is None:
            data = pd.read_csv(filename)  # load social review data set

        try:  # load user_name, different data sets may have different column name for user_name
            users = data.user_id.get_values().astype('str')
        except:
            users = data.reviewerID.get_values()
        users = [i for i in users]

        try:  # load item_name
            items = data.prod_id.get_values().astype('str')
        except:
            try:
                items = data.restaurantID.get_values()
            except:
                items = data.hotelID.get_values()
        items = [i for i in items]

        try:  # load reviews
            reviews = data.review.get_values()
        except:
            reviews = data.reviewContent.get_values()

        try:  # load label
            labels = data.label.get_values()
            for i, l in enumerate(labels):
                if l == '1' or l == 1:
                    labels[i] = 0
                else:
                    labels[i] = 1
        except:
            labels = data.flagged.get_values()
            for i, l in enumerate(labels):
                if l == 'N' or l == 'NR':
                    labels[i] = 0
                else:
                    labels[i] = 1
        labels = [int(float(i)) for i in labels]

        ratings = data.rating.get_values()  # load rating
        ratings = [int(float(i)) for i in ratings]

        # Stage 2. Preprocess Data
        reviews = [clean_str(str(i)) for i in reviews]  # clean review

        index = 0
        for i, user in enumerate(users):  # user_name --> user_id
            if user not in self.user:
                self.user[user] = index
                self.user_reverse[index] = user
                self.node_u[index] = list()
                index += 1

        index = 0
        for i, item in enumerate(items):  # item_name --> item_id
            if item not in self.item:
                self.item[item] = index
                self.item_reverse[index] = item
                self.node_i[index] = list()
                index += 1

        # Stage 3. Build Graph
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            user_id = self.user[user]
            item_id = self.item[item]
            self.node_u[user_id].append(item_id)
            self.node_i[item_id].append(user_id)
            if (user_id, item_id) not in self.edge:
                self.edge[(user_id, item_id)] = 0
            self.review[(user_id, item_id)] = reviews[i]
            self.rating[(user_id, item_id)] = ratings[i]
            self.label[(user_id, item_id)] = labels[i]
            self.edge[(user_id, item_id)] += 1

    def split(self):
        user_graph = Graph()
        item_graph = Graph()

        # Generate user graph
        print('User graph generating...')
        user_triplets = list()
        for user in tqdm(self.node_u):
            items = self.node_u[user]
            for item in items:
                users = self.node_i[item]
                for u in users:
                    user_triplets.append((user, u, 1))
        user_graph.build(user_triplets)
        print('Finished!')

        # Generate item graph
        print('Item graph generating...')
        item_triplets = list()
        for item in tqdm(self.node_i):
            users = self.node_i[item]
            for user in users:
                items = self.node_u[user]
                for i in items:
                    item_triplets.append((item, i, 1))
        item_graph.build(item_triplets)
        print('Finished!')

        return user_graph, item_graph

# # For test
# graph = SocialGraph()
# graph.build('/data/qli1/Experiment/Qian/Fraud Detection/Yelp Shibuti Datasets/Data/yelp_Zip_data.csv')
# user_graph, item_graph = graph.split()