"""
This code defines a social graph class, and is to build social graph from a social review data set.

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-22
"""

import pandas as pd
import re


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


class SocialGraph:

    def __init__(self):
        self.user = dict()  # user dictionary, user_name --> user_id
        self.item = dict()  # item dictionary, item_name --> item_id
        self.node_u = dict()  # user node dictionary, user_id --> item_id_list (connected to the user)
        self.node_i = dict()  # item node dictionary, item_id --> user_id_list (connected to the item)
        self.edge = dict()   # edge dictionary, (user_id, item_id) --> edge weight
        self.review = dict()  # review dictionary, (user_id, item_id) --> review content
        self.rating = dict()  # rating dictionary, (user_id, item_id) --> rating

    def build(self, filename):

        # Stage 1. Load Data
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

        ratings = data.rating.get_values()  # load rating
        ratings = [int(float(i)) for i in ratings]

        # Stage 2. Preprocess Data
        reviews = [clean_str(str(i)) for i in reviews]  # clean review

        index = 0
        for i, user in enumerate(users):  # user_name --> user_id
            if user not in self.user:
                self.user[user] = index
                self.node_u[index] = list()
                index += 1

        index = 0
        for i, item in enumerate(items):  # item_name --> item_id
            if item not in self.item:
                self.item[item] = index
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
            self.edge[(user_id, item_id)] += 1