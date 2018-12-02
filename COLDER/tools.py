"""
This code contains tools used in the experiments

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-12-2
"""

import pandas as pd
from graph import clean_str
import re


def split_train_test_data(file_name, train_begin_date, train_end_date, test_begin_date, test_end_date):
    data = pd.read_csv(file_name)
    if 'user_id' in data:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)  # sort data by date
        review_date = data['date'].get_values()  # extract review date
        users = data.user_id.get_values().astype('str')
    else:
        data_re = re.compile(r'\d+\/\d+\/\+')
        data['date'] = data['date'].apply(lambda x: data_re.search(x).group())
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)  # sort data by date
        review_date = data['date'].get_values()  # extract review date
        users = data.reviewerID.get_values()
    review_user = [i for i in users]
    # Generate cold start indicator
    user_list = dict()
    cold_start_indicator = list()
    for i in range(len(review_user)):
        if review_user[i] not in user_list:
            user_list[review_user[i]] = review_date[i]
            cold_start_indicator.append(True)
        else:
            cold_start_indicator.append(False)
    data['cold_start_indicator'] = cold_start_indicator
    train_data = data.loc[(data.date>= train_begin_date) & (data.date<=train_end_date)].reset_index(drop=True)
    test_data = data.loc[(data.date>=test_begin_date) & (data.date<=test_end_date)].reset_index(drop=True)
    return train_data, test_data


def generate_test_samples(data, cold_start=False):
    if cold_start:
        data = data.loc[data['cold_start_indicator'] == True,:].reset_index(drop=True)
    test_data = dict()
    if 'user_id' in data:
        test_data['user'] = data.user_id.get_values()
        test_data['item'] = data.prod_id.get_values()
        test_data['review'] = data.review.get_values()
        test_data['label'] = data.label.get_values()
        for i, l in enumerate(test_data['labels']):
            if l == '1' or l == 1:
                test_data['labels'][i] = 0
            else:
                test_data['labels'][i] = 1
    else:
        test_data['user'] = data.reviewerID.get_values()
        if 'hotelID' in data:
            test_data['item'] = data.hotelID.get_values()
        else:
            test_data['item'] = data.restaurantID.get_values()
        test_data['review'] = data.reviewContent.get_values()
        test_data['labels'] = data.flagged.get_values()
        for i, l in enumerate(test_data['labels']):
            if l == 'N' or l == 'NR':
                test_data['labels'][i] = 0
            else:
                test_data['labels'][i] = 1
    test_data['rating'] = data.rating.get_values()
    test_data['user'] = [i for i in test_data['user']]
    test_data['item'] = [i for i in test_data['item']]
    test_data['review'] = [clean_str(str(i)) for i in test_data['review']]
    test_data['rating'] = [int(float(i)) for i in test_data['rating']]
    test_data['label'] = [int(float(i)) for i in test_data['label']]
    return test_data




    