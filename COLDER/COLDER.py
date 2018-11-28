"""
This code define the COLDER class.
It includes the COLDER structure, COLDER training, and COLDER prediction.

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-28
"""
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np


class COLDER:
    """
    This class define a COLDER model
    """
    def __init__(self):
        self.uid = None  # the existing user id
        self.iid = None  # the existing item id
        self.classifier = None  # the fraud classifier
        self.estimator = None  # the new user embedding estimator


class Network:
    """
    This class define the network in COLDER model
    """
    def __init__(self):
        self.inputs = None  # the list of inputs
        self.fraud_detector = None  # the fraud detector network

    def build_fraud_detector(self, input_dim, num_nodes=None, activate_func='relu'):
        if num_nodes is None:
            num_nodes = [100, 100]
        fraud_input = Input(shape=(input_dim,), name='fraud_detector_input')
        x = Dense(num_nodes[0], activation=activate_func)(fraud_input)
        for i in range(len(num_nodes)-1):
            x = Dense(num_nodes[i+1], activation=activate_func)(x)
        output = Dense(2, activation='softmax', name='fraud_detector_output')(x)
        model = Model(inputs=fraud_input,outputs=output)
        self.fraud_detector = model

    def build_encoder(self, dim, activate_func='relu'):
        user_input = Embedding()
