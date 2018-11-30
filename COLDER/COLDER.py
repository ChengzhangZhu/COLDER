"""
This code define the COLDER class.
It includes the COLDER structure, COLDER training, and COLDER prediction.

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-28
"""
from keras.layers import Input, Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import unit_norm


class UnitNorm(Layer):
    def __init__(self, **kwargs):
        super(UnitNorm, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **kwargs):
        return x*1.0/(K.epsilon() + K.sqrt(K.sum(K.square(x), axis = -1, keepdims=True)))


class BehaviorSuccessLoss(Layer):
    def __init__(self, **kwargs):
        super(BehaviorSuccessLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def call(self, inputs, **kwargs):
        user = inputs[0]
        item = inputs[1]
        review = inputs[2]
        rating = inputs[3]
        label = inputs[4]
        b = user + item + review + rating #Eq (2)
        s = 2*1.0/(1 + K.clip(K.exp(-K.l2_normalize(b)), K.epsilon(), 1)) - 1
        loss = -label*K.log(s)
        return loss


class SocialRelationLoss(Layer):
    def __init__(self, **kwargs):
        super(SocialRelationLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def call(self, inputs, **kwargs):
        object_1 = inputs[0]
        object_2 = inputs[1]
        label = inputs[2]
        similarity = K.sum(object_1*object_2)
        s = 2*1.0/(1 + K.clip(K.exp(-similarity), K.epsilon(), 1)) - 1
        loss = -label*K.log(s)
        return loss


class FraudDetectionLoss(Layer):
    def __init__(self, **kwargs):
        super(FraudDetectionLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        y_true = inputs[0]
        return K.binary_crossentropy(y_true, y_pred)
    

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
    def __init__(self, dim=100, fraud_detector_nodes=None, max_len=200, filter_size=2, num_filters=100, pre_word_embedding_dim=100, pre_word_embedding_file='glove.6B.100d.txt'):
        self.inputs = None  # the list of inputs
        self.fraud_detector = None  # the fraud detector network
        self.user_tokenizer = None  # user tokenizer
        self.item_tokenizer = None  # item tokenizer
        self.review_tokenizer = None  # review tokenizer
        self.dim = dim  # the embedding dimension
        self.max_len = max_len  # the max length of a sentence
        self.num_filters = num_filters  # the number of filters in ConvNet in review embedding
        self.filter_size = filter_size  # the filter size in ConvNet in review embedding
        self.rating_embedding_model = None
        self.user_embedding_model = None
        self.item_embedding_model = None
        self.review_embedding_model = None
        self.joint_model = None  # the joint training model
        self.pre_word_embedding_dim = pre_word_embedding_dim  # the dimension of pre-trained word embedding
        self.pre_word_embedding_file = pre_word_embedding_file  # the pre-trained word embedding file
        if fraud_detector_nodes is None:
            fraud_detector_nodes = [100, 100]
        self.fraud_detector_nodes = fraud_detector_nodes  # the layer structure and nodes in the fraud detector

    def build_model(self, rating_input_dim = None, user_input_dim = None, item_input_dim = None):
        if rating_input_dim is None:
            rating_input_dim = 5
        if user_input_dim is None:
            user_input_dim = len(self.user_tokenizer.word_counts)
        if item_input_dim is None:
            item_input_dim = len(self.item_tokenizer.word_counts)

        # Rating Embedding
        rating_input = Input(shape=(1,), dtype='int32')
        rating_embedding = Embedding(input_dim=rating_input_dim+1, output_dim=self.dim, embeddings_constraint=unit_norm(), name='Rating_Embedding')(rating_input)
        self.rating_embedding_model = Model(inputs=rating_input, outputs=rating_embedding)

        # User Embedding
        user_input = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(input_dim=user_input_dim+1, output_dim=self.dim, embeddings_constraint=unit_norm(), name='User_Embedding')(user_input)
        self.user_embedding_model = Model(inputs=user_input, outputs=user_embedding)

        # Item Embedding
        item_input = Input(shape=(1,), dtype='int32')
        item_embedding = Embedding(input_dim=item_input_dim+1, output_dim=self.dim, embeddings_constraint=unit_norm(), name='Item_Embedding')(item_input)
        self.item_embedding_model = Model(inputs=item_input, outputs=item_embedding)

        # Review Embedding
        word_index = self.review_tokenizer.word_index
        # # Load pre-trained word embedding
        embeddings_index = {}
        f = open(self.pre_word_embedding_file)
        for line in f:
            values = line.split()
            word = values[0]
            embed_value = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embed_value
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))
        embedding_matrix = np.random.random((len(word_index) + 1, self.pre_word_embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        # # Build review embedding layers
        review_input = Input(shape=(self.max_len,), dtype='int32')
        word_embedding = Embedding(len(word_index)+1,
                                   self.pre_word_embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=self.max_len,
                                   name='Word_Embedding')(review_input)
        review_embedding = Conv1D(self.num_filters,
                                  kernel_size=self.filter_size,
                                  strides=1,
                                  activation='tanh',
                                  name='Conv_Layer')(word_embedding)
        review_embedding = MaxPooling1D(pool_size=int(review_embedding.shape[1]),
                                        strides=None,
                                        padding='valid')(review_embedding)
        review_embedding = Flatten()(review_embedding)
        review_embedding = UnitNorm(name='Review_Embedding')(review_embedding)
        self.review_embedding_model = Model(inputs=review_input, outputs=review_embedding)

        # Embedding Calculation:
        # <user_1, item_1, review_1, rating_1> and <user_2, item_2, review_2, rating_2>
        user_input_1 = Input(shape=(1,), dtype='int32')
        user_input_2 = Input(shape=(1,), dtype='int32')
        item_input_1 = Input(shape=(1,), dtype='int32')
        item_input_2 = Input(shape=(1,), dtype='int32')
        review_input_1 = Input(shape=(self.max_len,), dtype='int32')
        review_input_2 = Input(shape=(self.max_len,), dtype='int32')
        rating_input_1 = Input(shape=(1,), dtype='int32')
        rating_input_2 = Input(shape=(1,), dtype='int32')
        user_1 = self.user_embedding_model(user_input_1)
        user_2 = self.user_embedding_model(user_input_2)
        item_1 = self.item_embedding_model(item_input_1)
        item_2 = self.item_embedding_model(item_input_2)
        review_1 = self.review_embedding_model(review_input_1)
        review_2 = self.review_embedding_model(review_input_2)
        rating_1 = self.rating_embedding_model(rating_input_1)
        rating_2 = self.rating_embedding_model(rating_input_2)

        # Fraud detector
        fraud_input = Input(shape=(4*self.dim,), name='fraud_detector_input')
        fraud_hidden_output = Dense(self.fraud_detector_nodes[0], activation='relu')(fraud_input)
        for i in range(len(self.fraud_detector_nodes)-1):
            fraud_hidden_output = Dense(self.fraud_detector_nodes[i+1], activation='relu')(fraud_hidden_output)
        fraud_output = Dense(2, activation='sigmoid', name='fraud_detector_output')(fraud_hidden_output)
        self.fraud_detector = Model(inputs=fraud_input,outputs=fraud_output)

        # Define Label Inputs
        user_context_input = Input(shape=(1,), dtype='int32', name='user_context_flag')
        item_context_input = Input(shape=(1,), dtype='int32', name='item_context_flag')
        fraud_label_input_1 = Input(shape=(1,), dtype='int32', name='fraud_label_input_1')
        fraud_label_input_2 = Input(shape=(1,), dtype='int32', name='fraud_label_input_2')
        behavior_success_input_1 = Input(shape=(1,), dtype='int32', name='behavior_success_flag_1')
        behavior_success_input_2 = Input(shape=(1,), dtype='int32', name='behavior_success_flag_2')

        # Calculate Loss Value
        behavior_success_loss_1 = BehaviorSuccessLoss()([user_1, item_1, review_1, rating_1, behavior_success_input_1])
        behavior_success_loss_2 = BehaviorSuccessLoss()([user_2, item_2, review_2, rating_2, behavior_success_input_2])
        user_social_relation_loss = SocialRelationLoss()([user_1, user_2, user_context_input])
        item_social_relation_loss = SocialRelationLoss()([item_1, item_2, item_context_input])
        fraud_detection_loss_1 = FraudDetectionLoss()([user_1, item_1, review_1, rating_1, fraud_label_input_1])
        fraud_detection_loss_2 = FraudDetectionLoss()([user_2, item_2, review_2, rating_2, fraud_label_input_2])
        loss = JointLoss()([fraud_detection_loss_1, fraud_detection_loss_2, behavior_success_loss_1, behavior_success_loss_2, user_social_relation_loss, item_social_relation_loss])
        self.joint_model = Model(inputs=[user_input_1, item_input_1,
                                         review_input_1, rating_input_1,
                                         fraud_label_input_1,
                                         user_context_input,
                                         behavior_success_input_1,
                                         user_input_2, item_input_2,
                                         review_input_2, rating_input_2,
                                         fraud_label_input_2,
                                         item_context_input,
                                         behavior_success_input_2],
                                 outputs=loss)
        self.joint_model.compile(optimizer='adam', loss=None)