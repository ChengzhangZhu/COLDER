"""
This code define the COLDER class.
It includes the COLDER structure, COLDER training, and COLDER prediction.

Author: Qian Li <linda.zhu.china@gmail.com>
Date: 2018-11-28
"""
from keras.layers import Input, Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import unit_norm
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from graph import SocialGraph
from tqdm import tqdm


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
        return (1, 1)

    def call(self, inputs, **kwargs):
        user = inputs[0]
        item = inputs[1]
        review = inputs[2]
        rating = inputs[3]
        label = inputs[4]
        b = user + item + review + rating #Eq (2)
        s = 2*1.0/(1 + K.clip(K.exp(-K.l2_normalize(b)), K.epsilon(), 1)) - 1
        loss = -label*K.log(s)
        return K.mean(loss)


class SocialRelationLoss(Layer):
    def __init__(self, **kwargs):
        super(SocialRelationLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1, 1)

    def call(self, inputs, **kwargs):
        object_1 = inputs[0]
        object_2 = inputs[1]
        label = inputs[2]
        similarity = K.sum(object_1*object_2)
        s = 2*1.0/(1 + K.clip(K.exp(-similarity), K.epsilon(), 1)) - 1
        loss = -label*K.log(s)
        return K.mean(loss)


class FraudDetectionLoss(Layer):
    def __init__(self, **kwargs):
        super(FraudDetectionLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1, 1)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        y_true = inputs[1]
        mask = inputs[2]
        loss = K.binary_crossentropy(y_true, y_pred)*mask # do not consider the negative samples in fraud detector
        return loss/(K.sum(mask) + K.epsilon())  # generate the mean loss of positive samples


class JointLoss(Layer):
    def __init__(self, **kwargs):
        super(JointLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1,1)

    def call(self, inputs, alpha=None, **kwargs):
        fraud_detection_loss_1 = inputs[0]
        fraud_detection_loss_2 = inputs[1]
        behavior_success_loss_1 = inputs[2]
        behavior_success_loss_2 = inputs[3]
        user_social_relation_loss = inputs[4]
        item_social_relation_loss = inputs[5]
        if alpha is None:
            alpha = np.ones(4)
        loss = alpha[0]*(fraud_detection_loss_1+fraud_detection_loss_2) + alpha[1]*(behavior_success_loss_1+behavior_success_loss_2) + alpha[2]*user_social_relation_loss + alpha[3]*item_social_relation_loss
        self.add_loss(loss, inputs=inputs)
        return loss


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
    def __init__(self, dim=100, fraud_detector_nodes=None, max_len=200, filter_size=2, num_filters=100, pre_word_embedding_dim=100, pre_word_embedding_file='glove.6B.100d.txt', max_num_words=100000):
        self.inputs = None  # the list of inputs
        self.fraud_detector = None  # the fraud detector network
        self.user_id = None  # processed user id
        self.item_id = None  # processed item id
        self.review_tokenizer = None  # review tokenizer
        self.dim = dim  # the embedding dimension
        self.max_len = max_len  # the max length of a sentence
        self.max_num_words = max_num_words  # the max number of words in reviews
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
        self.loss_history = list()  # record the training loss in each epoch

    def build_model(self, rating_input_dim = None, user_input_dim = None, item_input_dim = None):
        if rating_input_dim is None:
            rating_input_dim = 5
        if user_input_dim is None:
            user_input_dim = len(self.user_id)
        if item_input_dim is None:
            item_input_dim = len(self.item_id)

        # Rating Embedding
        rating_input = Input(shape=(1,), dtype='int32')
        rating_embedding = Embedding(input_dim=rating_input_dim+1, output_dim=self.dim, embeddings_constraint=unit_norm(), name='Rating_Embedding')(rating_input)
        rating_embedding = Flatten()(rating_embedding)
        self.rating_embedding_model = Model(inputs=rating_input, outputs=rating_embedding, name='rating_embedding_model')

        # User Embedding
        user_input = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(input_dim=user_input_dim+1, output_dim=self.dim, embeddings_constraint=unit_norm(), name='User_Embedding')(user_input)
        user_embedding = Flatten()(user_embedding)
        self.user_embedding_model = Model(inputs=user_input, outputs=user_embedding, name='user_embedding_model')

        # Item Embedding
        item_input = Input(shape=(1,), dtype='int32')
        item_embedding = Embedding(input_dim=item_input_dim+1, output_dim=self.dim, embeddings_constraint=unit_norm(), name='Item_Embedding')(item_input)
        item_embedding = Flatten()(item_embedding)
        self.item_embedding_model = Model(inputs=item_input, outputs=item_embedding, name='item_embedding_model')

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
        self.review_embedding_model = Model(inputs=review_input, outputs=review_embedding, name='review_embedding_model')

        # Embedding Calculation:
        # <user_1, item_1, review_1, rating_1> and <user_2, item_2, review_2, rating_2>
        user_input_1 = Input(shape=(1,), dtype='int32', name='user_input_1')
        user_input_2 = Input(shape=(1,), dtype='int32', name='user_input_2')
        item_input_1 = Input(shape=(1,), dtype='int32', name='item_input_1')
        item_input_2 = Input(shape=(1,), dtype='int32', name='item_input_2')
        review_input_1 = Input(shape=(self.max_len,), dtype='int32', name='review_input_1')
        review_input_2 = Input(shape=(self.max_len,), dtype='int32', name='review_input_2')
        rating_input_1 = Input(shape=(1,), dtype='int32', name='rating_input_1')
        rating_input_2 = Input(shape=(1,), dtype='int32', name='rating_input_2')
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
        fraud_output = Dense(1, activation='sigmoid', name='fraud_detector_output')(fraud_hidden_output)
        self.fraud_detector = Model(inputs=fraud_input,outputs=fraud_output, name='fraud_detector')

        # Define Label Inputs
        user_context_input = Input(shape=(1,), name='user_context_flag')
        item_context_input = Input(shape=(1,), name='item_context_flag')
        fraud_label_input_1 = Input(shape=(1,), name='fraud_label_input_1')
        fraud_label_input_2 = Input(shape=(1,), name='fraud_label_input_2')
        behavior_success_input_1 = Input(shape=(1,), name='behavior_success_flag_1')
        behavior_success_input_2 = Input(shape=(1,), name='behavior_success_flag_2')

        # Calculate Loss Value
        joint_features_1 = concatenate([user_1, item_1, review_1, rating_1])  #  concatenate embedding features as fraud detector's input
        joint_features_2 = concatenate([user_2, item_2, review_2, rating_2])
        fraud_prediction_1 = self.fraud_detector(joint_features_1)
        fraud_prediction_2 = self.fraud_detector(joint_features_2)
        behavior_success_loss_1 = BehaviorSuccessLoss()([user_1, item_1, review_1, rating_1, behavior_success_input_1])
        behavior_success_loss_2 = BehaviorSuccessLoss()([user_2, item_2, review_2, rating_2, behavior_success_input_2])
        user_social_relation_loss = SocialRelationLoss()([user_1, user_2, user_context_input])
        item_social_relation_loss = SocialRelationLoss()([item_1, item_2, item_context_input])
        fraud_detection_loss_1 = FraudDetectionLoss()([fraud_prediction_1, fraud_label_input_1, behavior_success_input_1])
        fraud_detection_loss_2 = FraudDetectionLoss()([fraud_prediction_2, fraud_label_input_2, behavior_success_input_2])
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

    def fit(self, data, g=SocialGraph(), epoch=5, batch_size=32):
        # preprocess data
        if self.user_id is None:
            self.user_id = np.asarray(g.node_u.keys())
        else:
            self.user_id = np.asarray(set(self.user_id + np.asarray(g.node_u.keys())))
        if self.item_id is None:
            self.item_id = np.asarray(g.node_i.keys())
        else:
            self.item_id = np.asarray(set(self.item_id + np.asarray(g.node_i.keys())))
        print('Review preprocessing...')
        if self.review_tokenizer is None:
            reviews, self.review_tokenizer = self.preprocess(g.review.values())
        else:
            reviews, self.review_tokenizer = self.preprocess(g.review.values(), token=self.review_tokenizer)
        g.review = dict(zip(g.review.keys(), reviews))

        # Build Model
        if self.joint_model is None:
            self.build_model()

        # Training Model
        num_train = len(data['user1'])
        for i in range(epoch):
            print('{}-th epoch begin:'.format(i))
            num_iters = num_train/batch_size if num_train%batch_size == 0 else int(num_train/batch_size) + 1
            iters = tqdm(range(num_iters))
            train_loss = []
            for j in iters:
                train_data = self.train_data_generator(data, g, j, batch_size, num_train)
                history = self.joint_model.fit(train_data, epoch=epoch, batch_size=batch_size)
                train_loss.append(history.history['loss'][-1])
                iters.set_description('Training loss: {:.4} >>>>'.format(history.history['loss'][-1]))
            self.loss_history.append(np.mean(train_loss))
            print('{}-th epoch ended, training loss {:.4}.'.format(i, np.mean(train_loss)))

    def train_data_generator(self, data, g, j, batch_size, num_train):
        if j*batch_size + batch_size < num_train:
            user1 = data['user1'][j*batch_size:j*batch_size+batch_size]
            user2 = data['user2'][j*batch_size:j*batch_size+batch_size]
            item1 = data['item1'][j*batch_size:j*batch_size+batch_size]
            item2 = data['item2'][j*batch_size:j*batch_size+batch_size]
            rating1 = data['rating1'][j*batch_size:j*batch_size+batch_size]
            rating2 = data['rating2'][j*batch_size:j*batch_size+batch_size]
            label1 = data['label1'][j*batch_size:j*batch_size+batch_size]
            label2 = data['label2'][j*batch_size:j*batch_size+batch_size]
            context_u = data['context_u'][j*batch_size:j*batch_size+batch_size]
            context_i = data['context_i'][j*batch_size:j*batch_size+batch_size]
            success1 = data['success1'][j*batch_size:j*batch_size+batch_size]
            success2 = data['success2'][j*batch_size:j*batch_size+batch_size]
            review_id_1 = data['review1'][j*batch_size:j*batch_size+batch_size]
            review_id_2 = data['review2'][j*batch_size:j*batch_size+batch_size]
        else:
            user1 = data['user1'][j * batch_size:]
            user2 = data['user2'][j * batch_size:]
            item1 = data['item1'][j * batch_size:]
            item2 = data['item2'][j * batch_size:]
            rating1 = data['rating1'][j * batch_size:]
            rating2 = data['rating2'][j * batch_size:]
            label1 = data['label1'][j * batch_size:]
            label2 = data['label2'][j * batch_size:]
            context_u = data['context_u'][j * batch_size:]
            context_i = data['context_i'][j * batch_size:]
            success1 = data['success1'][j * batch_size:]
            success2 = data['success2'][j * batch_size:]
            review_id_1 = data['review1'][j * batch_size:]
            review_id_2 = data['review2'][j * batch_size:]
        review1 = [g.review[i] for i in review_id_1]
        review2 = [g.review[i] for i in review_id_2]
        inputs = [user1, item1, review1, rating1, label1, context_u, success1,
                  user2, item2, review2, rating2, label2, context_i, success2]
        return inputs
    
    def preprocess(self, data, token=None):
        if token is None:
            corpus = []
            for i in data:
                corpus.append(i)
            print('Initializing review tokenizer...')
            tokenizer = Tokenizer(num_words=self.max_num_words)
            tokenizer.fit_on_texts(corpus)
            print('Review tokenizing finished...')
        else:
            tokenizer = token
        processed_data = np.zeros((len(data), self.max_len), dtype='int32')

        for j, paragraph in enumerate(data):
            word_token = text_to_word_sequence(paragraph)
            k = 0
            for ii, word in enumerate(word_token):
                if word in tokenizer.word_index:
                    if k < self.max_len and tokenizer.word_index[word]<self.max_num_words:
                        processed_data[j,k] = tokenizer.word_index[word]
                        k += 1
        return processed_data, tokenizer
