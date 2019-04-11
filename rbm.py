import pandas as pd
import numpy as np
import tensorflow as tf
from utils import Util
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from IPython.display import display
class RBM(object):
    '''
    Class definition for a simple RBM
    '''
    def __init__(self, alpha, H, num_vis):

        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis # might face an error here, call preprocess if you do
        self.errors = []
        self.energy_train = []
        self.energy_valid = []

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose, filename):
        '''
        Function where RBM training takes place
        '''
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        print("Phase 1: Input Processing")
        _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
        # Gibb's Sampling
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
        print("Phase 2: Reconstruction")
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

        print("Creating the gradients")
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Calculate the Contrastive Divergence to maximize
        CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

        # Create methods to update the weights and biases
        update_w = W + self.alpha * CD
        update_vb = vb + self.alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = hb + self.alpha * tf.reduce_mean(h0 - h1, 0)

        # Set the error function, here we use Mean Absolute Error Function
        err = v0 - v1
        err_sum = tf.reduce_mean(err * err)

        # Initialize our Variables with Zeroes using Numpy Library
        # Current weight
        cur_w = np.zeros([self.num_vis, self.num_hid], np.float32)
        # Current visible unit biases
        cur_vb = np.zeros([self.num_vis], np.float32)

        # Current hidden unit biases
        cur_hb = np.zeros([self.num_hid], np.float32)

        # Previous weight
        prv_w = np.random.normal(loc=0, scale=0.01,
                                size=[self.num_vis, self.num_hid])
        # Previous visible unit biases
        prv_vb = np.zeros([self.num_vis], np.float32)

        # Previous hidden unit biases
        prv_hb = np.zeros([self.num_hid], np.float32)

        print("Running the session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")
        util = Util()
        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train[start:end]
                cur_w = sess.run(update_w, feed_dict={
                                 v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb

            if valid:
                etrain = np.mean(util.free_energy(train, cur_w, cur_vb, cur_hb))
                self.energy_train.append(etrain)
                evalid = np.mean(util.free_energy(valid, cur_w, cur_vb, cur_hb))
                self.energy_valid.append(evalid)
            self.errors.append(sess.run(err_sum, feed_dict={
                          v0: train, W: cur_w, vb: cur_vb, hb: cur_hb}))
            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
        if not os.path.exists('rbm_models'):
            os.mkdir('rbm_models')
        filename = 'rbm_models/'+filename
        if not os.path.exists(filename):
            os.mkdir(filename)
        np.save(filename+'/w.npy', prv_w)
        np.save(filename+'/vb.npy', prv_vb)
        np.save(filename+'/hb.npy',prv_hb)
        
        if free_energy:
            print("Exporting free energy plot")
            self.export_free_energy_plot(filename)
        print("Exporting errors vs epochs plot")
        self.export_errors_plot(filename)
        inputUser = [train[user]]
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        return rec, prv_w, prv_vb, prv_hb

    def load_predict(self, filename, train, user):
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])
        
        prv_w = np.load('rbm_models/'+filename+'/w.npy')
        prv_vb = np.load('rbm_models/'+filename+'/vb.npy')
        prv_hb = np.load('rbm_models/'+filename+'/hb.npy')
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        print("Model restored")
        
        inputUser = [train[user]]
        
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        
        return rec, prv_w, prv_vb, prv_hb
        
    def calculate_scores(self, ratings, attractions, rec, user):
        '''
        Function to obtain recommendation scores for a user
        using the trained weights
        '''
        # Creating recommendation score for books in our data
        ratings["Recommendation Score"] = rec[0]

        """ Recommend User what books he has not read yet """
        # Find the mock user's user_id from the data
#         cur_user_id = ratings[ratings['user_id']

        # Find all books the mock user has read before
        visited_places = ratings[ratings['user_id'] == user]['attraction_id']
        visited_places

        # converting the pandas series object into a list
        places_id = visited_places.tolist()

        # getting the book names and authors for the books already read by the user
        places_names = []
        places_categories = []
        places_prices = []
        for place in places_id:
            places_names.append(
                attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
            places_categories.append(
                attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
            places_prices.append(
                attractions[attractions['attraction_id'] == place]['price'].tolist()[0])

        # Find all books the mock user has 'not' read before using the to_read data
        unvisited = attractions[~attractions['attraction_id'].isin(places_id)]['attraction_id']
        unvisited_id = unvisited.tolist()
        
        # extract the ratings of all the unread books from ratings dataframe
        unseen_with_score = ratings[ratings['attraction_id'].isin(unvisited_id)]

        # grouping the unread data on book id and taking the mean of the recommendation scores for each book_id
        grouped_unseen = unseen_with_score.groupby('attraction_id', as_index=False)['Recommendation Score'].max()
        display(grouped_unseen.head())
        
        # getting the names and authors of the unread books
        unseen_places_names = []
        unseen_places_categories = []
        unseen_places_prices = []
        unseen_places_scores = []
        for place in grouped_unseen['attraction_id'].tolist():
            unseen_places_names.append(
                attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
            unseen_places_categories.append(
                attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
            unseen_places_prices.append(
                attractions[attractions['attraction_id'] == place]['price'].tolist()[0])
            unseen_places_scores.append(
                grouped_unseen[grouped_unseen['attraction_id'] == place]['Recommendation Score'].tolist()[0])

        # creating a data frame for unread books with their names, authors and recommendation scores
        unseen_places = pd.DataFrame({
            'att_id' : grouped_unseen['attraction_id'].tolist(),
            'att_name': unseen_places_names,
            'att_cat': unseen_places_categories,
            'att_price': unseen_places_prices,
            'score': unseen_places_scores
        })

        # creating a data frame for read books with the names and authors
        seen_places = pd.DataFrame({
            'att_id' : places_id,
            'att_name': places_names,
            'att_cat': places_categories,
            'att_price': places_prices
        })

        return unseen_places, seen_places

    def export(self, unseen, seen, filename, user):
        '''
        Function to export the final result for a user into csv format
        '''
        # sort the result in descending order of the recommendation score
        sorted_result = unseen.sort_values(
            by='score', ascending=False)
        
        x = sorted_result[['score']].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler((0,5))
        x_scaled = min_max_scaler.fit_transform(x)
        
        sorted_result['score'] = x_scaled
        
        # exporting the read and unread books  with scores to csv files

        seen.to_csv(filename+'/user'+user+'_seen.csv')
        sorted_result.to_csv(filename+'/user'+user+'_unseen.csv')
#         print('The attractions visited by the user are:')
#         print(seen)
#         print('The attractions recommended to the user are:')
#         print(sorted_result)

    def export_errors_plot(self, filename):
        plt.plot(self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(filename+"/error.png")

    def export_free_energy_plot(self, filename):
        fig, ax = plt.subplots()
        ax.plot(self.energy_train, label='train')
        ax.plot(self.energy_valid, label='valid')
        leg = ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Free Energy")
        plt.savefig(filename+"/free_energy.png")
