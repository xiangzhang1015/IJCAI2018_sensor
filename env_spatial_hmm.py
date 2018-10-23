"""
This is the environment of the reinforcement learning.
This environment loads  the dataset and returns the attention bar indices.
This environment defines the bar action strategy.
"""

import numpy as np
import math
import pickle
import Tkinter as tk
import tensorflow as tf
import scipy.io as sc
import numpy as np
from hmmlearn import hmm
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from statsmodels.tsa.ar_model import AR
# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class Env(object):
    viewer = None

    def __init__(self, len_max=128, n_fe=64, n_classes=8):
        # we only have one attention bar, therefore, only have 1*2 table. [I,L]
        # I denotes Initial point, L denotes length
        self.bar = np.zeros(1, dtype=[('start', np.float32), ('end', np.float32), ('length', np.float32)])
        self.bar['start'] = 0        # the initial point of attention bar
        self.bar['length'] = 64    # the length of attention bar
        self.bar['end'] = self.bar['start'] + self.bar['length']
        self.n_actions = 4
        self.n_features = 2
        self.len_max = len_max
        self.r = 0
        self.n_fe = n_fe
        self.n_classes = n_classes
        self.min_length = 10

        ################################################  scratch
        ## Unimib SHAR, only the first 20 samples of 3 acceration data.
        # unimib SHAR, (smartphone for AR) 11771 samples, 453 features, the 454th column is activity ID
        # 17labels (1-17), n_classes =18
        feature = sc.loadmat("/home/xiangzhang/matlabwork/unimib.mat")
        all = feature['unimib']
        all = np.array(all)
        feature_ = all[0:10000, 0:453]
        feature_ = np.reshape(feature_, [10000, 3, 151])
        n_ = 20
        feature_ = feature_[:, :, 0:n_]
        feature_ = feature_.reshape([10000, 3 * n_])
        print feature_.shape, feature_[0:1]

        all = np.hstack((feature_, all[0:10000, 453:454]))
        np.random.shuffle(all)

        feature_raw = all[:, 0:self.n_fe]
        label = all[:, -1]
        print all.shape

        ## shuffle the indices and the features
        len_max = len_max
        repeat_time = len_max/self.n_fe +1
        indices_ = list(xrange(self.n_fe))
        indices_repeat = np.repeat(indices_, repeat_time)
        np.random.shuffle(indices_repeat)
        self.indices_shuffed = indices_repeat[0:len_max]
        print "shuffled indices", self.indices_shuffed

        pickle.dump(self.indices_shuffed, open("indices_shuffed", "wb"))

        feature_ = np.transpose(feature_raw)

        fea_shuffled =[]
        for i in self.indices_shuffed:
            fea_shuffled += [list(feature_[i])]

        # print fea_shuffled.shape
        fea_shuffled = np.array(fea_shuffled)
        fea_shuffled = np.transpose(fea_shuffled)
        print fea_shuffled.shape
        # self.all = all
        self.fea_shuffled = fea_shuffled
        # self.label_ = all[28000:28000 + 2800, self.n_fe:self.n_fe + 1]

        # for the EEG ID dataset
        self.label_ = label

    def clip(self, dd):
        return np.clip(dd, a_min=0, a_max=self.len_max - 1)

    def step(self, action, step, episode):
        done = False

        start = self.bar['start']
        length = self.bar['length']
        end = self.bar['end']
        # self.step = step
        # self.episode = episode
        # print start, end, length

        ## The RNN_mytemplate.py deep learning algorithm, give out the acc
        # input start, end,  output the acc
        # calculate the reward
#################################################################################################

        feature_ = self.fea_shuffled[:, start:end + 1]

        print "AR start", self.label_.shape,
        start_time = time.clock()
        self.label_ = self.label_.reshape([self.label_.shape[0], 1])
        print feature_.shape, self.label_.shape
        # feature_ = np.hstack((feature_, self.label_))

        arp = []
        for i in range(feature_.shape[0]):
            AR_model = AR(feature_[i]).fit()
            p = AR_model.params
            arp.append(p)


        coe = np.array(arp)

        from sklearn import metrics
        ar_time = time.clock()
        # AR time: 177.5
        print "AR time", ar_time-start_time
        Score_all = metrics.silhouette_score(coe[0:10000], self.label_[0:10000])


        end_time = time.clock()
        print "score time,Total time", end_time-ar_time, end_time-start_time

#################################################################################################################
        # Silhouette Coefficient score ranges from [-1,1], Score_all+1 range [0,2]
        self.r = math.exp(Score_all+1) / (math.exp(2) - 1) - 0.5*(self.bar['start'] - self.bar['end']) / self.len_max
        # done and reward
        if Score_all+1 > 1.5:
            done = True


        # To caculate the next state from the current state
        if action == 0:  # left shift, inital left move, length is fixed
            # start point shift to left, the distance is random int in [1,10]
            self.bar['start'] = self.bar['start'] - np.random.randint(1, high=5, size=1)
            self.bar['start'] = self.clip(self.bar['start'])
            # the length is fixed
            self.bar['length'] = self.bar['length']
            # calculate the end and clip it.
            self.bar['end'] = self.bar['start'] + self.bar['length']
            self.bar['end'] = self.clip(self.bar['end'])
        elif action == 1:
            # start point shift to right, the distance is random int in [1,10]
            self.bar['start'] = self.bar['start'] + np.random.randint(1, high=5, size=1)
            self.bar['start'] = self.clip(self.bar['start'])
            # length keeps fix
            self.bar['length'] = self.bar['length']
            # calculate the end and clip it.
            self.bar['end'] = self.bar['start'] + self.bar['length']
            self.bar['end'] = self.clip(self.bar['end'])
        elif action == 2:  # condense, make the attention bar shorter
            len_decrement = np.random.randint(1, high=5, size=1)
            # start point shift to right, the distance is len_decrement
            self.bar['start'] = self.bar['start'] + len_decrement
            self.bar['start'] = self.clip(self.bar['start'])
            # the end shift to left, the distance is len_decrement,  and clip it.
            self.bar['end'] = self.bar['end'] - len_decrement
            self.bar['end'] = self.clip(self.bar['end'])

            # calculate the length
            self.bar['length'] = self.bar['end'] - self.bar['start']

        elif action == 3: # extend, make the attention bar longer
            len_increment = np.random.randint(1, high=5, size=1)
            # start point shift to left, the distance is len_increment
            self.bar['start'] = self.bar['start'] - len_increment
            self.bar['start'] = self.clip(self.bar['start'])
            # the end shift to left, the distance is len_increment,  and clip it.
            self.bar['end'] = self.bar['end'] + len_increment
            self.bar['end'] = self.clip(self.bar['end'])
            # calculate the length
            self.bar['length'] = self.bar['end'] - self.bar['start']
        # make sure the length is longer than five,
        # if the length is less than 5, make the start point shift to left
        if self.bar['end'] - self.bar['start'] < self.min_length:
            self.bar['start'] = self.bar['end'] - self.min_length
        self.bar['start'] = self.clip(self.bar['start'])
        # self.arm_info['length'] = self.arm_info['length'] +1  # the length = end-start +1.THis is the actual length
        return self.bar, self.r, done, Score_all, self.indices_shuffed


    def reset(self):
        # self.arm_info['start'] = np.random.randint(0, high=self.no_fea-self.min_length, size=1)        # the initial point of attention bar
        # self.arm_info['length'] = np.random.randint(self.min_length, high=self.no_fea, size=1)    # the length of attention bar
        self.bar['start'] = (self.len_max - self.n_fe) / 2
        self.bar['length'] = self.n_fe
        # self.bar['length'] = self.n_fe+6  # for short features, e.g., only 3 dimensions

        self.bar['end'] = self.bar['start'] + self.bar['length']
        self.bar['start'] = self.clip(self.bar['start'])
        self.bar['end'] = self.clip(self.bar['end'])
        # if self.arm_info['end'] - self.arm_info['start'] < self.min_length:
        #     self.arm_info['start'] = self.arm_info['end'] - self.min_length
        # self.arm_info['start'] = self.clip(self.arm_info['start'])
        return self.bar


    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians

