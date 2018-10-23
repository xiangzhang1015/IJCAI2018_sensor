"""
This is the WAS-LSTM classifier which receives the attention bar indices, loads dataset, and return the classification results.
"""
import tensorflow as tf
import scipy.io as sc
import numpy as np
import random
import pickle
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from sklearn.metrics import classification_report


def one_hot(y_):
    # this function is used to transfer one column label to one hot label
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

indices = pickle.load( open("/home/xiangzhang/scratch/indices_shuffled_unimib", "rb" ) )

indices = np.array(indices)
print indices, indices.shape

# unimib SHAR, (smartphone for AR) 11771 samples, 453 features, the 454th column is activity ID
# 17labels (1-17), n_classes =18
feature = sc.loadmat("/home/xiangzhang/scratch/unimib.mat")
all = feature['unimib']
all = np.hstack((all[0:10000, 0:453], all[0:10000, 453:454]))
np.random.shuffle(all)
n_classes = 18

#####################################Select the bar#########3
s_ = 247
e_ = 788
bar = indices[s_:e_+1]
attened_feature = []
for i in range(len(bar)):
    x_ = all[:,bar[i]]
    attened_feature.append(x_)
attened_feature = np.array(attened_feature)
attened_feature = np.transpose(attened_feature)
print "attended feature", attened_feature.shape, len(bar)

a_ = attened_feature
nof_ = all.shape[-1]
all = np.hstack((a_, all[:, nof_-1:nof_]))

# data batach seperation
np.random.shuffle(all)
data_size = all.shape[0]
no_fea = all.shape[1]-1
feature_all = all[:, 0:no_fea]
print all[:, -1]

# z-score scaling
feature_normalized=preprocessing.scale(feature_all)

label_all = all[:, no_fea:no_fea + 1]
all = np.hstack((feature_normalized, label_all))
print all.shape

# use the first subject as testing subject
train_data = all[0:data_size*0.9]  # 1 million samples
test_data = all[data_size*0.9:data_size]
n_steps = 1

feature_training = train_data[:,0:no_fea]
feature_training = feature_training.reshape([data_size*0.9, no_fea/n_steps, n_steps])

feature_testing = test_data[:,0:no_fea]
feature_testing = feature_testing.reshape([data_size*0.1, no_fea/n_steps, n_steps])

label_training = train_data[:,no_fea]
label_training = one_hot(label_training)
label_testing = test_data[:,no_fea]
label_testing = one_hot(label_testing)

print all.shape


# Manually batch split
a = feature_training
b = feature_testing
nodes = 164
lambda_ = 0.001
lr = 0.001
fg = 0.3

batch_size=int(data_size*0.1)
train_fea=[]
n_group=9
for i in range(n_group):
    f = a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f = label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)


# hyper-parameters
# n_inputs = no_fea/n_steps  # MNIST data input (img shape: 11*99)
n_inputs = 1
n_steps = no_fea  # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units = nodes
n_classes = n_classes   # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="x")
y = tf.placeholder(tf.float32, [None, n_classes])
# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
    'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
    'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
    'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
    'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
    'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
    'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units])),
    'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
    'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True)
}


def RNN(X, weights, biases):
    # transpose the inputs shape from
    X = tf.reshape(X, [-1, n_inputs])
    # into hidden
    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])
    X_hidd2 = tf.sigmoid(tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2'])
    # X_hidd3 = tf.sigmoid(tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3'])
    # X_hidd4 = tf.sigmoid(tf.matmul(X_hidd3, weights['hidd4']) + biases['hidd4'])
    X_hidd2=tf.nn.dropout(X_hidd2,0.5)
    X_in = tf.reshape(X_hidd2, [-1, n_steps, n_hidden4_units])
    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=fg, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=fg, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul((outputs[-1]+outputs[-2])/2, weights['out']) + biases['out']
    return results, outputs[-1]

pred, Feature = RNN(x, weights, biases)
l2 = lambda_ * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# L2 loss prevents this overkill neural network to over-fitting the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+l2  # Softmax loss

# tf.scalar_summary('loss', cost)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
pred_result = tf.argmax(pred, 1,name="pred_result")
label_true = tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    step = 0
    start = time.clock()
    acc_his = []
    while step < 1500:  # 1500 iterations
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
                })
        if sess.run(accuracy, feed_dict={x: b, y: label_testing}) > 0.96:
            print(
                "The lambda is :", lambda_, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
                sess.run(accuracy, feed_dict={
                    x: b,
                    y: label_testing,
                })
            )
            break
        if step % 10 == 0:
            pp = sess.run(pred_result, feed_dict={x: b, y: label_testing})
            # print "predict",pp[0:10]
            gt = np.argmax(label_testing, 1)
            # print "groundtruth", gt[0:10]
            hh = sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            })
            h2 = sess.run(accuracy,  feed_dict={x: train_fea[i], y: train_label[i]})
            # print "training acc", h2
            print("The lambda is :", lambda_, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is:", hh)

            print("The cost is :", sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))
            acc_his.append(hh)
        step += 1
    endtime = time.clock()

    print "run time:, max acc", endtime-start, max(acc_his)
    # save the model
    # save_path = saver.save(sess, "results/eeg_type_model" + str(lambda_) + str(
    #     lr) +str(nodes)+str(n_group)+ ".ckpt")
    # saver = tf.train.Saver()
    # save_path=saver.save(sess, "results/eeg_type_new_model")
    # print("save to path", save_path)




