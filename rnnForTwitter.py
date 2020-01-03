'''
	An RNN with LSTM for the Kaggle competition "Real or Not: NLP with Disaster Tweets"
'''

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Necessary fix in order to use the v1 tensorflow functionality 
tf.disable_v2_behavior()


'''
	Shuffle the data in each batch
'''

def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) / batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch

debug = False

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')

if debug:
	print(df_train.shape)
	print(df_train.head(60))

# Convert words in each Tweet to a vector
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(df_train["text"])
test_vectors = count_vectorizer.transform(df_test)

# Split the training set into separate training and validation sets
X_train, X_val, y_train, y_val = train_vectors[:6000], train_vectors[6000:], df_train["target"][:6000], df_train["target"][6000:]

if debug:
	print(X_train)
	print(y_train)

'''
	Set up the RNN with LSTM
'''

n_steps = 1
n_inputs = 1
n_neurons = 100
n_outputs = 1
n_layers = 3

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]

# Deep RNN
multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

# Use dynamic unrolling through time
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

top_layer_h_state = states[-1][1]

# Use Softmax in the final layer, together with the cross-entropy loss
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

# Adam for optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialization and saving    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

'''
	Load a TensorFlow session to run the training
'''
n_epochs = 40
batch_size = 60

with tf.Session() as sess:
	init.run() # Init node initializes all of the variables
	
	for epoch in range(n_epochs): # A single epoch is a complete iteration through the data	
		# Iterate through the number of mini-batches at each epoch
		for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) # Run the training operation

		# At the end of each epoch, evaluate the model on the last mini-batch and the validation set
		acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
		print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

	# Save the TensorFlow session
	save_path = saver.save(sess, "./my_model_final.ckpt")


