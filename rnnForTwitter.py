'''
	An RNN with LSTM for the Kaggle competition "Real or Not: NLP with Disaster Tweets"
'''

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Necessary fix in order to use the v1 TensorFlow functionality 
tf.disable_v2_behavior()

# Debug flag (might move this later)
debug = False

'''
	Shuffle the data in each batch
'''
def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(X.shape[0]) # Use shape[0] for a length since we're dealing with Numpy arrays 
	n_batches = X.shape[0] / batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch

'''
	Pick up the test and train data-sets and transfrom the words
	to vector using a bag-of-words approach
'''
df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')

if debug:
	print(df_train.shape)
	print(df_train.head(60))

# Convert words in each Tweet to a vector
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(df_train["text"][:])
test_vectors = count_vectorizer.transform(df_test["text"][:])

# Outputs are sparse vectors, so convert them to dense vectors
train_vectors_dense = [t.todense() for t in train_vectors]
test_vectors_desne = [t.todense() for t in test_vectors]

# Split the training set into separate training and validation sets
size = 6000
X_train, X_val, y_train, y_val = np.asarray(train_vectors_dense[:size]), np.asarray(train_vectors_dense[size:]), np.asarray(df_train["target"][:size]), np.array(df_train["target"][size:])

if debug:
	print(X_train)
	print(y_train)

'''
	Set up the RNN with LSTM
'''

# Basic RNN settings
n_steps = 1 # Vertical length for instance, since we have vertorized a single tweet at a time
n_inputs = 21637 # Length of each vertorized instance
n_neurons = 10
n_outputs = 2 # Binary classification: disaster(1) or not (0)
n_layers = 100

# New types and shapes for TensorFlow compatitibility
X_train = X_train.astype(np.float32).reshape(-1, n_steps, n_inputs) 
X_val = X_val.astype(np.float32).reshape(-1,  n_steps, n_inputs)
y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)

learning_rate = 0.005

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="vertorized_tweets")
y = tf.placeholder(tf.int32, [None], name="disaster_or_not")

# LSTM cells, with Dropout regularization implemented in each layer (20 % of nodes are randomly dropped)
lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons), input_keep_prob=0.8, output_keep_prob=0.8) for layer in range(n_layers)]

# Put the layers together to make the deep RNN
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
batch_size = 100

with tf.Session() as sess:
	init.run() # Init node initializes all of the variables
	
	for epoch in range(n_epochs): # A single epoch is a complete iteration through the data	
		# Iterate through the number of mini-batches at each epoch
		for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
			X_batch = X_batch.reshape((-1, n_steps, n_inputs))
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) # Run the training operation

		# At the end of each epoch, evaluate the model on the last mini-batch and the validation set
		acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
		print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

	# Save the TensorFlow session
	save_path = saver.save(sess, "./RNN_LSTM_Twitter_Model.ckpt")
