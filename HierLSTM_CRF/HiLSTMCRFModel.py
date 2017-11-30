# Implementing an RNN in TensorFlow
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return (text_string)
# Set RNN parameters
epochs = 200
batch_size = 250
max_sequence_length = 60
rnn_size = 50
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
nclass = 9
dropout_keep_prob = tf.placeholder(tf.float32)

# Download or open data
data_dir = 'data'
# data_file = 'text_data.txt'
data_file = 'LSTM.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Open data from text file
text_data = []
with open(os.path.join(data_dir, data_file), 'r') as file_conn:
    for row in file_conn:
        text_data.append(row)
text_data = text_data[: -1]
text_data = [x.split('\t') for x in text_data if len(x) >= 1]
# 标注类别
text_data_target = []
# 训练数据
text_data_train = []
for xi, x in enumerate(text_data):
    if (len(text_data[xi]) >= 2):
        text_data_target.append(x[0])
        text_data_train.append(x[1])
# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]
seq_len = []
for x in text_data_train:
    seq_len.append(min(max_sequence_length, len(x.split())))
# prepare traindata end
# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
# Shuffle and split data
text_processed = np.array(text_processed)
data_array = []
for x in text_data_target:
     if x=='Dongzuo':
         data_array.append(0)
     elif x=='Xinli':
         data_array.append(1)
     elif x=='Shuqing':
         data_array.append(2)
     elif x == 'Waimao':
         data_array.append(3)
     elif x == 'Shentai':
         data_array.append(4)
     elif x == 'Huanjing':
         data_array.append(5)
     elif x == 'Changjing':
         data_array.append(6)
     elif x =='Yuyan':
         data_array.append(7)
     elif x == 'None':
         data_array.append(8)
text_data_target = np.array(data_array)
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]
seq_shuffled = np.array(seq_len)[shuffled_ix]
# Split train/test set
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
seq_train, seq_test = seq_shuffled[:ix_cutoff], seq_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
#Tensorflow
# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])
seqlen = tf.placeholder(tf.int32, [None])
# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
# Define the LSTM cell
with tf.variable_scope('wordNetWork'):
    with tf.name_scope('cell_1'):
          cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)
          cell_1 = tf.contrib.rnn.MultiRNNCell([cell_1] * 2)
    output, state = tf.nn.dynamic_rnn(cell_1, embedding_output, dtype=tf.float32)
    output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of LSTM sequence
batch_in_size = tf.shape(output)[0]
# Start indices for each sample
index = tf.range(0, batch_in_size) * max_sequence_length + (seqlen - 1)
# Indexing
output_final = tf.gather(tf.reshape(output, [-1, rnn_size]), index)

utt_inputs = []
for i in range(1000):
    utt_input = tf.slice(output_final, begin=[i, 0], size=[5, rnn_size])
    utt_inputs.append(utt_input)
# shape: [self.args.batchSize+self.args.uttContextSize, self.args.uttContextSize, self.args.wordUnits]
last_input = tf.stack(utt_inputs)

with tf.variable_scope('contextNetWork'):
    with tf.name_scope('cell_2'):
           cell2 = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)
           cell2 = tf.contrib.rnn.MultiRNNCell([cell2] * 2)
    lastoutput, laststate = tf.nn.dynamic_rnn(cell2, last_input, dtype=tf.float32)
    lastoutput = tf.nn.dropout(lastoutput, dropout_keep_prob)

weight = tf.Variable(tf.truncated_normal([rnn_size, nclass], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[nclass]))
logits_out = tf.nn.softmax(tf.matmul(output_final, weight) + bias)
# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out,
                                                        labels=y_output)  # logits=float32, labels=int32
loss = tf.reduce_mean(losses)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# Start a graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Start training
for epoch in range(epochs):
    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    seq_train = seq_train[shuffled_ix]
    num_batches = int(len(x_train) / batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        seq_train_batch = seq_train[min_ix:max_ix]
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, seqlen: seq_train_batch, dropout_keep_prob: 0.5}
        sess.run(train_step, feed_dict=train_dict)
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_accuracy.append(temp_train_acc)
    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, seqlen: seq_test, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))
epoch_seq = np.arange(1, epochs + 1)

