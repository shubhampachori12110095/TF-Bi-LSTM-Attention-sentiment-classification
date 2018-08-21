import numpy as np
import tensorflow as tf
from keras.datasets import imdb  #IMDB Data
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn #Bi-rnn
from tqdm import tqdm #Python progress bar, users only need to wrap arbitrary iterators。

from attention import attention
from utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator

NUM_WORDS = 10000 # Consider only the 10,000 words that appear most often
INDEX_FROM = 3 
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 128
NUM_EPOCHS = 3 # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
MODEL_PATH = './model'


# DTAT（'~/.keras/datasets/'+path），load, https://s3.amazonaws.com/text-datasets/imdb.npz
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz", num_words=NUM_WORDS, index_from=INDEX_FROM)

# Sequences pre-processing
vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)
X_train = zero_pad(X_train, SEQUENCE_LENGTH) 
X_test = zero_pad(X_test, SEQUENCE_LENGTH)
print(X_train.shape) #(25000, 250)
print(X_test.shape) #(25000, 250)

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    tf.summary.histogram('embeddings_var', embeddings)
    batch_embedded = tf.nn.embedding_lookup(embeddings, batch_ph)

# Bi-LSTM layer
rnn_outputs, _ = bi_rnn(cell_fw=LSTMCell(HIDDEN_SIZE), cell_bw=LSTMCell(HIDDEN_SIZE),
                        inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
tf.summary.histogram('Bi-LSTM_outputs', rnn_outputs)

# Attention layer
with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    tf.summary.histogram('alphas', alphas)

# Dropout layer，avoid overfiting
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[1]))
    
    y_ = tf.nn.xw_plus_b(drop, W, b) # y=W*x+b
    y_ = tf.squeeze(y_) 
    tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    tf.summary.scalar('loss', loss)

    # AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # use a mini-batch size Training
            num_batches = X_train.shape[0] // BATCH_SIZE # Calculate how many batches (steps)
            # Verify model accuracy every num_batches step
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                               target_ph: y_batch,
                                                               seq_len_ph: seq_len,
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.4f}, val_loss: {:.4f}, acc: {:.4f}, val_acc: {:.4f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
