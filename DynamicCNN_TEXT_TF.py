# A Convolutional Neural Network for Modelling Sentences

# https://arxiv.org/abs/1404.2188

import tensorflow as tf
from collections import Counter
import itertools
import numpy as np
import re
import time
import os

# data preprocessing:

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():

    # Load data from files
    folder_prefix = '/dcnn_data'
    x_train = list(open(folder_prefix+"train_dcnn.txt").readlines())
    x_test = list(open(folder_prefix+"test_dcnn.txt").readlines())
    test_size = len(x_test)
    x_text = x_train + x_test

    x_text = [clean_str(sent) for sent in x_text]
    y = [s.split(' ')[0].split(':')[0] for s in x_text]
    x_text = [s.split(" ")[1:] for s in x_text]
    # Generate labels
    all_label = dict()
    for label in y:
        if not label in all_label:
            all_label[label] = len(all_label) + 1
    one_hot = np.identity(len(all_label))
    y = [one_hot[ all_label[label]-1 ] for label in y]
    return [x_text, y, test_size]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, test_size = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, test_size]

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index > data_size:
                end_index = data_size
                start_index = end_index - batch_size
            yield shuffled_data[start_index:end_index]

# define the model class
class DCNN():
    def __init__(self, batch_size, sentence_length, num_filters, embed_size, top_k, k1):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.top_k = top_k # dynamic k-max
        self.k1 = k1 # k-max
    
    # k-max pooling after 1d conv
    def per_dim_conv_k_max_pooling_layer(self, x, w, b, k):
        self.k1 = k
        # given a tensor of shape (A, B, C, D) => unstack axis == 1: (A, C, D) ( (Note that the dimension unpacked along is gone)
        input_unstack = tf.unstack(x, axis=2) # input
        w_unstack = tf.unstack(w, axis=1) # window size
        b_unstack = tf.unstack(b, axis=1) # bias
        convs = []
        
        with tf.name_scope("per_dim_conv_k_max_pooling"):
            for i in range(self.embed_size):
                # define conv fiter
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])
                # conv shape:[batch_size, sent_length+ws-1, num_filters]
                conv = tf.reshape(conv, [self.batch_size, self.num_filters[0], self.sentence_length])#[batch_size, sentence_length, num_filters]
                values = tf.nn.top_k(conv, k, sorted=False).values # Finds values and indices of the k largest entries for the last dimension
                values = tf.reshape(values, [self.batch_size, k, self.num_filters[0]])
                #k_max pooling in axis=1
                convs.append(values)
            conv = tf.stack(convs, axis=2)
        #[batch_size, k1, embed_size, num_filters[0]]
        #print conv.get_shape()
        return conv
    
    # conv along the embedding
    def per_dim_conv_layer(self, x, w, b):
        
        input_unstack = tf.unstack(x, axis=2) # input
        w_unstack = tf.unstack(w, axis=1) # window size
        b_unstack = tf.unstack(b, axis=1) # bias
        convs = []
        
        with tf.name_scope("per_dim_conv"):
            for i in range(len(input_unstack)):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return conv

    # fold the embedding
    def fold_k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2): # fold every two dimension
                fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        return fold

    # fully connected layer
    def full_connect_layer(self, x, w, b, wo, dropout_keep_prob):
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(x, w) + b)
            h = tf.nn.dropout(h, dropout_keep_prob)
            o = tf.matmul(h, wo)
        return o
    
    
    # define the model
    def DCNN(self, sent, W1, W2, b1, b2, k1, top_k, Wh, bh, Wo, dropout_keep_prob):
        conv1 = self.per_dim_conv_layer(sent, W1, b1)
        conv1 = self.fold_k_max_pooling(conv1, k1)
        conv2 = self.per_dim_conv_layer(conv1, W2, b2)
        fold = self.fold_k_max_pooling(conv2, top_k)
        fold_flatten = tf.reshape(fold, [-1, int(top_k*100*14/4)])
        print(fold_flatten.get_shape())
        out = self.full_connect_layer(fold_flatten, Wh, bh, Wo, dropout_keep_prob)
        return out
    
    
# model training
        
# define hyperparameters
        
embed_dim = 100
embed_dim2 = 50
ws = [7, 5]
top_k = 4
k1 = 19
num_filters = [6, 14]
val = 300
batch_size = 50
n_epochs = 30
num_hidden = 100
sentence_length = 37 #37
num_class = 7 # 7
lr = 0.01
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5

# Load data
print("Loading data...")
x_, y_, vocabulary, vocabulary_inv, test_size = load_data()

# Randomly shuffle data
x, x_test = x_[:-test_size], x_[-test_size:]
y, y_test = y_[:-test_size], y_[-test_size:]
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

x_train, x_val = x_shuffled[:-val], x_shuffled[-val:]
y_train, y_val = y_shuffled[:-val], y_shuffled[-val:]

print("Train/val/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_val), len(y_test)))


def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

sent = tf.placeholder(tf.int64, [None, sentence_length])
y = tf.placeholder(tf.float64, [None, num_class])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")


with tf.name_scope("embedding_layer"):
    W = tf.Variable(tf.random_uniform([len(vocabulary), embed_dim], -1.0, 1.0), name="embed_W")
    sent_embed = tf.nn.embedding_lookup(W, sent)
    #input_x = tf.reshape(sent_embed, [batch_size, -1, embed_dim, 1])
    input_x = tf.expand_dims(sent_embed, -1)
    #[batch_size, sentence_length, embed_dim, 1]

W1 = init_weights([ws[0], embed_dim, 1, num_filters[0]], "W1")
b1 = tf.Variable(tf.constant(0.1, shape=[num_filters[0], embed_dim]), "b1")

W2 = init_weights([ws[1], embed_dim2, num_filters[0], num_filters[1]], "W2")
b2 = tf.Variable(tf.constant(0.1, shape=[num_filters[1], embed_dim]), "b2")

Wh = init_weights([int(top_k*embed_dim*num_filters[1]/4), num_hidden], "Wh")
bh = tf.Variable(tf.constant(0.1, shape=[num_hidden]), "bh")



Wo = init_weights([num_hidden, num_class], "Wo")

model = DCNN(batch_size, sentence_length, num_filters, embed_dim, top_k, k1)
out = model.DCNN(input_x, W1, W2, b1, b2, k1, top_k, Wh, bh, Wo, dropout_keep_prob)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
# train_step = tf.train.AdamOptimizer(lr).minimize(cost)

predict_op = tf.argmax(out, axis=1, name="predictions")
with tf.name_scope("accuracy"):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), tf.float32))

#batches = batch_iter(zip(x_train, y_train), batch_size, n_epochs)

print('Started training')
with tf.Session() as sess:
    #init = tf.global_variables_initializer().run()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cost)
    acc_summary = tf.summary.scalar("accuracy", acc)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # val summaries
    val_summary_op = tf.summary.merge([loss_summary, acc_summary])
    val_summary_dir = os.path.join(out_dir, "summaries", "val")
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            sent: x_batch,
            y: y_batch,
            dropout_keep_prob: 0.5
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cost, acc],
            feed_dict)
        print("TRAIN step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def val_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a val set
        """
        feed_dict = {
            sent: x_batch,
            y: y_batch,
            dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, val_summary_op, cost, acc],
            feed_dict)
        print("VALID step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return accuracy, loss


#    batches = batch_iter(zip(x_train, y_train), batch_size, n_epochs)
    batches = batch_iter(list(zip(x_train, y_train)), batch_size, n_epochs)

    # Training loop. For each batch...
    max_acc = 0
    best_at_step = 0
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            acc_val, _ = val_step(x_val, y_val, writer=val_summary_writer)
            if acc_val >= max_acc:
                max_acc = acc_val
                best_at_step = current_step
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("")
        if current_step % checkpoint_every == 0:
            print('Best of valid = {}, at step {}'.format(max_acc, best_at_step))

    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print('Finish training. On test set:')
    acc, loss = val_step(x_test, y_test, writer=None)
    print(acc, loss)