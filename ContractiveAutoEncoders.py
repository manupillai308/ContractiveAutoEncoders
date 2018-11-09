import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data")

x_test = mnist.test.images
y_test = mnist.test.labels

tf.reset_default_graph()

with tf.variable_scope('ioMetrics'):
    X = tf.placeholder(dtype = tf.float32, name='X', shape = (None, 784))
    y = tf.placeholder(dtype = tf.float32, name='y', shape = (None, 784))

n_hidden1 = 500
n_hidden2 = 100
n_codings = 20
n_hidden3 = n_hidden2
n_hidden4 = n_hidden1
batch_size = 150
n_epochs = 50
n_outputs = 784
learning_rate = 0.01

from functools import partial

create_dense = partial(tf.layers.dense, activation = tf.nn.sigmoid,
                       kernel_initializer = tf.contrib.layers.variance_scaling_initializer())

with tf.variable_scope('Ae'):
    hidden1 = create_dense(inputs=X, units = n_hidden1)
    hidden2 = create_dense(inputs= hidden1, units = n_hidden2)
    codings = create_dense(inputs = hidden2, units = n_codings)
    hidden3 = create_dense(inputs = codings, units = n_hidden3)
    hidden4 = create_dense(inputs = hidden3, units = n_hidden4)
    logits = create_dense(inputs = hidden4, units = n_outputs, activation = None)

with tf.variable_scope('Loss'):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels=y)
    reconstruction_loss = tf.reduce_sum(xentropy)
    contractive_loss = tf.reduce_sum(tf.square(tf.gradients(codings, X, stop_gradients = [X])))
    loss = reconstruction_loss + contractive_loss

with tf.variable_scope('Training_op'):
    optimizer = tf.train.MomentumOptimizer(momentum=0.9,use_nesterov=True, learning_rate = learning_rate)
    training_op = optimizer.minimize(loss)

with tf.variable_scope('performance'):
    predictions = tf.nn.sigmoid(logits)
    mse = tf.reduce_mean(tf.square(predictions - y))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for batch_no in range(mnist.train.num_examples//batch_size):
            x_train, y_train = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {X:x_train, y:x_train})
        mse_train = mse.eval(feed_dict={X:x_train, y:x_train})
        mse_test = mse.eval(feed_dict={X:x_test, y:x_test})
        c_loss = contractive_loss.eval(feed_dict = {X:x_test, y:x_test})
        print('epoch:', epoch, 'Training mse:', mse_train, 'Testing accuracy:', mse_test,'Contractive Loss:', c_loss)
