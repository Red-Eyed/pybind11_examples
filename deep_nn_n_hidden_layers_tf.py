import os

import cv2
import numpy as np
import tensorflow as tf

from DataSetGenerator import DataSet


def init_params(nodes, prev_nodes):
    xavier_init = tf.to_double(tf.sqrt(2.0 / prev_nodes))
    w = tf.Variable(tf.random_normal(mean=0.0, stddev=1.0, dtype=tf.float64, shape=(nodes, prev_nodes)) * xavier_init)
    b = tf.Variable(tf.zeros(dtype=tf.float64, shape=(nodes, 1)))

    return w, b


if __name__ == "__main__":
    tf.logging.set_verbosity("INFO")

    # Parameters
    dataset_path = ""
    lambd = 0.01
    alpha = 0.1
    batch_size = 100
    learning_rate = 1e-7
    training_epochs = 10
    display_step = 1
    img_shape = (200, 200, 3)
    img_shape_flatten = np.prod(img_shape)
    h_layer_shape = 200
    y_shape = 1  # class A or B
    L = 2

    dataset = DataSet(batch_size=batch_size,
                      image_size=img_shape,
                      allchannel=True)
    for p in os.listdir(dataset_path):
        class_path = os.path.abspath(os.path.join(dataset_path, p))
        if os.path.isdir(class_path):
            dataset.add_class(class_path, "jpg")
            if dataset.class_cnt == 2:
                break

    # Inputs
    X = tf.placeholder(tf.float64, shape=(img_shape_flatten, None), name="X")
    Y = tf.placeholder(tf.float64, shape=(y_shape, None), name="Y")

    # Weights
    W = dict()
    b = dict()
    Z = dict()
    A = dict()
    A[0] = X

    tf.set_random_seed(0)
    np.random.seed(0)
    # Init input layer
    W[1], b[1] = init_params(h_layer_shape, img_shape_flatten)

    # Init hidden layers
    for l in range(2, L - 1):
        W[l], b[l] = init_params(h_layer_shape, h_layer_shape)

    # Init output layer
    W[L - 1], b[L - 1] = init_params(y_shape, h_layer_shape)

    # Forward prop
    for l in range(1, L):
        Z[l] = tf.matmul(W[l], A[l - 1]) + b[l]
        A[l] = tf.nn.sigmoid(Z[l], name="Prediction" + str(l))

    # L2 Regularization
    R = 0
    for l in range(1, L):
        R += tf.norm(W[l])

    loss = -Y * tf.log(A[L - 1]) - (1 - Y) * tf.log(1 - A[L - 1]) #+ lambd * R
    cost = tf.reduce_mean(loss, name="Cost")

    # Optimizing
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost, name="Opt")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./logs/graphs", graph=tf.get_default_graph())

        # Training model
        for epoch in range(training_epochs):
            avg_cost = .0
            batch = dataset.get_batches_train()

            for y, x in batch:
                c, _ = sess.run([cost, opt],
                                feed_dict={X: x.reshape(img_shape_flatten, -1), Y: y.reshape(1, -1)})

                avg_cost = c / len(x)
            if (epoch + 1) % display_step == 0:
                print("Epoch: {}, cost = {}".format(epoch + 1, avg_cost))

        # Testing model
        batch = dataset.get_batches_test()

        accuracy = []
        for y, x in batch:

            local_acc = sess.run(A[L-1], feed_dict={X: x.reshape(img_shape_flatten, -1)}).flatten()
            local_acc = np.where(local_acc >= 0.5, 1, 0)
            acc = np.mean(local_acc == y.flatten())

            accuracy.append(acc)

        writer.close()

        print("accuracy: {}".format(np.mean(accuracy)))

        # Predict
        np.random.seed(0)
        while True:
            image = dataset.get_random_example()
            y_head = sess.run(A[L-1], feed_dict={X: image.reshape(-1, 1)})

            if y_head > 0.5:
                msg = "Image of class A!"
            else:
                msg = "Image of class B!"

            cv2.imshow(msg, image)
            cv2.waitKey()
            cv2.destroyAllWindows()
