import os

import cv2
import numpy as np
import tensorflow as tf

from DataSetGenerator import DataSet

if __name__ == "__main__":

    tf.logging.set_verbosity("INFO")

    # Parameters
    dataset_path = ""
    batch_size = 128
    learning_rate = 1e-7
    training_epochs = 10
    display_step = 1
    img_shape = (200, 200, 3)
    img_shape_flatten = np.prod(img_shape)
    y_shape = 1  # class A or B

    dataset = DataSet(batch_size=batch_size,
                      image_size=img_shape,
                      allchannel=True)
    for p in os.listdir(dataset_path):
        class_path = os.path.abspath(os.path.join(dataset_path, p))
        if os.path.isdir(class_path):
            dataset.add_class(class_path)
            if dataset.class_cnt == 2:
                break

    # Inputs
    X = tf.placeholder(tf.float32, shape=(None, img_shape_flatten), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, y_shape), name="Y")

    # Weights
    W = tf.Variable(tf.zeros(dtype=tf.float32, shape=(img_shape_flatten, y_shape), name="W"))
    b = tf.Variable(tf.zeros(dtype=tf.float32, shape=(y_shape, 1), name="b"))

    Z = tf.matmul(X, W) + b
    A = tf.nn.sigmoid(Z, name="Prediction")

    loss = -Y * tf.log(A) - (1 - Y) * tf.log(1 - A)
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
                                feed_dict={X: x.reshape(-1, img_shape_flatten), Y: y.reshape(-1, 1)})

                avg_cost = c / len(x)
            if (epoch + 1) % display_step == 0:
                print("Epoch: {}, cost = {}".format(epoch + 1, avg_cost))

        # Testing model
        batch = dataset.get_batches_test()

        accuracy = []
        for y, x in batch:

            local_acc = sess.run(A, feed_dict={X: x.reshape(-1, img_shape_flatten)}).flatten()
            local_acc = np.where(local_acc >= 0.5, 1, 0)
            acc = np.mean(local_acc == y.flatten())

            accuracy.append(acc)

        writer.close()

        print("accuracy: {}".format(np.mean(accuracy)))

        # Predict
        np.random.seed(0)
        while True:
            image = dataset.get_random_example()
            y_head = sess.run([A], feed_dict={X: image.reshape(1, -1)})[0]

            if y_head > 0.5:
                msg = "Image of class A!"
            else:
                msg = "Image of class B!"

            cv2.imshow(msg, image)
            cv2.waitKey()
            cv2.destroyAllWindows()
