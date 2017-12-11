import cv2
import numpy as np
import tensorflow as tf

from DataSetGenerator import DataSetGenerator


def init_params(nodes, prev_nodes):
    W = tf.Variable(tf.random_normal(dtype=tf.float32, shape=(nodes, prev_nodes)))
    b = tf.Variable(tf.zeros(dtype=tf.float32, shape=(nodes, 1)))

    return W, b


if __name__ == "__main__":
    path_class_a = "/home/v.stupakov/Work/Projects/CalliScan/DataSet/linecutter2/dataset_lined"
    path_class_b = "/home/v.stupakov/Work/Projects/CalliScan/DataSet/linecutter2/dataset_white"
    tf.logging.set_verbosity("INFO")

    # Parameters
    batch_size = 128
    learning_rate = 1e-4
    training_epochs = 5
    display_step = 1
    img_shape = (200, 200, 3)
    img_shape_flatten = np.prod(img_shape)
    h_layer_shape = 200
    y_shape = 1  # class A or B
    L = 6

    # Inputs
    X = tf.placeholder(tf.float32, shape=(img_shape_flatten, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(y_shape, None), name="Y")

    # Weights
    W = dict()
    b = dict()
    Z = dict()
    A = dict()
    A[0] = X

    # Init input layer
    W[1], b[1] = init_params(h_layer_shape, img_shape_flatten)

    # Init hidden layers
    for l in range(2, L - 1):
        W[l], b[l] = init_params(h_layer_shape, h_layer_shape)

    # Init output layer
    W[L-1], b[L-1] = init_params(y_shape, h_layer_shape)

    # Forward prop
    for l in range(1, L):
        Z[l] = tf.matmul(W[l], A[l-1]) + b[l]
        A[l] = tf.nn.sigmoid(Z[l], name="Prediction" + str(l))

    loss = -Y * tf.log(A[L-1]) - (1 - Y) * tf.log(1 - A[L-1])
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
            batch_class_a = DataSetGenerator(path_class_a).get_batches_train(
                batch_size=batch_size,
                image_size=img_shape,
                allchannel=True)

            batch_class_b = DataSetGenerator(path_class_b).get_batches_train(
                batch_size=batch_size,
                image_size=img_shape,
                allchannel=True)

            for class_a_ex, class_b_ex in zip(batch_class_a, batch_class_b):
                y = np.concatenate([np.array([1] * len(class_a_ex), dtype=np.uint8).reshape(-1, y_shape),
                                    np.array([0] * len(class_b_ex), dtype=np.uint8).reshape(-1, y_shape)])
                x = np.concatenate(
                    [class_a_ex.reshape(-1, img_shape_flatten), class_b_ex.reshape(-1, img_shape_flatten)])

                c, _ = sess.run([cost, opt],
                                feed_dict={X: x.T, Y: y.T})

                avg_cost = c / len(x)
            if (epoch + 1) % display_step == 0:
                print("Epoch: {}, cost = {}".format(epoch + 1, avg_cost))

        # Testing model
        batch_class_a = DataSetGenerator(path_class_a).get_batches_test(
            batch_size=batch_size,
            image_size=img_shape,
            allchannel=True)

        batch_class_b = DataSetGenerator(path_class_b).get_batches_test(
            batch_size=batch_size,
            image_size=img_shape,
            allchannel=True)

        accuracy = []
        for class_a_ex, class_b_ex in zip(batch_class_a, batch_class_b):
            y = np.concatenate([np.array([1] * len(class_a_ex), dtype=np.uint8).reshape(-1, y_shape),
                                np.array([0] * len(class_b_ex), dtype=np.uint8).reshape(-1, y_shape)])
            x = np.concatenate(
                [class_a_ex.reshape(-1, img_shape_flatten), class_b_ex.reshape(-1, img_shape_flatten)])

            local_acc = sess.run([A[L-1]], feed_dict={X: x.T})[0].flatten()
            local_acc = np.where(local_acc >= 0.5, 1, 0)
            acc = np.sum(local_acc == y.flatten()) / len(local_acc)

            accuracy.append(acc)

        writer.close()

        print("accuracy: {}".format(np.mean(accuracy)))

        # Predict
        class_a = DataSetGenerator(path_class_a).get_data_paths()
        class_b = DataSetGenerator(path_class_b).get_data_paths()

        np.random.seed(0)
        while True:
            paths = np.random.permutation(class_a + class_b)

            concrete_path = np.random.choice(paths, 1)
            image = cv2.imread(concrete_path[0])
            y_head = sess.run([A[L-1]], feed_dict={X: image.reshape(1, -1).T})[0]

            if y_head > 0.5:
                msg = "Image of class A!"
            else:
                msg = "Image of class B!"

            cv2.imshow(msg, image)
            cv2.waitKey()
            cv2.destroyAllWindows()
