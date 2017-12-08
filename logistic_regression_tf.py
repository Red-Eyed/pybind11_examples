import cv2
import numpy as np
import tensorflow as tf

from DataSetGenerator import DataSetGenerator

if __name__ == "__main__":
    path_class_a = "/path/to/folder/with/cats"
    path_class_b = "/path/to/folder/with/dogs"
    tf.logging.set_verbosity("INFO")

    # Parameters
    batch_size = 64
    learning_rate = 1e-10
    training_epochs = 5
    batch_size = 100
    display_step = 1
    img_shape = (200, 200, 3)
    img_shape_flatten = np.prod(img_shape)
    y_shape = 1  # class A or B

    # Inputs
    X = tf.placeholder(tf.float32, shape=(None, img_shape_flatten), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, y_shape), name="Y")

    # Weights
    W = tf.Variable(tf.zeros(dtype=tf.float32, shape=(img_shape_flatten, y_shape), name="W"))
    b = tf.Variable(tf.zeros(dtype=tf.float32, shape=(y_shape, 1), name="b"))

    Z = tf.matmul(X, W) + b
    A = tf.nn.sigmoid(Z, name="Prediction")

    loss = -Y * tf.log(A) - (1 - Y) * tf.log(1 - A)
    cost = tf.reduce_mean(loss)

    # Optimizing
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name="Opt")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

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
                                    np.array([0] * len(class_a_ex), dtype=np.uint8).reshape(-1, y_shape)])
                x = np.concatenate(
                    [class_a_ex.reshape(-1, img_shape_flatten), class_b_ex.reshape(-1, img_shape_flatten)])

                c, _ = sess.run([cost, opt],
                                feed_dict={X: x, Y: y})

                avg_cost = c / len(x)
            if (epoch + 1) % display_step == 0:
                print("Epoch: {}, cost = {}".format(epoch + 1, avg_cost))

        # Predict
        class_a = DataSetGenerator(path_class_a).get_data_paths()
        class_b = DataSetGenerator(path_class_b).get_data_paths()

        np.random.seed(0)
        while True:
            paths = np.random.permutation(class_a + class_b)

            concrete_path = np.random.choice(paths, 1)
            image = cv2.imread(concrete_path[0])
            accuracy = sess.run([A], feed_dict={X: image.reshape(1, -1)})[0]

            if accuracy > 0.5:
                msg = "Image of class A!"
            else:
                msg = "Image of class B!"

            cv2.imshow(msg, image)
            cv2.waitKey()
            cv2.destroyAllWindows()
