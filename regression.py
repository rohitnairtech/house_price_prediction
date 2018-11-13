import tensorflow as tf
import numpy as np
import os
cwd = os.getcwd()
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 1

vec_size = 2

x = tf.placeholder('float', [None, vec_size])
y = tf.placeholder('float')


dataset = tf.contrib.data.make_csv_dataset('./house.csv', batch_size=40)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([vec_size, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

    return output


hm_epochs = 1000

prediction = neural_network_model(x)
cost = tf.losses.mean_squared_error(y, prediction)
#cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(hm_epochs):
        epoch_loss = 0
        iterator = dataset.make_one_shot_iterator()
        justArr = []
        for _ in range(2):
            nextIter = iterator.get_next()
            currData = sess.run(nextIter)
            house_size, BHK, rates = currData['house_size'], currData['BHK'], currData['rates']
            for i in range(len(BHK)):
                justArr.append([house_size[i],BHK[i]])

        _, c = sess.run([optimizer, cost], feed_dict={x: justArr, y: rates})
        epoch_loss += c
        print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
    saver.save(sess, cwd+"/model.ckpt")

print(cwd)

def train_neural_network(x):
    print(x)
    prediction = neural_network_model(x)
    cost = tf.losses.mean_squared_error(y, prediction)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    
    hm_epochs = 500
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #training the model
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                end = i + batch_size

                batch_x = np.array(train_x[i:end])
                batch_y = np.array(train_y[i:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        #testing the model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

#train_neural_network(x)