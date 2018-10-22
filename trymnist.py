from uwnet import *

def softmax_model():
    l = [make_connected_layer(784, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(784, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

def convnet():
    l = [   make_convolutional_layer(28, 28,  1,  8, 3, 1, LRELU),
            make_maxpool_layer(28, 28, 8, 3, 2),
            make_convolutional_layer(14, 14,  8, 16, 3, 1, LRELU),
            make_maxpool_layer(14, 14, 16, 3, 2),
            make_convolutional_layer( 7,  7, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(7, 7, 32, 3, 2),
            make_convolutional_layer( 4,  4, 32, 32, 3, 1, LRELU),
            make_connected_layer(512, 10, SOFTMAX)]
    return make_net(l)

def le_net():
    l = [   make_convolutional_layer(28, 28,  1,  8, 3, 1, LRELU),
            make_convolutional_layer(28, 28,  8, 16, 3, 2, LRELU),
            make_convolutional_layer(14, 14, 16, 32, 3, 2, LRELU),
            make_convolutional_layer( 7,  7, 32, 32, 3, 2, LRELU),
            make_connected_layer(512, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .0005

m = le_net()
print("training...")
train_image_classifier(m, train, batch, iters/2, rate, momentum, decay)
train_image_classifier(m, train, batch, iters/2, rate*.1, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

