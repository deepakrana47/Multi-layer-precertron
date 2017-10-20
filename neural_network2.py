import numpy as np
import pickle
from random import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import warnings

log_file = open('log.txt','w')
warnings.filterwarnings("error")

def init_weight(size1, size2):
    mean = 0
    sigma = 0.01
    return np.random.normal(mean, sigma, (size1, size2))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def desigmoid(x):
    t = sigmoid(x)
    return (1 - t) * t

def softmax(x):
    try:
        # x = np.where(x>50,50,x)
        # x = np.where(x<-50,-50,x)
        sum = np.sum(np.exp(x))
    except RuntimeWarning:
        pass
    try:
        t = np.exp(x)/sum
    except RuntimeWarning:
        pass
    return t

def desoftmax(x,y,layr='out'):
    if layr == 'out':
        return x-y

def relu(x):
    return np.where(x<0,0,x)

def derelu(x):
    return np.where(x>0,1,0.01)

# error
def erro(o, t):
    try:
        # o = np.where(o<1e-50, 1e-50, o)
        e=-np.sum(t*np.log(o))
    except RuntimeWarning:
        e = -np.sum(t * np.log(np.where(o == 0.0, 1.0, o)))
        pass
    return e

# gradiant calc
def grad(g, x):
    return np.dot(g.reshape(g.shape[0], 1), x.reshape(1, x.shape[0])).T

def train_process(x, t, wi2h1, wh12h2, wh22o, neta, regu):
    global m1, m2, m3, v1, v2, v3, lr1, lr2, lr3
    err = 0.0
    dw_h22o = np.zeros(wh22o.shape)
    dw_i2h1 = np.zeros(wi2h1.shape)
    dw_h12h2 = np.zeros(wh12h2.shape)
    for i in range(x.shape[0]):
        ## feedforward
        # input to hidden1 activation
        h1 = relu(np.dot(x[i], wi2h1))

        # hidden1 to hidden2 activation
        h2 = relu(np.dot(h1, wh12h2))

        # hidden to output activation
        o = softmax(np.dot(h2, wh22o))

        #error calculation
        t0=erro(o,t[i])
        err += t0

        ## backpropogation

        # gradient
        g_o2h2 = o-t[i]
        g_h22h1 = np.dot(g_o2h2, wh22o.T)*derelu(h2)
        g_h12i = np.dot(g_h22h1, wh12h2.T)*derelu(h1)
        to = grad(g_o2h2, h2)
        th2 = grad(g_h22h1, h1)
        th1 = grad(g_h12i, x[i])

        dw_h22o += to
        dw_h12h2 += th2
        dw_i2h1 += th1

    # dw_i2h1 += regu * wi2h1 * np.sign(wi2h1)
    # dw_h12h2 += regu * wh12h2 * np.sign(wh12h2)
    # dw_h22o += regu * wh22o * np.sign(wh22o)

    wh22o, m3, v3, lr3 = adam_weight_update(wh22o, dw_h22o / x.shape[0], m3, v3, lr3)
    wh12h2, m2, v2, lr2 = adam_weight_update(wh12h2, dw_h12h2 / x.shape[0], m2, v2, lr2)
    wi2h1, m1, v1, lr1 = adam_weight_update(wi2h1, dw_i2h1 / x.shape[0], m1, v1, lr1)
    return  wi2h1, wh12h2, wh22o, err

def adam_weight_update(w, g, m, v, lr, b1=0.9, b2=0.99, e=1e-8):
    i_t = 1.0
    fix1 = (1. - b1)
    fix2 = (1. - b2)
    m_t = (b1 * g) + ((1. - b1) * m)
    v_t = (b2 * np.square(g)) + ((1. - b2) * v)
    g_t = m_t / (np.sqrt(v_t) + e)
    w = w - (lr * g_t)
    m = m_t
    v = v_t
    lr = lr * (np.sqrt(fix2) / fix1)
    return w, m, v, lr

def shuff(t, l, n):
    a = [(t[i], l[i]) for i in range(n)]
    shuffle(a)
    x = [];
    y = [];
    for i in a:
        x.append(i[0])
        y.append(i[1])
    return np.array(x), np.array(y)

def train(train_data, train_label, wi2h1, wh12h2, wh22o, log=0, batchs=200, training_epoc=20, neta=0.01, regu=0.01,layrs=[]):
    global m1, m2, m3, v1, v2, v3, lr1, lr2, lr3
    epoc = 0
    nsample = len(train_data)
    print "Number of batchs : %d" % (nsample/batchs)
    if log:
        log_file.write("Number of batchs : %d\n" % (nsample/batchs))
    err = 1.0
    while err > 0.01 and epoc < training_epoc:
        m1 = np.zeros((layrs[0], layrs[1]))
        v1 = np.zeros((layrs[0], layrs[1]))
        m2 = np.zeros((layrs[1], layrs[2]))
        v2 = np.zeros((layrs[1], layrs[2]))
        m3 = np.zeros((layrs[2], layrs[3]))
        v3 = np.zeros((layrs[2], layrs[3]))
        lr1 = 0.001
        lr2 = 0.001
        lr3 = 0.001
        index = 0
        terr = 0.0
        # count = 0
        x, y = shuff(train_data, train_label, nsample)
        while index < nsample:
            wtrain = (index + batchs) if index + batchs < nsample else nsample
            wi2h1, wh12h2, wh22o, err = train_process(x[index:wtrain], y[index:wtrain], wi2h1, wh12h2, wh22o, neta, regu)
            terr += err
            index = wtrain

        err = terr/(nsample/batchs)
        print "trainning error at epoch %d : %f" % (epoc, err)
        print "Weights sum:"
        print "\ti - h1 :\t", np.sum(np.abs(wi2h1))
        print "\th1 - h2 :\t", np.sum(np.abs(wh12h2))
        print "\th2 - o :\t", np.sum(np.abs(wh22o))
        if log:
            log_file.write("trainning error at epoch %d : %f\n" % (epoc, err))
        epoc += 1
    return wi2h1, wh12h2, wh22o

def test(x, y, wi2h1, wh12h2, wh22o, log=0):
    correct = 0
    total = x.shape[0]
    for i in range(x.shape[0]):
        t = y[i]
        ## feedforward
        # input to hidden1 activation
        h1 = relu(np.dot(x[i], wi2h1))

        # hidden1 to hidden2 activation
        h2 = relu(np.dot(h1, wh12h2))

        # hidden to output activation
        o = softmax(np.dot(h2, wh22o))

        if np.argmax(t) == np.argmax(o):
            correct +=1
    print "accuracy : %f" % (correct/total)
    if log:
        log_file.write("accuracy : %f\n" % (correct/total))

if __name__=="__main__":
    # i-h-o layer
    i_size = 28 * 28
    h1_size = 256
    h2_size = 256
    o_size = 10
    # weights
    w_i2h1 = init_weight(i_size, h1_size)
    w_h12h2 = init_weight(h1_size, h2_size)
    w_h22o = init_weight(h2_size, o_size)
    # w_i2h = np.zeros((i_size, h_size))
    # w_h2o = np.zeros((h_size, o_size))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_data = mnist.train.images
    train_label = mnist.train.labels
    test_data = mnist.test.images
    test_label = mnist.test.labels

    w_i2h1, w_h12h2, w_h22o = train(train_data, train_label, w_i2h1, w_h12h2, w_h22o, log=1, layrs=[i_size, h1_size, h2_size, o_size])
    # w_i2h1, w_h12h2, w_h22o = train(train_data[:1000], train_label[:1000], w_i2h1, w_h12h2, w_h22o, log=1, layrs=[i_size, h1_size, h2_size, o_size])
    pickle.dump([w_i2h1, w_h12h2, w_h22o],open("weights.pickle",'wb'))
    # w_i2h1, w_h12h2, w_h22o = pickle.load(open('weights.pickle','rb'))
    test(test_data, test_label, w_i2h1, w_h12h2, w_h22o, log=1)
    # test(train_data[:100], train_label[:100], w_i2h1, w_h12h2, w_h22o, log=1)
