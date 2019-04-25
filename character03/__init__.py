import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.fluid as fluid

TRAIN_DATA = None
X_RAW = None
TEST_DATA = None

def load_data(filename, feature_num=2, ratio=0.8):
    global  TRAIN_DATA, X_RAW, TEST_DATA
    data = np.loadtxt(filename, delimiter=',')
    X_RAW = data.T[0].copy()
    maximums = data.max(axis=0)
    minimums = data.min(axis=0)
    avgs = data.sum(axis=0) / data.shape[0]
    for i in range(feature_num - 1):
        data[:,i] = (data[:,i] - avgs[i]) / (maximums - minimums)

    offset = int(data.shape[0] * ratio)
    TRAIN_DATA = data[:offset].copy()
    TEST_DATA = data[offset:].copy()

def read_data(data_set):
    def reader():
        for data in data_set:
            yield data[:-1], data[-1:]
    return reader()

def train():
    global TRAIN_DATA
    load_data("data.txt")
    return read_data(TRAIN_DATA)

def test():
    global TEST_DATA
    load_data("data.txt")
    return read_data(TEST_DATA)

def network_config():
    x = fluid.layers.data(name='x', shape=[1], dtype="float32")
    y = fluid.layers.data(name='y', shape=[1], dtype="float32")
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    test_program = main_program.clone(for_test=True)

    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace

    exe = fluid.Executor(place=place)

    num_epochs = 100

    def train_test(executor, program, reader, feeder, fetch_list):
        accumulated = 1 * [0]
        count = 0
        for data_test in reader():
            outs = executor.run(program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
            accumulated = [x_c[0]]
