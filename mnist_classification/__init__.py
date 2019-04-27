import os

import numpy as np
import paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt

BATCH_SIZE = 64
PASS_NUM = 5

def cost_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=prediction,label=label)
    return prediction, avg_cost, acc

#Softmax回归(Softmax Regression)
def softmax_regression(input, label):
    return cost_net(input, label=label)

#定义多层感知器
def multilayers_perceptron(input, label):
    #第一个全连接层
    hidden1 = fluid.layers.fc(input=input, size=200, act="relu")
    #第二个全连接层
    hidden2 = fluid.layers.fc(input=hidden1, size=200, act="relu")
    #以softmax为激活函数的全连接输出层，大小为label大小
    return cost_net(hidden=hidden2,label=label)

#卷积神经网络
def convolutional_neural_network(input, label):
    #第一个卷积层+池化层，卷积核大小5*5，一共有20个卷积核，默认步长为1，激活函数为“relu”；池化层大小2*2，步长为2，默认最大池化
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=input, num_filters=20, filter_size=5, pool_size=2, pool_stride=2, act="relu")
    conv_pool_1 = fluid.layers.batch_norm(input=conv_pool_1)
    #第二个卷积层+池化层，卷积核大小5*5，一共有50个卷积核，默认步长为1，激活函数为“relu”；池化层大小2*2，步长为2，默认最大池化
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=conv_pool_1, num_filters=50, filter_size=5, pool_size=2, pool_stride=2, act="relu")
    #以softmax为激活函数的全连接层，大小为label大小
    return cost_net(conv_pool_2, label)

def train(nn_type, use_cuda, save_dirname=None, model_filename=None, params_filename=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    #定义输入层
    image = fluid.layers.data(name="image", shape=[1,28,28], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    if nn_type == 'softmax_regression':
        net_conf = softmax_regression
    elif nn_type == 'multilayer_perceptron':
        net_conf = multilayers_perceptron
    else:
        net_conf = convolutional_neural_network
    prediction, avg_cost, acc = net_conf(input=image, label=label)

    # 获取测试程序
    test_program = fluid.default_main_program().clone(for_test=True)

    #定义优化方法，交叉熵损失函数常用于分类任务
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_list = []
        avg_cost_list = []
        for test_data in train_test_reader():
            acc_np, avg_cost_np = exe.run(program=train_test_program, feed=train_test_feed.feed(test_data),fetch_list=[acc, avg_cost])
            acc_list.append(float(acc_np))
            avg_cost_list.append(float(avg_cost_np))
        acc_val_mean = np.sum(acc_list) / len(acc_list)
        avg_cost_val_mean = np.sum(avg_cost_list) / len(avg_cost_list)
        return acc_val_mean, avg_cost_val_mean

    # 定义执行器
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place=place)

    train_reader = paddle.batch(paddle.reader.shuffle(mnist.train(), buf_size=500), batch_size=BATCH_SIZE)
    test_reader = paddle.batch(paddle.reader.shuffle(mnist.test(),buf_size=500), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[image,label], place=place)

    # 进行参数初始化
    exe.run(fluid.default_startup_program())
    main_program = fluid.default_main_program()
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]

    lists = []
    step = 0

    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(program=main_program, feed=feeder.feed(data), fetch_list=[acc, avg_cost])
            if step % 100 == 0:
                print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[1]))
            step += 1

        acc_val, avg_cost_val = train_test(train_test_program=test_program, train_test_feed=feeder, train_test_reader=test_reader)
        print("Test with Epoch %d, avg_cost: %s, acc: %s" % (epoch_id, avg_cost_val, acc_val))

        lists.append((epoch_id, avg_cost_val, acc_val))
        #保存模型
        if save_dirname is not None:
            fluid.io.save_inference_model(dirname=save_dirname, feeded_var_names=["image"], target_vars=[prediction],
                                          executor=exe, model_filename=model_filename, params_filename=params_filename)

        # find the best pass
        best = sorted(lists, key=lambda list: float(list[1]))[0]
        print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
        print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

def infer(use_cuda,save_dirname=None,model_filename=None,params_filename=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def load_image(file):
        image = Image.open(file).convert('L')
        image = image.resize((28, 28), Image.ANTIALIAS)
        image = np.array(image).reshape(1, 1, 28, 28).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        return image

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tensor_img = load_image(cur_dir + '/image/infer_3.png')

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        #使用fluid.io.load_inference_model获取推理程序desc，feed_target_names（将使用feed运算符提供数据的变量的名称），
        # 以及fetch_targets（我们想要使用获取运算符获取数据的变量）。
        [inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(save_dirname, exe, model_filename, params_filename)

        #将feed构造为{feed_target_name：feed_target_data}的字典，结果将包含与fetch_targets对应的数据列表。
        results = exe.run(program=inference_program,feed={feed_target_names[0]: tensor_img},fetch_list=fetch_targets)
        lab = np.argsort(results)
        print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])

def main(use_cuda, nn_type):
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    # train(nn_type=nn_type, use_cuda=use_cuda, save_dirname=save_dirname, model_filename=model_filename, params_filename=params_filename)
    infer(use_cuda=use_cuda,save_dirname=save_dirname, model_filename=model_filename, params_filename=params_filename)



if __name__ == '__main__':
    use_cuda = False
    # predict = 'softmax_regression' # uncomment for Softmax
    # predict = 'multilayer_perceptron' # uncomment for MLP
    predict = 'convolutional_neural_network'  # uncomment for LeNet5
    main(use_cuda=use_cuda, nn_type=predict)
