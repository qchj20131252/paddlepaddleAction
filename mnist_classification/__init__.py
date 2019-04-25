import numpy as np
import paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt

#定义多层感知器
def multilayers_perceptron(input):
    #第一个全连接层
    hidden1 = fluid.layers.fc(input=input, size=100, act="relu")
    #第二个全连接层
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act="relu")
    #以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2, size=10, act="softmax")
    return fc

#卷积神经网络
def convolutional_neural_network(input):
    #第一个卷积层，卷积核大小3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input=input, num_filters=32, filter_size=3, stride=1)
    #第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1, pool_size=2, pool_stride=1, pool_type="max")
    #第二个卷积层，卷积核大小3*3，一共有64个卷积核
    conv2 = fluid.layers.conv2d(input=pool1, num_filters=64, filter_size=3, stride=1)
    #第二个池化层，池化大小2*2，步长为1，最大池化
    pool2 = fluid.layers.pool2d(input=conv2, pool_size=2, pool_stride=1, pool_type="max")
    #以softmax为激活函数的全连接层，大小为label大小
    fc = fluid.layers.fc(input=pool2,size=10,act="softmax")
    return fc

def main():
    #获取mnist数据
    batch_size = 128
    train_reader = paddle.batch(mnist.train(),batch_size=batch_size)
    test_reader = paddle.batch(mnist.test(),batch_size=batch_size)

    #定义输入层
    image = fluid.layers.data(name="image", shape=[1,28,28], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    #获取分类器
    model = multilayers_perceptron(image)
    # model = convolutional_neural_network(image)

    main_program = fluid.default_main_program()#获取默认/全局主函数
    startup_program = fluid.default_startup_program()#获取默认/全局启动程序

    #获取损失函数和准确率函数
    cost = fluid.layers.cross_entropy(input=model,label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=model,label=label)

    #获取测试程序
    test_program = main_program.clone(for_test=True)

    #定义优化方法，交叉熵损失函数常用于分类任务
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    #定义执行器
    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 指定保存参数的目录
    params_dirname = "./model/mnist_classification.inference.model"
    num_epochs = 10

    #定义输入数据维度
    feeder = fluid.DataFeeder(place=place, feed_list=[image,label])

    #进行参数初始化
    exe.run(startup_program)

    exe_test = fluid.Executor(place)

    #开始训练和进行测试
    for pass_id in range(num_epochs):
        #进行训练
        for batch_id, data in enumerate(train_reader()):
            train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_cost,acc])
            #每100个batch打印一次信息
            if batch_id % 100 == 0:
                print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))


        #进行测试
        test_accs = []
        test_costs = []
        for batch_id, data in enumerate(test_reader()):
            test_cost, test_acc = exe_test.run(program=test_program, feed=feeder.feed(data), fetch_list=[avg_cost,acc])
            test_accs.append(test_acc[0])
            test_costs.append(test_cost[0])

        #求测试结果的平均值
        test_cost = np.sum(test_costs) / len(test_costs)
        test_acc = np.sum(test_accs) / len(test_accs)

        print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))



if __name__ == '__main__':
    main()
