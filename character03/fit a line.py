from __future__ import print_function

import sys

import math
import numpy

import paddle
import paddle.fluid as fluid


# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]

# 保存图片
def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')


def main():
    batch_size = 20
    train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500), batch_size=batch_size)
    test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500), batch_size=batch_size)

    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')#定义输入的形状和数据类型
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')#定义输出的形状和数据类型
    hidden =fluid.layers.fc(input=x, size=100, act="relu")
    y_predict = fluid.layers.fc(input=hidden, size=1, act=None)#连接输入和输出的全连接层

    main_program = fluid.default_main_program()#获取默认/全局主函数
    startup_program = fluid.default_startup_program()#获取默认/全局启动程序

    cost = fluid.layers.square_error_cost(input=y_predict, label=y) # 利用标签数据和输出的预测数据估计方差
    avg_loss = fluid.layers.mean(cost) # 对方差求均值，得到平均损失

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    # 克隆main_program得到test_program
    # 有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
    # 该api不会删除任何操作符,请在backward和optimization之前使用
    test_program = main_program.clone(for_test=True)

    # can use CPU or GPU
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()# 指明executor的执行场所
    # executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)向program中添加数据输入算子和结果
    # 获取算子。使用close()关闭该executor，调用run(...)执行program。
    exe = fluid.Executor(place)

    # 指定保存参数的目录
    params_dirname = "fit_a_line.inference.model"
    num_epochs = 100

    # 训练主循环
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)

    train_prompt = "Train cost"
    test_prompt = "Test cost"
    step = 0

    exe_test = fluid.Executor(place)

    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(main_program, feed=feeder.feed(data_train), fetch_list=[avg_loss])
            if step % 10 == 0:  # 每10个批次记录并输出一下训练损失
                print("%s, Step %d, Cost %f" % (train_prompt, step, avg_loss_value[0]))

            if step % 100 == 0:  # 每100批次记录并输出一下测试损失
                test_metics = train_test(executor=exe_test, program=test_program, reader=test_reader, fetch_list=[avg_loss], feeder=feeder)
                print("%s, Step %d, Cost %f" % (test_prompt, step, test_metics[0]))
                # 如果准确率达到要求，则停止训练
                if test_metics[0] < 10.0:
                    break

            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")
        if params_dirname is not None:
            # 我们可以将经过训练的参数保存到以后的推理中
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

    #类似于训练过程，预测器需要一个预测程序来做预测
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # 通过fluid.io.load_inference_model，预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。
    with fluid.scope_guard(inference_scope):
        # 载入预训练模型
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10

        infer_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=batch_size) # 准备测试集

        infer_data = next(infer_reader())
        infer_feat = numpy.array([data[0] for data in infer_data]).astype("float32")# 提取测试集中的数据
        infer_label = numpy.array([data[1] for data in infer_data]).astype("float32") # 提取测试集中的标签

        assert feed_target_names[0] == 'x'
        # 进行预测
        results = infer_exe.run(inference_program, feed={feed_target_names[0]: numpy.array(infer_feat)}, fetch_list=fetch_targets)

        # 打印预测结果和标签并可视化结果
        print("infer results: (House Price)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))# 打印预测结果

        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))# 打印标签值

        save_result(results[0], infer_label)# 保存图片


if __name__ == '__main__':
    main()