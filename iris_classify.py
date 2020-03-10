import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# 数据集读入
x_data = datasets.load_iris().data      # 返回iris数据集所有输入特征
y_data = datasets.load_iris().target    # 返回iris数据集所有标签

# 数据集乱序
# 使用相同的seed，使输入特征/标签一一对应
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

# 分割数据为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 数据类型转换
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

# 配成【输入特征，标签】对，每次喂入一小撮
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# 定义神经网络所有可训练的参数
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

lr = 0.1    # 学习率为0.1
train_loss_results = []     #将每轮的loss记录在此列表中，为后续loss曲线提供数据
test_acc = []       #将每轮的acc记录在此列表中，为后续accuracy曲线提供数据
epoch = 500     # 训练500轮
loss_all = 0        # 每轮分为4个step， loss_all记录四个step生成的4个loss的和

# 训练部分，嵌套循环迭代，with结构更新参数，并显示当前loss值
for epoch in range(epoch):      # 数据集级别的循环，每个epoch循环一次数据集
    for step,(x_train,y_train) in enumerate(train_db):      # batch级别的循环
        with tf.GradientTape() as tape:     # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)    # 使输出符合概率分布
            y_ = tf.one_hot(y_train, depth=3)   # 将标签转换为独热码格式，方便后续计算loss
            loss = tf.reduce_mean(tf.square(y_ - y))    # 采用均方误差损失函数
            loss_all +=loss.numpy() # 将每个step计算的loss累加，为后续求loss平均值提供数据

        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss,[w1,b1])

        # 实现梯度更新，w1 = w1 - lr * w1_grad; b1 = b1 - lr * b_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # 每个epoch，打印loss信息
    print("Epoch {},loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)   # 将4个step的平均loss记录在变量中
    loss_all = 0    # loss_all归零，为记录下一个epoch做准备

    # 测试部分
    # total_correct为预测正确的样本个数， total_number为样本总数
    total_correct, total_number = 0, 0
    for x_test,y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)    # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct为1，否则为0，将bool型转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct相加
        total_correct += int(correct)
        total_number +=x_test.shape[0]
    # 计算正确率
    acc = total_correct/total_number
    test_acc.append(acc)
    print("Test_acc: ",acc)

# 绘制loss、accuracy曲线
fig1 = plt.figure(num=1,figsize=(6,4))
plt.title("Loss Function Curve")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(train_loss_results, label="$loss$")
plt.legend()

fig2 = plt.figure(num=2,figsize=(6,4))
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()



