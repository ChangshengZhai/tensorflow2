import tensorflow as tf

w = tf.Variable(tf.constant (5, dtype=tf.float32))
lr = 0.2  #Initialize the learning rate
epoch = 40
for epoch in range(epoch):
    with tf.GradientTape() as tape:  #with结构到grads框起了梯度的计算过程
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  #gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  #.assign_sub 对变量做自减 即：w -= lr * grads
    print("After %s epoch, w is %f, loss is %f"%(epoch, w.numpy(),loss))

