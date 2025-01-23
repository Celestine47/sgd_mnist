# from tensorflow.keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

import tensorflow as tf
import datetime
# 用于可视化
from tensorflow.keras.utils import plot_model


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# 使用mnist数据集，编写一个函数对数据进行归一化
# 创建一个简单的keras模型使图像分为10类
def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
    tf.keras.layers.Dropout(0.2, name='layers_dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
  ])


model = create_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 将模型model的结构保存到model.png图片
plot_model(model, to_file='model.png')

# 添加 tf.keras.callback.TensorBoard 回调可确保创建和存储日志
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 另外，在每个时期启用 histogram_freq=1 的直方图计算功能（默认情况下处于关闭状态）
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

# 可视化
# tensorboard --logdir logs/fit