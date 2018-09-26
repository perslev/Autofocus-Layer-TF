from autofocus import Autofocus2D
import tensorflow as tf
import numpy as np

# Dilation rates, here 4 parallel conv applications
dilations = [1, 2, (3, 3), (4, 6)]

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[128, 128, 3], batch_size=64),
    Autofocus2D(dilations, 
                filters=20, 
                kernel_size=(5, 5), 
                activation='relu',
                attention_activation=tf.nn.relu,
                attention_filters=10,
                attention_kernel_size=3,
                use_bn=True,
                use_bias=True),
    tf.keras.layers.Conv2D(10, 3, activation="relu")
    # etc....
])

# Build model by passing random data...
in_ = tf.constant(np.random.rand(64, 128, 128, 3).astype(np.float32))
model(in_)

print(model.summary())

# Write graph
writer = tf.summary.FileWriter(logdir="./")
with tf.Session() as s:
    writer.add_graph(s.graph)
    writer.flush()
