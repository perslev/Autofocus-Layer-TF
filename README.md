# Autofocus-Layer-TF
TensorFlow implementation of the autofocus layer as described in 'Autofocus Layer for Semantic Segmentation' (https://arxiv.org/pdf/1805.08403.pdf), MICCAI 2018.

Paper authors:
Yao Qin, Konstantinos Kamnitsas, Siddharth Ancha, Jay Nanavati, Garrison Cottrell, Antonio Criminisi, and Aditya Nori

Note: I am not among the authors, there may be differences between this implementation and the one suggested by the authors.

## Use
The layer can be used as a drop-in replacement of the tf.keras.layers.Conv2D layer:

'''python
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

'''
