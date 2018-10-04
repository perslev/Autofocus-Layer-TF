import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils


class Autofocus3D(Conv3D):
    """
    Implements the Autofocus layer as described in
    https://link.springer.com/content/pdf/10.1007%2F978-3-030-00931-1_69.pdf
    """
    def __init__(self, dilations, filters, kernel_size, activation=None,
                 attention_kernel_size=(3, 3, 3), attention_filters=None,
                 attention_activation=tf.nn.relu, use_bn=True, **kwargs):
        """
        dilations : (list of ints, list of 2-tuples of ints)
            List of dilation rates to process on the input in parallel
        filters: (int)
            Number of filters of the dilated convolutions
        kernel_size: (int, 2-tuple of intes)
            Kernel dim in dilated convolution layers
        activation: (string, func)
            Activation function to apply to final Autofocus layer output
        attention_kernel_size : (int, 2-tuple of intes)
            Kernel dim in attention conv layer 1
        attention_filters : (int, None)
            Number of filters in attention conv layer 1, uses 'filters' // 2
            if not specified
        attention_activation: (func)
            TF activation function to apply after attention conv layer 1
        use_bn : (bool)
            Apply batch normalization after dilated convolutions
        kwargs : (dict)
            Passed to tf.keras.layers.Conv3D, for instance...
            -  kernel_initializer='glorot_uniform'
            -  bias_initializer='zeros'
            -  kernel_regularizer=None
            -  bias_regularizer=None
            -  activity_regularizer=None
            -  kernel_constraint=None
            -  bias_constraint=None

            NOTE: These passed parameters are currently used for all
            convolutions across the layer - both in the dilated network
            and attention layer
        """
        # Assert parameters passed are compatible
        if kwargs.get("padding") and kwargs["padding"].upper() != "SAME":
            raise NotImplementedError("Only implemented for padding 'SAME'")
        if kwargs.get("dilation_rate"):
            raise ValueError("Should not pass arguments to 'dilation_rate'. "
                             "Pass a list to 'dilations' instead.")
        kwargs["dilation_rate"] = (1, 1, 1)
        kwargs["padding"] = "SAME"

        # Init base tf 3D Conv class
        super(Autofocus3D, self).__init__(filters=filters,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          **kwargs)

        # Use batch norm in dilation network?
        self.use_bn = use_bn

        # Attributes for attention network
        self.attention_filters = attention_filters or self.filters // 2
        self.attention_kernel_size = conv_utils.normalize_tuple(
                        attention_kernel_size, self.rank, 'kernel_size')
        self.attention_activation = attention_activation

        # Dilations
        self.dilations = [conv_utils.normalize_tuple(d, self.rank, 'dilation_rate')
                          for d in dilations]
        self.conv_ops = []
        self.attention_ops = []

    def build_attention(self, input_shape, input_dim):
        """
        Define trainable kernel and bias variables for attention layer
        Prepare conv operations
        """
        # Layer 1, standard convolution layer
        l1_kernel_shape = self.attention_kernel_size + (input_dim, self.attention_filters)
        self.att_K1 = self.add_weight(
            name='attention_kernel_L1',
            shape=l1_kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.att_B1 = self.add_weight(
                name='attention_bias_L1',
                shape=(self.attention_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.att_B1 = None

        # Layer 2, 1x1 convolution layer with 1 filter pr. dilation
        l2_kernel_shape = (1, 1, 1) + (self.attention_filters, len(self.dilations))
        self.att_K2 = self.add_weight(
            name='attention_kernel_L2',
            shape=l2_kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.att_B2 = self.add_weight(
                name='attention_bias_L2',
                shape=(len(self.dilations),),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.att_B2 = None

        # Prepare conv operations
        self.att_conv_1 = nn_ops.Convolution(
            input_shape,
            filter_shape=self.att_K1.get_shape(),
            dilation_rate=(1, 1, 1),
            strides=(1, 1, 1),
            padding="SAME",
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        self.att_conv_2 = nn_ops.Convolution(
            tf.TensorShape(input_shape[:-1].as_list() + [self.attention_filters]),
            filter_shape=self.att_K2.get_shape(),
            dilation_rate=(1, 1, 1),
            strides=(1, 1, 1),
            padding="SAME",
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))

    def build_dilated_conv(self, input_shape, input_dim):
        """
        Define trainable kernel and bias variables for dilated conv layers
        Prepare conv operations
        """
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        for dilation in self.dilations:
            convolution_op = nn_ops.Convolution(
                input_shape,
                filter_shape=self.kernel.get_shape(),
                dilation_rate=dilation,
                strides=self.strides,
                padding="SAME",
                data_format=conv_utils.convert_data_format(self.data_format,
                                                           self.rank + 2))
            self.conv_ops.append(convolution_op)

    def build(self, input_shape):
        """
        Reimplementation of tf.keras.layers.Conv3D creating multiple conv ops
        with varying degree of dilation with shared weights
        """
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1
        if input_shape[self.channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[self.channel_axis])

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={self.channel_axis: input_dim})

        # Prepare weights and conv ops for dilated conv layers
        with tf.name_scope("dilated_conv_weights"):
            self.build_dilated_conv(input_shape, input_dim)

        # Prepare weights and conv ops for attention mechanism
        with tf.name_scope("attention_weights"):
            self.build_attention(input_shape, input_dim)

        # Set build flag, needed if called directly
        self.built = True

    def call(self, x, **kwargs):
        """
        Build computation graph
        Applies a convolution operation to 'x' for each dilation specified
        in 'self.dilations' with shared kernel and bias weights
        Processes 'x' through the attention layer mechanism as
        att = softmax(conv3D(relu(conv3D(x))))
        """
        # Evaluate the dilated conv ops
        with tf.name_scope("dilated_conv_layers"):
            outs = []
            for i, op in enumerate(self.conv_ops):
                with tf.name_scope("dilation_rate_%i_%i_%i" % self.dilations[i]):
                    # Perform convolution with the same kernel across dilations
                    out = op(x, self.kernel)

                    if self.use_bias:
                        # Add bias if specified
                        cf = self.data_format == 'channels_first'
                        out = nn.bias_add(out, self.bias,
                                          data_format='NCHW' if cf else "NHWC")
                    if self.use_bn:
                        # Add BN layer if specified
                        bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis)
                        out = bn(out)
                    if self.activation is not None:
                        # Apply activation if specified
                        out = self.activation(out)

                outs.append(out)
            outs = tf.stack(outs, -1)

        # Compute attention mechanism
        with tf.name_scope("attention_mechanism"):
            # Layer 1 (standard conv, any number of filters)
            at1 = self.att_conv_1(x, self.att_K1)
            if self.use_bias:
                at1 = nn.bias_add(at1, self.att_B1,
                                  data_format='NCHW' if cf else "NHWC")

            # Layer 2 (1x1 conv, 1 feature map pr. dilation)
            at2 = self.att_conv_2(self.attention_activation(at1), self.att_K2)
            if self.use_bias:
                at2 = nn.bias_add(at2, self.att_B2,
                                  data_format='NCHW' if cf else "NHWC")
            at_map = tf.nn.softmax(at2, name="attention_map")

        # Compute attention weighted map
        with tf.name_scope("weight_map"):
            at_map = tf.expand_dims(at_map, axis=-2)
            output = tf.reduce_sum(tf.multiply(at_map, outs), axis=-1)

        if self.activation is not None:
            # Activation function on output if specified
            return self.activation(output)
        return output
