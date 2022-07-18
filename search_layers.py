import tensorflow as tf

from tensorflow.keras import Model, Sequential

from tensorflow.keras.layers import (Conv1D, SeparableConv1D, MaxPool1D, BatchNormalization,
                                     AveragePooling1D, ReLU, LayerNormalization, MultiHeadAttention, 
                                     Add, Dense, Dropout, Attention, Input, Dense, Flatten, Conv1D,
                                     MaxPool1D, Embedding, GlobalAveragePooling1D, Softmax,
                                     Reshape, TimeDistributed, Rescaling)


def regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def kernel_init(seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal(seed)


class Zero(tf.keras.layers.Layer):
    """Zero"""
    def __init__(self, name='Zero', **kwargs):
        super(Zero, self).__init__(name=name, **kwargs)

    def call(self, x):
        return x * 0.

class MultiHeadEncoderAttention(tf.keras.layers.Layer):
    def __init__(self, units, wd,
                 name='EncoderAttention', **kwargs):
        super(MultiHeadEncoderAttention, self).__init__(name=name, **kwargs)
        self.ln = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(num_heads=4, key_dim=units*2 , dropout=0.2,
                                        kernel_initializer=kernel_init(),
                                        kernel_regularizer=regularizer(wd))

    def call(self, x):
        a = self.ln(x)
        a = self.mha(a,a)
        return a

class MultiHeadDecoderAttention(tf.keras.layers.Layer):
    def __init__(self, units, wd,
                 name='DecoderAttention', **kwargs):
        super(MultiHeadDecoderAttention, self).__init__(name=name, **kwargs)
        self.h = 4
        self.units = units
        self.wd = wd

    def build(self, input_shape):
        # input_shape= self.units
        query_shape = input_shape 
        key_shape = input_shape
        value_shape = input_shape
        d_model = query_shape[-1]

        # Note: units can be anything, but this is what the paper does
        units = d_model * 2

        self.ln = LayerNormalization(epsilon=1e-6)

        self.layersQ = []
        for _ in range(self.h):
            layer = Dense(units, activation=None, use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
            layer.build(query_shape)
            self.layersQ.append(Sequential([layer, Dropout(0.2)]))

        self.layersK = []
        for _ in range(self.h):
            layer = Dense(units, activation=None, use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
            layer.build(key_shape)
            self.layersK.append(Sequential([layer, Dropout(0.2)]))

        self.layersV = []
        for _ in range(self.h):
            layer = Dense(units, activation=None, use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
            layer.build(value_shape)
            self.layersV.append(Sequential([layer, Dropout(0.2)]))

        self.attention = Attention(use_scale=False, causal=True)

        self.out = Dense(d_model, activation=None, use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
        self.out.build((query_shape[0], self.h * units))

    def call(self, input):
        input_norm = self.ln(input)
        query  = input_norm
        key  = input_norm
        value = input_norm

        q = [layer(query) for layer in self.layersQ]
        k = [layer(key) for layer in self.layersK]
        v = [layer(value) for layer in self.layersV]

        # Head is in multi-head, just like the paper
        head = [self.attention([q[i], k[i], v[i]]) for i in range(self.h)]

        out = self.out(tf.concat(head, -1))
        return out

# class MultiHeadDecoderAttention(tf.keras.layers.Layer):
#     """Multi-head decoder attention."""
#     def __init__(self, units,**kwargs):
#         super(MultiHeadDecoderAttention, self).__init__(**kwargs)
#         self.num_heads = 4
#         self.num_hiddens = units*2*self.num_heads
#         self.ln = LayerNormalization(epsilon=1e-6)
#         self.attention = Attention(use_scale=False, causal=True)
#         self.W_q = Dense(self.num_hiddens, use_bias=False)
#         self.W_k = Dense(self.num_hiddens, use_bias=False)
#         self.W_v = Dense(self.num_hiddens, use_bias=False)
#         self.W_o = Dense(units, use_bias=False)

#     def build(self, input_shape):
#          self.W_o.build((input_shape[0], self.num_hiddens))

#     def transpose_qkv(self, X, num_heads):
#         """Transposition for parallel computation of multiple attention heads."""
#         X = tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1], num_heads, -1))
#         X = tf.transpose(X, (0, 2, 1, 3))
#         return tf.reshape(X, (-1, tf.shape(X)[2], tf.shape(X)[3]))

#     def transpose_output(self, X, num_heads):
#         """Reverse the operation of `transpose_qkv`."""
#         X = tf.reshape(X,(-1, num_heads, tf.shape(X)[1], tf.shape(X)[2]))
#         X = tf.transpose(X, (0, 2, 1, 3))
#         return tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1], -1))

#     def call(self, input):
#         input_norm = self.ln(input)
#         queries  = input_norm
#         keys  = input_norm
#         values = input_norm
#         queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
#         keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
#         values = self.transpose_qkv(self.W_v(values), self.num_heads)

#         output = self.attention([queries, keys, values])
#         output_concat = self.transpose_output(output, self.num_heads)
#         return self.W_o(output_concat)


class Densely(tf.keras.layers.Layer):
    def __init__(self, units, wd,
                 name='Densely', **kwargs):
        super(Densely, self).__init__(name=name, **kwargs)
        self.op = Sequential([
            Dense(units=units*2,
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd),
                   use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            Dense(units=units,
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd),
                   use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2)
            ])

    def call(self, x):
        return self.op(x)

class Conv(tf.keras.layers.Layer):
    def __init__(self, ch_out, k, wd, padding='same',
                 name='Conv', **kwargs):
        super(Conv, self).__init__(name=name, **kwargs)
        self.op = Sequential([
            Conv1D(filters=ch_out*2, kernel_size=k, strides=1,
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd),
                   padding=padding, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            Conv1D(filters=ch_out, kernel_size=1, strides=1,
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd),
                   padding=padding, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),])

    def call(self, x):
        return self.op(x)

class SepConv(tf.keras.layers.Layer):
    """Separable Conv"""
    def __init__(self, ch_out, k, wd, name='SepConv',
                 **kwargs):
        super(SepConv, self).__init__(name=name, **kwargs)
        self.op = Sequential([
            SeparableConv1D(filters=ch_out*2, kernel_size=k, strides=1,
                            depthwise_initializer=kernel_init(),
                            pointwise_initializer=kernel_init(),
                            depthwise_regularizer=regularizer(wd),
                            pointwise_regularizer=regularizer(wd),
                            padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            SeparableConv1D(filters=ch_out, kernel_size=1, strides=1,
                            depthwise_initializer=kernel_init(),
                            pointwise_initializer=kernel_init(),
                            depthwise_regularizer=regularizer(wd),
                            pointwise_regularizer=regularizer(wd),
                            padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),])

    def call(self, x):
        return self.op(x)

class DilConv(tf.keras.layers.Layer):
    """Dilated Conv"""
    def __init__(self, ch_out, k, d, wd, name='DilConv',
                 **kwargs):
        super(DilConv, self).__init__(name=name, **kwargs)
        self.op = Sequential([
            SeparableConv1D(filters=ch_out*2, kernel_size=k, strides=1,
                            depthwise_initializer=kernel_init(),
                            pointwise_initializer=kernel_init(),
                            depthwise_regularizer=regularizer(wd),
                            pointwise_regularizer=regularizer(wd),
                            dilation_rate=d, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            SeparableConv1D(filters=ch_out, kernel_size=1, strides=1,
                            depthwise_initializer=kernel_init(),
                            pointwise_initializer=kernel_init(),
                            depthwise_regularizer=regularizer(wd),
                            pointwise_regularizer=regularizer(wd),
                            dilation_rate=d, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),])

    def call(self, x):
        return self.op(x)


class Identity(tf.keras.layers.Layer):
    """Identity"""
    def __init__(self, name='Identity', **kwargs):
        super(Identity, self).__init__(name=name, **kwargs)

    def call(self, x):
        return x 
