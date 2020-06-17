"""
References:
https://arxiv.org/abs/2003.13911
https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/losses.py
"""

import tensorflow as tf
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model


class ProxyAnchor(Layer):

    def __init__(self,
                 units,
                 kernel_initializer="he_normal",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(ProxyAnchor, self).__init__(**kwargs)

        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        last_dim = input_shape[1]
        self.kernel = self.add_weight(  # pylint: disable=attribute-defined-outside-init
            name="kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)

        super(ProxyAnchor, self).build(input_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ
        normalized_embeddings = tf.nn.l2_normalize(inputs, axis=1)
        normalized_proxies = tf.nn.l2_normalize(self.kernel, axis=0)
        cosine_similarity = tf.matmul(normalized_embeddings, normalized_proxies)
        return cosine_similarity

    def get_config(self):
        config = {
            "units":
                self.units,
            "kernel_initializer":
                initializers.serialize(self.kernel_initializer),
            "kernel_regularizer":
                regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint":
                constraints.serialize(self.kernel_constraint)
        }
        base_config = super(ProxyAnchor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def proxy_anchor_loss(y_true, y_pred, margin=0.1, alpha=32):
    cosine_similarity = y_pred
    class_num = cosine_similarity.get_shape().as_list()[1]
    P_one_hot = tf.one_hot(indices=tf.argmax(y_true, axis=1),
                           depth=class_num,
                           on_value=None,
                           off_value=None)
    N_one_hot = 1.0 - P_one_hot

    pos_exp = tf.exp(-alpha * (cosine_similarity - margin))
    neg_exp = tf.exp(alpha * (cosine_similarity + margin))

    P_sim_sum = tf.reduce_sum(pos_exp * P_one_hot, axis=0)
    N_sim_sum = tf.reduce_sum(neg_exp * N_one_hot, axis=0)

    num_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(P_one_hot, axis=0),
                                              dtype=tf.dtypes.float32)

    pos_term = tf.reduce_sum(tf.math.log(1.0 + P_sim_sum)) / num_valid_proxies
    neg_term = tf.reduce_sum(tf.math.log(1.0 + N_sim_sum)) / class_num
    loss = pos_term + neg_term

    return loss


def example(embedding_size=256, class_num=10):
    # Define the input and output tensors
    input_tensor = Input(shape=(embedding_size))
    output_tensor = ProxyAnchor(units=class_num)(input_tensor)

    # Define the model and compile it
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss=proxy_anchor_loss, optimizer="adam")


if __name__ == "__main__":
    example()
