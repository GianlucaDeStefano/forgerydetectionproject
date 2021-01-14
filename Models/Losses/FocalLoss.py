import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2):
    """
    Implementation of the focal loss function as found here:
    https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
    See the articles for more details about how it works

    :param alpha:
    :param gamma:
    :return:
    """
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
                    weight_a + weight_b) + logits * weight_b

    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    return loss