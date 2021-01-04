from tensorflow import reduce_sum,reshape


def dice_loss(y_true, y_pred):

    numerator = 2 * reduce_sum(y_true * y_pred, axis=(1))
    denominator = reduce_sum(y_true + y_pred, axis=(1))

    return reshape(1 - numerator / denominator, (-1, 1, 1))