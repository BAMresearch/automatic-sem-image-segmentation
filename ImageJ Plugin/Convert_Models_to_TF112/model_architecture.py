import tensorflow as tf
import keras

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    x = keras.layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    if activation is None:
        return x
    x = keras.layers.Activation(activation, name=name)(x)
    return x

def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    x = keras.layers.Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False, name=name)(x)
    return x

def multi_res_block(u, inp, alpha=1.67):
    w = alpha * u
    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(w*0.167) + int(w*0.333) + int(w*0.5), 1, 1, activation=None, padding='same')
    conv3x3 = conv2d_bn(inp, int(w*0.167), 3, 3, activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv3x3, int(w*0.333), 3, 3, activation='relu', padding='same')
    conv7x7 = conv2d_bn(conv5x5, int(w*0.5), 3, 3, activation='relu', padding='same')
    out = keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = keras.layers.BatchNormalization(axis=3)(out)
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization(axis=3)(out)
    return out

def res_path(filters, length, inp):
    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization(axis=3)(out)

    for i in range(length-1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
        out = keras.layers.add([shortcut, out])
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.BatchNormalization(axis=3)(out)
    return out

def multi_res_unet(inputs, output_channels=1, conv_filters=16):
        filters = conv_filters
        mresblock1 = multi_res_block(filters, inputs)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(mresblock1)
        mresblock1 = res_path(filters, 4, mresblock1)
        mresblock2 = multi_res_block(filters * 2, pool1)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(mresblock2)
        mresblock2 = res_path(filters * 2, 3, mresblock2)
        mresblock3 = multi_res_block(filters * 4, pool2)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(mresblock3)
        mresblock3 = res_path(filters * 4, 2, mresblock3)
        mresblock4 = multi_res_block(filters * 8, pool3)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(mresblock4)
        mresblock4 = res_path(filters * 8, 1, mresblock4)
        mresblock5 = multi_res_block(filters * 16, pool4)
        up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same', output_padding=None)(mresblock5), mresblock4], axis=3)
        mresblock6 = multi_res_block(32 * 8, up6)
        up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same', output_padding=None)(mresblock6), mresblock3], axis=3)
        mresblock7 = multi_res_block(32 * 4, up7)
        up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same', output_padding=None)(mresblock7), mresblock2], axis=3)
        mresblock8 = multi_res_block(32 * 2, up8)
        up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', output_padding=None)(mresblock8), mresblock1], axis=3)
        mresblock9 = multi_res_block(filters, up9)
        if output_channels == 1:
            conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid', name='output')
        else:
            conv10 = keras.layers.Conv2D(output_channels, (1, 1), strides=1, activation=None)(mresblock9)
            conv10 = keras.layers.Activation('softmax', name='output')(conv10)
        return conv10

def create_model(input_shape=(384, 384, 1), output_channels=1, conv_filters=16):
    input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    network = multi_res_unet(input_layer, output_channels=output_channels, conv_filters=conv_filters)
    return keras.models.Model(input_layer, network)
