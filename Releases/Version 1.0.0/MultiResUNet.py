from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add


class MultiResUNet:

    @staticmethod
    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        '''
        2D Convolutional layers

        Arguments:
            x {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(1, 1)})
            activation {str} -- activation function (default: {'relu'})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''

        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=3, scale=False)(x)

        if(activation == None):
            return x

        x = Activation(activation, name=name)(x)

        return x


    @staticmethod
    def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
        '''
        2D Transposed Convolutional layers

        Arguments:
            x {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(2, 2)})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''

        x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=3, scale=False)(x)

        return x


    @staticmethod
    def MultiResBlock(U, inp, alpha = 1.67):
        '''
        MultiRes Block

        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        '''

        W = alpha * U

        shortcut = inp

        shortcut = MultiResUNet.conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                             int(W*0.5), 1, 1, activation=None, padding='same')

        conv3x3 = MultiResUNet.conv2d_bn(inp, int(W*0.167), 3, 3,
                            activation='relu', padding='same')

        conv5x5 = MultiResUNet.conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                            activation='relu', padding='same')

        conv7x7 = MultiResUNet.conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                            activation='relu', padding='same')

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
        out = BatchNormalization(axis=3)(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

        return out


    @staticmethod
    def ResPath(filters, length, inp):
        '''
        ResPath

        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        '''


        shortcut = inp
        shortcut = MultiResUNet.conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = MultiResUNet.conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

        for i in range(length-1):

            shortcut = out
            shortcut = MultiResUNet.conv2d_bn(shortcut, filters, 1, 1,
                                 activation=None, padding='same')

            out = MultiResUNet.conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization(axis=3)(out)

        return out


    @staticmethod
    # def MultiResUnet(height, width, n_channels):
    def MultiResUnet(inputs, outputChannels=1, convFilters=16):
        '''
        MultiResUNet

        Arguments:
            height {int} -- height of image
            width {int} -- width of image
            n_channels {int} -- number of channels in image

        Returns:
            [keras model] -- MultiResUNet model
        '''


        # inputs = Input((height, width, n_channels))
        FILTERS = convFilters

        mresblock1 = MultiResUNet.MultiResBlock(FILTERS, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
        mresblock1 = MultiResUNet.ResPath(FILTERS, 4, mresblock1)

        mresblock2 = MultiResUNet.MultiResBlock(FILTERS*2, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
        mresblock2 = MultiResUNet.ResPath(FILTERS*2, 3, mresblock2)

        mresblock3 = MultiResUNet.MultiResBlock(FILTERS*4, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
        mresblock3 = MultiResUNet.ResPath(FILTERS*4, 2, mresblock3)

        mresblock4 = MultiResUNet.MultiResBlock(FILTERS*8, pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
        mresblock4 = MultiResUNet.ResPath(FILTERS*8, 1, mresblock4)

        mresblock5 = MultiResUNet.MultiResBlock(FILTERS*16, pool4)

        up6 = concatenate([Conv2DTranspose(
            FILTERS*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
        mresblock6 = MultiResUNet.MultiResBlock(32*8, up6)

        up7 = concatenate([Conv2DTranspose(
            FILTERS*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
        mresblock7 = MultiResUNet.MultiResBlock(32*4, up7)

        up8 = concatenate([Conv2DTranspose(
            FILTERS*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
        mresblock8 = MultiResUNet.MultiResBlock(32*2, up8)

        up9 = concatenate([Conv2DTranspose(FILTERS, (2, 2), strides=(
            2, 2), padding='same')(mresblock8), mresblock1], axis=3)
        mresblock9 = MultiResUNet.MultiResBlock(FILTERS, up9)

        if outputChannels == 1:
            conv10 = MultiResUNet.conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid', name='output')
            # conv10 = MultiResUNet.conv2d_bn(mresblock9, 1, 1, 1, activation=None)
        else:
            conv10 = Conv2D(outputChannels, (1, 1), strides=1, activation=None)(mresblock9)
            conv10 = Activation('softmax', name='output')(conv10)

        # model = Model(inputs=[inputs], outputs=[conv10])
        # return model

        return conv10


    @staticmethod
    def main():

        # Define the model

        model = MultiResUNet.MultiResUnet(128, 128, 3)
        print(model.summary())



    if __name__ == '__main__':
        main()
