
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


from space_exploration.models.PaperBase import PaperBase
from space_exploration.models.utils import res_block_gen, up_sampling_block, discriminator_block


class CBase(PaperBase):
    def __init__(self, name, checkpoint, up_sampling_indices):
        super().__init__(name, checkpoint, y_end=32)
        self.up_sampling_indices = up_sampling_indices

    def generator(self):
        inputs = keras.Input(shape=(*self.channel.prediction_sub_space.sizes(y=1), self.input_channels),
                             name='wall-input')
        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last',
                               padding='same')(inputs)
        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                               shared_axes=[1, 2, 3])(conv_1)
        res_block = prelu_1

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)
            if index in self.up_sampling_indices:
                res_block = up_sampling_block(res_block, 3, 64, 1)

                prelu_1 = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(prelu_1)
        conv_2 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(
            res_block)
        conv_2 = layers.Add()([prelu_1, conv_2])

        outputs = layers.Conv3D(filters=self.output_channels, kernel_size=9, strides=1, padding="same",
                                data_format='channels_last')(conv_2)
        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')
        return generator

    def discriminator(self):
        inputs = keras.Input(
            shape=(*self.channel.prediction_sub_space.sizes(), self.output_channels),
            name='flow-input')

        # Apply a convolutional layer

        model = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(inputs)
        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha=0.2)(model)

        # Apply 7 discriminator blocks

        model = discriminator_block(model, 64, 3, 4)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        # model = discriminator_block(model, 512, 3, 2)

        # Flatten the tensor into a vector

        model = layers.Flatten()(model)

        # Apply a fully-conncted layer

        model = layers.Dense(1024)(model)

        # Apply a convolutional layer

        model = layers.LeakyReLU(alpha=0.2)(model)

        # Apply a fully-conncted layer

        model = layers.Dense(1)(model)

        # Apply a sigmoid connection function

        model = layers.Activation('sigmoid')(model)

        # Connect input and output layers

        discriminator = keras.Model(inputs=inputs, outputs=model, name='GAN3D-Discriminator')
        return discriminator
