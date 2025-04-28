
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from space_exploration.models.PaperBase import PaperBase
from space_exploration.models.utils import res_block_gen, up_sampling_block, discriminator_block



class A03(PaperBase):
    def __init__(self):
        super().__init__("A03", "ckpt-15", y_end=64)

    def generator(self):
        inputs = keras.Input(shape=(*self.channel.prediction_sub_space.sizes(y=1), self.input_channels),
                             name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last',
                               padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                               shared_axes=[1, 2, 3])(conv_1)

        up_sampling_1 = up_sampling_block(prelu_1, 3, 64, 1)

        res_block = up_sampling_1

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)

            if index in [6, 12, 18, 24, 30]:
                res_block = up_sampling_block(res_block, 3, 64, 1)

                up_sampling_1 = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(up_sampling_1)

        conv_2 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(
            res_block)

        up_sampling = layers.Add()([up_sampling_1, conv_2])

        # for index in range(int(np.log2(self.ny))):

        # up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

        up_sampling = layers.Conv3D(filters=256, kernel_size=3, strides=1, padding="same", data_format='channels_last')(
            up_sampling)

        outputs = layers.Conv3D(filters=self.output_channels, kernel_size=9, strides=1, padding="same",
                                data_format='channels_last')(up_sampling)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        return generator


if __name__ == "__main__":
    model = A03()
    model.test(5)