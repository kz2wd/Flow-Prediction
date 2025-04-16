import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from FolderManager import FolderManager
from space_exploration.models.GAN3D import GAN3D
from space_exploration.models.PaperBase import PaperBase
from space_exploration.models.utils import res_block_gen, up_sampling_block, discriminator_block
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace
from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class A01(PaperBase):
    def __init__(self):
        super().__init__(name="A01", checkpoint="ckpt-15", y_end=64)

    def generator(self):

        inputs = keras.Input(shape=(*self.channel.prediction_sub_space.sizes(y=1), self.input_channels),
                             name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear',
                               data_format='channels_last',
                               padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                               shared_axes=[1, 2, 3])(conv_1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):
            res_block = res_block_gen(res_block, 3, 64, 1)

        conv_2 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(
            res_block)

        up_sampling = layers.Add()([prelu_1, conv_2])

        for index in range(int(np.log2(self.channel.prediction_sub_space.y_size))):
            up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

        outputs = layers.Conv3D(filters=self.output_channels, kernel_size=9, strides=1, padding="same",
                                data_format='channels_last')(up_sampling)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        return generator


    def generator_loss(self):

        def generator_loss_intern(fake_y, y_predic, y_target):
            cross_entropy = tf.keras.losses.BinaryCrossentropy()

            adversarial_loss = cross_entropy(
                np.ones(fake_y.shape) - np.random.random_sample(fake_y.shape) * 0.2,
                fake_y
            )

            content_loss = tf.keras.losses.MSE(
                y_target,
                y_predic
            )

            loss = content_loss + 1e-3 * adversarial_loss

            return loss

        return generator_loss_intern

    def discriminator_loss(self):

        # Define discriminator loss as a function to be returned for its later use during the training

        def discriminator_loss_intern(real_y, fake_y):
            cross_entropy = tf.keras.losses.BinaryCrossentropy()

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape) * 0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape) * 0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)

            return total_loss

        return discriminator_loss_intern



if __name__ == "__main__":
    model = A01()
    model.test(5)