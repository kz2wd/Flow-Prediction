import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from models.GEN3D import GEN3D


class C04(GEN3D):

    def __init__(self, model_name="C04", checkpoint="ckpt-19", prediction_area_x=64, prediction_area_y_start=0,
                 prediction_area_y_end=32, prediction_area_z=64, channel_x_resolution=64, channel_y_resolution=128,
                 channel_z_resolution=64, channel_x_length=np.pi,
                 channel_z_length=np.pi / 2,
                 learning_rate=1e-4, in_legend_name=None):
        super().__init__(model_name=model_name, checkpoint=checkpoint, prediction_area_x=prediction_area_x,
                         prediction_area_y_start=prediction_area_y_start, prediction_area_y_end=prediction_area_y_end,
                         prediction_area_z=prediction_area_z, channel_x_resolution=channel_x_resolution,
                         channel_y_resolution=channel_y_resolution, channel_z_resolution=channel_z_resolution,
                         channel_x_length=channel_x_length,
                         channel_z_length=channel_z_length, learning_rate=learning_rate, in_legend_name=in_legend_name)

    def res_block_gen(self, model, kernal_size, filters, strides):

        gen = model
        model = layers.Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same",
                              data_format='channels_last')(model)
        model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                             shared_axes=[1, 2, 3])(model)
        model = layers.Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same",
                              data_format='channels_last')(model)
        model = layers.Add()([gen, model])

        return model

    def up_sampling_block(self, model, kernel_size, filters, strides):

        model = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(model)
        model = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                              data_format='channels_last')(model)
        model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                             shared_axes=[1, 2, 3])(model)

        return model

    def generator(self):
        inputs = keras.Input(shape=(self.prediction_area_x, 1, self.prediction_area_z, self.input_channels), name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last',
                               padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                               shared_axes=[1, 2, 3])(conv_1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):

            res_block = self.res_block_gen(res_block, 3, 64, 1)

            if index == 6 or index == 12 or index == 18 or index == 24 or index == 30:
                res_block = self.up_sampling_block(res_block, 3, 64, 1)

                prelu_1 = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(prelu_1)

        conv_2 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(
            res_block)

        conv_2 = layers.Add()([prelu_1, conv_2])

        outputs = layers.Conv3D(filters=self.output_channels, kernel_size=9, strides=1, padding="same",
                                data_format='channels_last')(conv_2)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')
        return generator

    def discriminator_block(self, model, filters, kernel_size, strides):

        model = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                              data_format='channels_last')(model)

        # Apply Leaky ReLU activation function

        model = layers.LeakyReLU(alpha=0.2)(model)

        return model

    def discriminator(self):
        inputs = keras.Input(shape=(self.prediction_area_x, self.prediction_area_y, self.prediction_area_z, self.output_channels), name='flow-input')

        # Apply a convolutional layer

        model = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(inputs)

        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha=0.2)(model)

        # Apply 7 discriminator blocks

        model = self.discriminator_block(model, 64, 3, 4)
        model = self.discriminator_block(model, 128, 3, 1)
        model = self.discriminator_block(model, 128, 3, 2)
        model = self.discriminator_block(model, 256, 3, 1)
        model = self.discriminator_block(model, 256, 3, 2)
        model = self.discriminator_block(model, 512, 3, 1)
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

    def generator_loss(self):
        def generator_loss(fake_y, y_predic, y_target):
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
        return generator_loss

    def discriminator_loss(self):
        def discriminator_loss(real_y, fake_y):
            cross_entropy = tf.keras.losses.BinaryCrossentropy()

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape) * 0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape) * 0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)

            return total_loss
        return discriminator_loss
