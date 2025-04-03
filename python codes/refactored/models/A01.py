import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from GEN3D import GEN3D


class A01(GEN3D):

    def __init__(self, root_folder, COORDINATES_FOLDER_PATH, model_name="architecture-A01", checkpoint="ckpt-15", nx=64, ny=64, nz=64, NX=64, NY=128, NZ=64, LX=np.pi, LZ=np.pi / 2,
                 learning_rate=1e-4, flow_range=slice(0, 64)):
        super().__init__(root_folder, COORDINATES_FOLDER_PATH, model_name, checkpoint, nx, ny, nz, NX, NY, NZ, LX, LZ, learning_rate, flow_range)

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

        inputs = keras.Input(shape=(self.nx, 1, self.nz, self.input_channels), name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear',
                               data_format='channels_last',
                               padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                               shared_axes=[1, 2, 3])(conv_1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):
            res_block = self.res_block_gen(res_block, 3, 64, 1)

        conv_2 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", data_format='channels_last')(
            res_block)

        up_sampling = layers.Add()([prelu_1, conv_2])

        for index in range(int(np.log2(self.ny))):
            up_sampling = self.up_sampling_block(up_sampling, 3, 256, 1)

        outputs = layers.Conv3D(filters=self.output_channels, kernel_size=9, strides=1, padding="same",
                                data_format='channels_last')(up_sampling)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        return generator

    def discriminator(self):

        def discriminator_block(model, filters, kernel_size, strides):
            model = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                                  data_format='channels_last')(model)

            # Apply Leaky ReLU activation function

            model = layers.LeakyReLU(alpha=0.2)(model)

            return model

        # Define input layer

        inputs = keras.Input(shape=(self.nx, self.ny, self.nz, self.output_channels), name='flow-input')

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
        model = discriminator_block(model, 512, 3, 2)

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
    model = A01("../../../", "../../channel coordinates")
    model.test()
