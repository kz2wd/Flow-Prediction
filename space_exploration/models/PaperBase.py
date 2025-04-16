from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from FolderManager import FolderManager
from space_exploration.models.GAN3D import GAN3D
from space_exploration.models.utils import discriminator_block
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace
from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class PaperBase(GAN3D, ABC):
    """
    A base for models coming from the paper
    """
    def __init__(self, name, checkpoint, y_end, y_start=0):
        prediction_sub_space = PredictionSubSpace(x_start=0, x_end=64, y_start=y_start, y_end=y_end, z_start=0, z_end=64)
        channel = SimulationChannel(x_length=np.pi, z_length=np.pi / 2, x_resolution=64, z_resolution=64,
                                    prediction_sub_space=prediction_sub_space,
                                    channel_data_file=FolderManager.tfrecords / "scaling.npz")
        super().__init__(name=name, checkpoint=checkpoint, channel=channel)
        self.BATCH_SIZE_PER_REPLICA = 4
        self.GLOBAL_BATCH_SIZE = 1


    def discriminator(self):

        # Define input layer

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
        def generator_loss(fake_y, y_predic, y_target):
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            adversarial_loss = tf.reshape(
                cross_entropy(
                    np.ones(fake_y.shape) - np.random.random_sample(fake_y.shape) * 0.2,
                    fake_y
                ),
                shape=(self.BATCH_SIZE_PER_REPLICA, 1, 1, 1)
            )

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            content_loss = mse(
                y_target,
                y_predic
            )

            loss = content_loss + 1e-3 * adversarial_loss

            scale_loss = tf.reduce_sum(loss, axis=(1, 2,
                                                   3)) / self.GLOBAL_BATCH_SIZE / self.prediction_area_x / self.prediction_area_y / self.prediction_area_z

            return scale_loss

        return generator_loss

    def discriminator_loss(self):
        def discriminator_loss(real_y, fake_y):
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape) * 0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape) * 0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)

            return tf.nn.compute_average_loss(total_loss, global_batch_size=self.GLOBAL_BATCH_SIZE)

        return discriminator_loss

