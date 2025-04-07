import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from A01 import A01


class A03(A01):
    def __init__(self, model_name="A03", checkpoint="ckpt-16", learning_rate=1e-4):
        super().__init__(model_name=model_name, checkpoint=checkpoint, learning_rate=learning_rate)
        self.BATCH_SIZE_PER_REPLICA = 4
        self.GLOBAL_BATCH_SIZE = 1

    def generator(self):
        inputs = keras.Input(shape=(self.prediction_area_x, 1, self.prediction_area_z, self.input_channels),
                             name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last',
                               padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                               shared_axes=[1, 2, 3])(conv_1)

        up_sampling_1 = self.up_sampling_block(prelu_1, 3, 64, 1)

        res_block = up_sampling_1

        for index in range(self.n_residual_blocks):

            res_block = self.res_block_gen(res_block, 3, 64, 1)

            if index in [6, 12, 18, 24, 30]:
                res_block = self.up_sampling_block(res_block, 3, 64, 1)

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

            scale_loss = tf.reduce_sum(loss, axis=(1, 2, 3)) / self.GLOBAL_BATCH_SIZE / self.prediction_area_x / self.prediction_area_y / self.prediction_area_z

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


if __name__ == "__main__":
    model = A03()
    model.test(1)
    # model.plot_results()
    # model.export_vts()
    u, v, w = model.get_losses(10)
    print(u, v, w)
