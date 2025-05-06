# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.layers as layers
#
#
# def res_block_gen(model, kernal_size, filters, strides):
#     gen = model
#     model = layers.Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same",
#                           data_format='channels_last')(model)
#     model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
#                          shared_axes=[1, 2, 3])(model)
#     model = layers.Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same",
#                           data_format='channels_last')(model)
#     model = layers.Add()([gen, model])
#
#     return model
#
#
# def up_sampling_block(model, kernel_size, filters, strides):
#     model = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(model)
#     model = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
#                           data_format='channels_last')(model)
#     model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
#                          shared_axes=[1, 2, 3])(model)
#
#     return model
#
#
# def discriminator_block(model, filters, kernel_size, strides):
#     model = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
#                           data_format='channels_last')(model)
#
#     # Apply Leaky ReLU activation function
#
#     model = layers.LeakyReLU(alpha=0.2)(model)
#
#     return model