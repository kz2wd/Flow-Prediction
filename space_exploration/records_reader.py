import tensorflow as tf


class RecordsFeatures:
        features = {
            'i_sample': tf.io.FixedLenFeature([], tf.int64),
            'nx': tf.io.FixedLenFeature([], tf.int64),
            'ny': tf.io.FixedLenFeature([], tf.int64),
            'nz': tf.io.FixedLenFeature([], tf.int64),
            'x': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'y': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'z': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_u': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_v': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_w': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        }
