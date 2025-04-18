import os
import re
import sys
import time
from abc import ABC, abstractmethod
from plistlib import loads

import h5py
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error

from FolderManager import FolderManager
from space_exploration.simulation_channel import SimulationChannel



class GAN3D(ABC):
    def __init__(self, name, checkpoint, channel: SimulationChannel,
                 n_residual_blocks=32, input_channels=3, output_channels=3, learning_rate=1e-4, ):

        FolderManager.init(self)  # ensure all related folders are created
        self.checkpoint = checkpoint

        self.channel: SimulationChannel = channel

        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name
        self.n_residual_blocks = n_residual_blocks

    @abstractmethod
    def generator(self):
        pass

    @abstractmethod
    def discriminator(self):
        pass

    @abstractmethod
    def generator_loss(self):
        pass

    @abstractmethod
    def discriminator_loss(self):
        pass

    def generator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def discriminator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def generate_datasets(self, target_folder, batch_size):
        records = []
        records_folder = FolderManager.tfrecords / target_folder
        for file in os.listdir(records_folder):
            res = re.search(".tfrecord", file)
            if res:
               records.append(records_folder / file)
        raw_dataset = tf.data.TFRecordDataset(records)
        parsed_dataset = raw_dataset.map(self.channel.tf_parser)
        DATASET_SIZE = tf.data.experimental.cardinality(parsed_dataset).numpy()
        train_size = int(0.7 * DATASET_SIZE)
        val_size = int(0.15 * DATASET_SIZE)
        test_size = int(0.15 * DATASET_SIZE)
        parsed_dataset.shuffle(buffer_size=10000)
        train_dataset = parsed_dataset.take(train_size)
        train_dataset = train_dataset.batch(batch_size=batch_size)
        test_dataset = parsed_dataset.skip(train_size)
        val_dataset = test_dataset.skip(test_size)
        val_dataset = val_dataset.batch(batch_size=batch_size)
        test_dataset = test_dataset.take(test_size)
        test_dataset = test_dataset.batch(batch_size=batch_size)
        return train_dataset, val_dataset, test_dataset

    def generate_pipeline_training(self, folder="test", validation_split:float = 1.0, shuffle_buffer=200, batch_size=4,
                                   n_prefetch=4):

        tfr_path = FolderManager.tfrecords / folder
        tfr_files = sorted(
            [os.path.join(tfr_path, f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path, f))])
        regex = re.compile(f'.tfrecords')
        tfr_files = ([string for string in tfr_files if re.search(regex, string)])

        n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])
        n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
        cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
        tot_samples_per_ds = sum(n_samples_per_tfr)
        # n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-3:])
        n_tfr_loaded_per_ds = 119  # some were corrupted
        # n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-4:])
        tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_tfr_loaded_per_ds]
        # tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:4]) <= n_tfr_loaded_per_ds]

        n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
        n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

        (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

        if samples_train_left > 0:
            n_files_train += 1

        tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_files_train]
        # tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:4]) <= n_files_train]
        n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1

        if sum([int(s.split('.')[-2][-2:]) for s in tfr_files_train]) != n_samp_train:

            shared_tfr = tfr_files_train[-1]
            tfr_files_valid = [shared_tfr]
        else:

            shared_tfr = ''
            tfr_files_valid = list()

        tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])
        tfr_files_valid = sorted(tfr_files_valid)

        shared_tfr_out = tf.constant(shared_tfr)
        n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
        n_samples_loaded_per_tfr = list()

        if n_tfr_loaded_per_ds > 1:

            n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds - 1])
            n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds - 2])

        else:

            n_samples_loaded_per_tfr.append(tot_samples_per_ds)

        n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)

        # tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
        tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

        if n_tfr_left > 1:

            samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left - 2]
            n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left - 1]

        else:

            samples_train_shared = samples_train_left
            n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

        # tfr_files_train_ds = tfr_files_train_ds.interleave(
        #    lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)),
        #    cycle_length=16,
        #    num_parallel_calls=tf.data.experimental.AUTOTUNE
        # )

        tfr_files_val_ds = tfr_files_val_ds.interleave(
            lambda x: tf.data.TFRecordDataset(x).skip(samples_train_shared).take(
                n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x,
                                                                              shared_tfr_out) else tf.data.TFRecordDataset(
                x).take(tf.gather(n_samples_loaded_per_tfr,
                                  tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3], sep='-')[0],
                                                       tf.int32) - 1)),
            cycle_length=16,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset_train = tfr_files_val_ds.map(lambda x: self.channel.tf_parser(x),
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_train = dataset_train.shuffle(shuffle_buffer)
        dataset_train = dataset_train.batch(batch_size=batch_size)
        dataset_train = dataset_train.prefetch(n_prefetch)

        # dataset_valid = dataset_valid.shuffle(shuffle_buffer)

        return dataset_train

    def test(self, test_sample_amount=50):
        physical_devices = tf.config.list_physical_devices('GPU')
        available_GPUs = len(physical_devices)
        print('Using TensorFlow version: ', tf.__version__, ', GPU:', available_GPUs)
        print('Using Keras version: ', tf.keras.__version__)
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        dataset_valid = self.generate_pipeline_training(validation_split=1, batch_size=1)

        generator = self.generator()
        discriminator = self.discriminator()
        generator_loss = self.generator_loss()
        discriminator_loss = self.discriminator_loss()
        generator_optimizer = self.generator_optimizer()
        discriminator_optimizer = self.discriminator_optimizer()

        # Generate checkpoint object to track the generator and discriminator architectures and optimizers

        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        ckpt_file = FolderManager.checkpoints(self) / self.checkpoint
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator
        )
        checkpoint.restore(ckpt_file)

        # samples, stream wise resolution, wall normal wise resolution, span wise resolution, velocities.
        # velocities -> 0, u, stream wise | 1, v, wall normal wise | 2, w, spawn wise
        self.x_target_normalized = np.zeros((test_sample_amount, *self.channel.prediction_sub_space.sizes(y=1), 3),
                                            np.float32)
        self.y_target_normalized = np.zeros(
            (test_sample_amount, *self.channel.prediction_sub_space.sizes(), 3), np.float32)
        self.y_predict_normalized = np.zeros(
            (test_sample_amount, *self.channel.prediction_sub_space.sizes(), 3), np.float32)


        itr = iter(dataset_valid)
        try:
            for i in range(test_sample_amount):
                if float.is_integer(i / 50):
                    print(i)
                # print(idx)
                x, y = next(itr)

                self.x_target_normalized[i] = x.numpy()
                self.y_target_normalized[i] = y.numpy()

                self.y_predict_normalized[i] = generator(np.expand_dims(self.x_target_normalized[i], axis=0),
                                                         training=False)
        except Exception as e:
            print(e)

        mse = np.mean((self.y_target_normalized - self.y_predict_normalized) ** 2)
        print(mse)
        print(f'Predict mean: {np.mean(self.y_predict_normalized)}, std: {np.std(self.y_predict_normalized)}')

    def save(self):
        save_dataset = FolderManager.predictions_file(self)
        with h5py.File(save_dataset, 'w') as f:
            f.create_dataset('y_pred', data=self.y_predict_normalized, compression='gzip')
            f.create_dataset('y_target', data=self.y_target_normalized, compression='gzip')

    def load(self, amount):
        save_dataset = FolderManager.predictions_file(self)
        with h5py.File(save_dataset, 'r') as f:
            self.y_predict_normalized = f['y_pred'][:amount, ...]
            self.y_target_normalized = f['y_target'][:amount, ...]

    def lazy_test(self, amount):
        try:
            self.load(amount)
        except (FileNotFoundError, KeyError):
            self.test(amount)
            self.save()

    def benchmark(self):
        benchmark_dataset = FolderManager.benchmark_file(self)
        # MSE along Y
        mse = np.mean((self.y_predict_normalized - self.y_predict_normalized) ** 2, axis=(0, 1, 3))
        u_mse = np.mean((self.y_target_normalized[..., 0] - self.y_predict_normalized[..., 0]) ** 2, axis=(0, 1, 3))
        v_mse = np.mean((self.y_target_normalized[..., 1] - self.y_predict_normalized[..., 1]) ** 2, axis=(0, 1, 3))
        w_mse = np.mean((self.y_target_normalized[..., 2] - self.y_predict_normalized[..., 2]) ** 2, axis=(0, 1, 3))
        with h5py.File(benchmark_dataset, 'w') as f:
            f.create_dataset('total_mse_y_wise', data=mse, compression='gzip')
            f.create_dataset('u_mse_y_wise', data=u_mse, compression='gzip')
            f.create_dataset('v_mse_y_wise', data=v_mse, compression='gzip')
            f.create_dataset('w_mse_y_wise', data=w_mse, compression='gzip')


    def train(self, epochs, saving_freq, batch_size):
        @tf.function
        def train_step(x_target, y_target, generator, discriminator,
                       generator_loss_fn, discriminator_loss_fn,
                       generator_optimizer, discriminator_optimizer):
            """
            Perform one training step for both generator and discriminator.
            """
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Forward pass through the generator
                y_pred = generator(x_target, training=True)

                # Discriminator predictions for real and fake data
                real_output = discriminator(y_target, training=True)
                fake_output = discriminator(y_pred, training=True)

                # Compute losses
                gen_loss = generator_loss_fn(fake_output, y_pred, y_target)
                disc_loss = discriminator_loss_fn(real_output, fake_output)

            # Compute gradients
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            # Apply gradients
            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            return gen_loss, disc_loss

        @tf.function
        def valid_step(x_target, y_target, generator, discriminator,
                       generator_loss_fn, discriminator_loss_fn):
            """
            Perform one validation step (no gradients applied).
            """
            # Forward pass only (no gradient computation)
            y_pred = generator(x_target, training=False)

            real_output = discriminator(y_target, training=False)
            fake_output = discriminator(y_pred, training=False)

            # Compute losses
            gen_loss = generator_loss_fn(fake_output, y_pred, y_target)
            disc_loss = discriminator_loss_fn(real_output, fake_output)

            return gen_loss, disc_loss

        dataset_train, dataset_test, dataset_valid = self.generate_datasets("test", batch_size)
        dataset_train = dataset_train.take(10)

        generator = self.generator()
        discriminator = self.discriminator()
        generator_loss = self.generator_loss()
        discriminator_loss = self.discriminator_loss()
        generator_optimizer = self.generator_optimizer()
        discriminator_optimizer = self.discriminator_optimizer()


        # Metrics
        train_gen_loss = tf.metrics.Mean()
        train_disc_loss = tf.metrics.Mean()
        valid_gen_loss = tf.metrics.Mean()
        valid_disc_loss = tf.metrics.Mean()

        # Paths
        log_folder = FolderManager.logs(self)
        log_folder.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = FolderManager.checkpoints(self)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_prefix = str(checkpoint_dir / "ckpt")

        # Checkpoint
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator
        )

        # Init log file
        log_path = log_folder / f"log_{self.name}.log"
        with log_path.open("w") as fd:
            fd.write("epoch,gen_loss,disc_loss,val_gen_loss,val_disc_loss,time\n")

        start_time = time.time()

        with mlflow.start_run(run_name=self.name):

            # Optional: add experiment parameters/tags
            mlflow.set_tag("model_type", "GAN")
            mlflow.log_params({
                "epochs": epochs,
                "saving_freq": saving_freq,
                "model_name": self.name
            })

            for epoch in range(1, epochs + 1):
                # Reset metrics
                train_gen_loss.reset_state()
                train_disc_loss.reset_state()
                valid_gen_loss.reset_state()
                valid_disc_loss.reset_state()

                # Training loop
                for x_target, y_target in dataset_train:
                    gen_loss, disc_loss = train_step(x_target, y_target, generator, discriminator,
                                                     generator_loss, discriminator_loss,
                                                     generator_optimizer, discriminator_optimizer)
                    train_gen_loss.update_state(gen_loss)
                    train_disc_loss.update_state(disc_loss)

                # Validation loop
                for x_target, y_target in dataset_valid:
                    gen_loss, disc_loss = valid_step(x_target, y_target, generator, discriminator,
                                                     generator_loss, discriminator_loss)
                    valid_gen_loss.update_state(gen_loss)
                    valid_disc_loss.update_state(disc_loss)

                # Save checkpoint
                if epoch % saving_freq == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

                # Log to file
                elapsed = time.time() - start_time
                with log_path.open("a") as fd:
                    fd.write(
                        f"{epoch},{train_gen_loss.result().numpy():.6f},{train_disc_loss.result().numpy():.6f},"
                        f"{valid_gen_loss.result().numpy():.6f},{valid_disc_loss.result().numpy():.6f},{elapsed:.2f}\n")

                # Log to MLflow
                mlflow.log_metrics({
                    "train_gen_loss": train_gen_loss.result().numpy(),
                    "train_disc_loss": train_disc_loss.result().numpy(),
                    "val_gen_loss": valid_gen_loss.result().numpy(),
                    "val_disc_loss": valid_disc_loss.result().numpy()
                }, step=epoch)

                print(f"[Epoch {epoch:04d}/{epochs:04d}] "
                      f"gen_loss: {train_gen_loss.result().numpy():.4f}, "
                      f"disc_loss: {train_disc_loss.result().numpy():.4f}, "
                      f"val_gen_loss: {valid_gen_loss.result().numpy():.4f}, "
                      f"val_disc_loss: {valid_disc_loss.result().numpy():.4f}, "
                      f"time: {elapsed:.2f}s")

        return
