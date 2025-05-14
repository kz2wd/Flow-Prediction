import time
from abc import ABC, abstractmethod

import h5py
import mlflow
import numpy as np
import tqdm
import vtk
from torch.utils.data import DataLoader, random_split
from vtk.util import numpy_support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FolderManager import FolderManager
from space_exploration.data_viz.PlotData import PlotData, save_benchmarks
from space_exploration.models.dataset import HDF5Dataset
from space_exploration.simulation_channel import SimulationChannel
from visualization.saving_file_names import *

class ResBlockGen(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        out = out + identity
        return out

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 1), mode='nearest')
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, channel: SimulationChannel, input_channels=3, n_residual_blocks=32, output_channels=3):
        super().__init__()
        self.ny = channel.prediction_sub_space.y[1]

        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.up_sample_1 = UpSamplingBlock(64, 64)
        res_block = []
        up_sampling = []

        for index in range(n_residual_blocks):
            res_block.append(ResBlockGen(64))
            if index in [6, 12, 18, 24, 30]:
                res_block.append(UpSamplingBlock(64, 64))
                up_sampling.append(nn.Upsample(scale_factor=(1, 2, 1), mode='nearest'))

        self.res_block = nn.Sequential(*res_block)
        self.up_sampling = nn.Sequential(*up_sampling)

        self.conv2 = nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1)

        self.output_conv = nn.Conv3d(256, output_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, X, Y, Z)
        x = self.initial(x)
        x = self.up_sample_1(x)
        up_samp = self.up_sampling(x)
        x = self.res_block(x)
        x = x + up_samp
        x = self.conv2(x)
        x = self.output_conv(x)
        x = x.permute(0, 2, 3, 4, 1)  # put it back ...
        return x

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels, channel: SimulationChannel):
        nx = channel.prediction_sub_space.x[1]
        ny = channel.prediction_sub_space.y[1]
        nz = channel.prediction_sub_space.z[1]

        super().__init__()
        # flatten_size = nx // 16 * ny // 16 * nz // 16 * 512
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 64, kernel_size=3, stride=4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, X, Y, Z)
        x = self.model(x)
        return x

# Loss functions
def generator_loss(fake_y, y_pred, y_true, batch_size, global_batch_size, nx, ny, nz):
    adversarial_labels = torch.ones_like(fake_y) - torch.rand_like(fake_y) * 0.2
    adversarial_loss = F.binary_cross_entropy(fake_y, adversarial_labels, reduction='none')

    content_loss = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=(1, 2, 3, 4))
    total_loss = content_loss + 1e-3 * adversarial_loss

    return total_loss.mean()

def discriminator_loss(real_y, fake_y, global_batch_size):
    real_labels = torch.ones_like(real_y) - torch.rand_like(real_y) * 0.2
    fake_labels = torch.rand_like(fake_y) * 0.2

    real_loss = F.binary_cross_entropy(real_y, real_labels, reduction='none')
    fake_loss = F.binary_cross_entropy(fake_y, fake_labels, reduction='none')

    total_loss = 0.5 * (real_loss + fake_loss)
    return total_loss.mean() / global_batch_size

class GAN3D(ABC):
    def __init__(self, name, checkpoint_number, channel: SimulationChannel,
                 n_residual_blocks=32, input_channels=3, output_channels=3, learning_rate=1e-4, ):
        self.checkpoint_number = checkpoint_number

        self.channel: SimulationChannel = channel

        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name
        self.n_residual_blocks = n_residual_blocks
        self.already_built = False

        FolderManager.init(self)  # ensure all related folders are created

        self.device = torch.device("cuda")  # if we cannot get cuda, don't even try...

        self.generator = Generator(channel).to(self.device)
        self.discriminator = Discriminator(input_channels, channel).to(self.device)

    def make_dataset(self, target_file, sample_amount):
        return HDF5Dataset(target_file, sample_amount)

    def get_dataloader(self, target_file, batch_size, shuffle=True):
        dataset = self.make_dataset(target_file)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_split_datasets(self, target_file, batch_size, sample_amount, seed=0):
        dataset = self.make_dataset(target_file, sample_amount)

        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader


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
            if len(self.y_predict_normalized) < amount:
                raise ValueError
        except (FileNotFoundError, KeyError, ValueError):
            self.test(amount)
            self.save()

    def benchmark(self):
        # MSE along Y
        mse = np.mean((self.y_target_normalized - self.y_predict_normalized) ** 2, axis=(0, 1, 3, 4))
        u_mse = np.mean((self.y_target_normalized[..., 0] - self.y_predict_normalized[..., 0]) ** 2, axis=(0, 1, 3))
        v_mse = np.mean((self.y_target_normalized[..., 1] - self.y_predict_normalized[..., 1]) ** 2, axis=(0, 1, 3))
        w_mse = np.mean((self.y_target_normalized[..., 2] - self.y_predict_normalized[..., 2]) ** 2, axis=(0, 1, 3))

        save_benchmarks(self, {PlotData.total_mse_y_wise: mse,
                               PlotData.u_mse_y_wise: u_mse,
                               PlotData.v_mse_y_wise: v_mse,
                               PlotData.w_mse_y_wise: w_mse,
                               })

    def train(self, epochs, saving_freq, batch_size, sample_amount=-1):
        # Dataloaders
        print("starting train")
        dataset_train, dataset_valid, dataset_test = self.get_split_datasets(FolderManager.dataset / "test.hdf5", batch_size, sample_amount)
        print("established dataset")
        nx, ny, nz = self.channel.prediction_sub_space.x_size, self.channel.prediction_sub_space.y_size, self.channel.prediction_sub_space.z_size
        self.generator_loss = lambda real_y, fake_y, y_true: generator_loss(real_y, fake_y, y_true, batch_size, 1, nx, ny, nz)
        self.discriminator_loss = lambda real_y, fake_y: discriminator_loss(real_y, fake_y, 1)
        print("established losses")
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        print("established optimizers")

        def train_step(x_target, y_target):
            # print("starting train step")
            self.generator.train()
            self.discriminator.train()

            # print("generating output")
            y_pred = self.generator(x_target)

            real_output = self.discriminator(y_target)
            fake_output = self.discriminator(y_pred.detach())

            gen_loss = self.generator_loss(fake_output, y_pred, y_target)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            self.generator_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True)
            self.generator_optimizer.step()

            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()

            return gen_loss.item(), disc_loss.item()

        def valid_step(x_target, y_target):
            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                y_pred = self.generator(x_target)
                real_output = self.discriminator(y_target)
                fake_output = self.discriminator(y_pred)

                gen_loss = self.generator_loss(fake_output, y_pred, y_target)
                disc_loss = self.discriminator_loss(real_output, fake_output)

            return gen_loss.item(), disc_loss.item()

        # Paths
        log_folder = FolderManager.logs(self)
        log_folder.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = FolderManager.checkpoints(self)
        checkpoint_prefix = checkpoint_dir / "ckpt"

        log_path = log_folder / f"log_{self.name}.log"
        with log_path.open("w") as fd:
            fd.write("epoch,gen_loss,disc_loss,val_gen_loss,val_disc_loss,time\n")

        start_time = time.time()
        torch.autograd.set_detect_anomaly(True)
        print('real start')
        with mlflow.start_run(run_name=self.name):
            mlflow.set_tag("model_type", "GAN")
            mlflow.log_params({
                "epochs": epochs,
                "saving_freq": saving_freq,
                "model_name": self.name,
                "batch_size": batch_size,
                "dataset_size": sample_amount,
            })
            # print('mlflow enable')
            for epoch in range(1, epochs + 1):
                print("epoch {}".format(epoch))
                train_gen_losses = []
                train_disc_losses = []
                valid_gen_losses = []
                valid_disc_losses = []

                for x_target, y_target in tqdm.tqdm(dataset_train):
                    # print("batching...")
                    x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                    # print("sent to device")
                    gen_loss, disc_loss = train_step(x_target, y_target)
                    train_gen_losses.append(gen_loss)
                    train_disc_losses.append(disc_loss)

                for x_target, y_target in tqdm.tqdm(dataset_valid):
                    x_target, y_target = x_target.to(self.device), y_target.to(self.device)
                    gen_loss, disc_loss = valid_step(x_target, y_target)
                    valid_gen_losses.append(gen_loss)
                    valid_disc_losses.append(disc_loss)

                mean_train_gen_loss = np.mean(train_gen_losses)
                mean_train_disc_loss = np.mean(train_disc_losses)
                mean_valid_gen_loss = np.mean(valid_gen_losses)
                mean_valid_disc_loss = np.mean(valid_disc_losses)

                if epoch % saving_freq == 0:
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': self.generator.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'gen_optimizer_state_dict': self.generator_optimizer.state_dict(),
                        'disc_optimizer_state_dict': self.discriminator_optimizer.state_dict()
                    }, checkpoint_prefix.with_suffix(f".e{epoch}.pt"))

                elapsed = time.time() - start_time
                with log_path.open("a") as fd:
                    fd.write(f"{epoch},{mean_train_gen_loss:.6f},{mean_train_disc_loss:.6f},"
                             f"{mean_valid_gen_loss:.6f},{mean_valid_disc_loss:.6f},{elapsed:.2f}\n")

                mlflow.log_metrics({
                    "train_gen_loss": mean_train_gen_loss,
                    "train_disc_loss": mean_train_disc_loss,
                    "val_gen_loss": mean_valid_gen_loss,
                    "val_disc_loss": mean_valid_disc_loss
                }, step=epoch)

                print(f"[Epoch {epoch:04d}/{epochs:04d}] "
                      f"gen_loss: {mean_train_gen_loss:.4f}, "
                      f"disc_loss: {mean_train_disc_loss:.4f}, "
                      f"val_gen_loss: {mean_valid_gen_loss:.4f}, "
                      f"val_disc_loss: {mean_valid_disc_loss:.4f}, "
                      f"time: {elapsed:.2f}s")

    # WARNING : Correct type here should be rectilinear grid
    # but for some reason my Paraview couldn't display it as a Volume, So I use StructuredGrid
    # If you want to try with rectilinear, add an export_vtr function or something alike.
    def export_vts(self):
        self._export_array_vts(self.y_target_normalized[0], TARGET_FILE_NAME, TARGET_ARRAY_NAME)
        self._export_array_vts(self.y_predict_normalized[0], PREDICTION_FILE_NAME, PREDICTION_ARRAY_NAME)

    # File name with no extension
    def _export_array_vts(self, target, file_name, array_name=None):
        if array_name is None:
            array_name = file_name
        structured_grid = vtk.vtkStructuredGrid()
        points = vtk.vtkPoints()
        for k in range(self.channel.prediction_sub_space.z_size):
            for j in range(self.channel.prediction_sub_space.y_size):
                for i in range(self.channel.prediction_sub_space.x_size):
                    points.InsertNextPoint(self.channel.x_dimension[i], self.channel.y_dimension[j],
                                           self.channel.z_dimension[k])

        structured_grid.SetPoints(points)
        structured_grid.SetDimensions(*self.channel.prediction_sub_space.sizes())

        velocity_array = numpy_support.numpy_to_vtk(num_array=target.reshape(-1, 3), deep=True,
                                                    array_type=vtk.VTK_FLOAT)
        velocity_array.SetName(array_name)

        structured_grid.GetPointData().AddArray(velocity_array)

        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(FolderManager.generated_data(self) / file_name)
        writer.SetInputData(structured_grid)
        writer.Write()
