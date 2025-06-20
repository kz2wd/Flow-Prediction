def make_dataset(self, target_file, sample_amount):
    return HDF5Dataset(target_file, self.channel, sample_amount)


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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8)
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


def train(model, dataset, max_epochs=50, saving_freq=5):
    # Dataloaders
    mlflow.set_tracking_uri("http://localhost:5000")

    dataset_train, dataset_valid, dataset_test = model.get_split_datasets(dataset)
    nx, ny, nz = model.channel.prediction_sub_space.x_size, model.channel.prediction_sub_space.y_size, model.channel.prediction_sub_space.z_size
    model.generator_optimizer = torch.optim.Adam(model.generator.parameters())
    model.discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters())

    def train_step(x_target, y_target):
        model.generator.train()
        model.discriminator.train()

        y_pred = model.generator(x_target)

        real_output = model.discriminator(y_target)
        fake_output = model.discriminator(y_pred.detach())

        gen_loss = model.generator_loss(fake_output, y_pred, y_target)
        disc_loss = model.discriminator_loss(real_output, fake_output)

        model.generator_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        model.generator_optimizer.step()

        model.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        model.discriminator_optimizer.step()

        return gen_loss.item(), disc_loss.item()

    def valid_step(x_target, y_target):
        model.generator.eval()
        model.discriminator.eval()
        with torch.no_grad():
            y_pred = model.generator(x_target)
            real_output = model.discriminator(y_target)
            fake_output = model.discriminator(y_pred)

            gen_loss = model.generator_loss(fake_output, y_pred, y_target)
            disc_loss = model.discriminator_loss(real_output, fake_output)

        return gen_loss.item(), disc_loss.item()

    start_time = time.time()
    torch.autograd.set_detect_anomaly(True)
    with mlflow.start_run(run_name=model.name):
        mlflow.set_tag("model_type", "GAN")
        mlflow.log_params({
            "epochs": epochs,
            "saving_freq": saving_freq,
            "model_name": model.name,
            "batch_size": batch_size,
            "dataset_size": sample_amount,
        })
        for epoch in range(1, epochs + 1):
            print("epoch {}".format(epoch))
            train_gen_losses = []
            train_disc_losses = []
            valid_gen_losses = []
            valid_disc_losses = []

            for x_target, y_target in tqdm.tqdm(dataset_train):
                x_target, y_target = x_target.to(model.device), y_target.to(model.device)
                gen_loss, disc_loss = train_step(x_target, y_target)
                train_gen_losses.append(gen_loss)
                train_disc_losses.append(disc_loss)

            for x_target, y_target in tqdm.tqdm(dataset_valid):
                x_target, y_target = x_target.to(model.device), y_target.to(model.device)
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
                    'generator_state_dict': model.generator.state_dict(),
                    'discriminator_state_dict': model.discriminator.state_dict(),
                    'gen_optimizer_state_dict': model.generator_optimizer.state_dict(),
                    'disc_optimizer_state_dict': model.discriminator_optimizer.state_dict()
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
