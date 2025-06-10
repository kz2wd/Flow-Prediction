import torch
import numpy as np
import time
import shutil
import tqdm
import mlflow
from pathlib import Path

from space_exploration.FolderManager import FolderManager

from torch.utils.data import DataLoader, random_split
import torch


def get_split_datasets(dataset, batch_size=32, val_ratio=0.1, test_ratio=0.1, num_workers=4, device=None):
    """
    Splits dataset into train, validation, and test sets and returns DataLoaders.

    Args:
        dataset (Dataset): Full dataset to split.
        batch_size (int): Batch size for training and validation.
        val_ratio (float): Fraction of dataset used for validation.
        test_ratio (float): Fraction of dataset used for testing.
        num_workers (int): Number of workers for DataLoader.
        device (torch.device or str, optional): If provided, uses pinned memory for faster transfer.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders ready for training.
    """
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    pin_memory = True if device and device.type == 'cuda' else False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True
    )

    return train_loader, val_loader, test_loader

def train_gan(model, dataset_train, dataset_valid, max_epochs=50, patience=7, saving_freq=5):
    mlflow.set_tracking_uri("http://localhost:5000")
    device = model.device

    # === OPTIMIZERS AND LR SCHEDULERS ===
    model.generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
    model.discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.generator_optimizer, mode='min', factor=0.5, patience=3
    )

    # === EARLY STOPPING SETUP ===
    best_val_loss = float('inf')
    best_checkpoint_path = None
    patience_counter = 0

    # === ARTIFACT MANAGEMENT ===
    artifact_dir = Path(FolderManager.artifact_backup_folder(model))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint_latest.pt"

    def train_one_epoch():
        model.generator.train()
        model.discriminator.train()
        gen_losses, disc_losses = [], []

        for x_target, y_target in tqdm.tqdm(dataset_train, desc="Training"):
            x_target, y_target = x_target.to(device), y_target.to(device)

            # === Generator forward and loss ===
            y_pred = model.generator(x_target)
            with torch.no_grad():
                fake_output_for_gen = model.discriminator(y_pred)
            gen_loss = model.generator_loss(fake_output_for_gen, y_pred, y_target)

            model.generator_optimizer.zero_grad()
            gen_loss.backward()
            model.generator_optimizer.step()

            # === Discriminator forward and loss ===
            real_output = model.discriminator(y_target)
            fake_output = model.discriminator(y_pred.detach())
            disc_loss = model.discriminator_loss(real_output, fake_output)

            model.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            model.discriminator_optimizer.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

        return np.mean(gen_losses), np.mean(disc_losses)

    def validate():
        model.generator.eval()
        model.discriminator.eval()
        gen_losses, disc_losses = [], []

        with torch.no_grad():
            for x_target, y_target in tqdm.tqdm(dataset_valid, desc="Validating"):
                x_target, y_target = x_target.to(device), y_target.to(device)
                y_pred = model.generator(x_target)
                real_output = model.discriminator(y_target)
                fake_output = model.discriminator(y_pred)

                gen_loss = model.generator_loss(fake_output, y_pred, y_target)
                disc_loss = model.discriminator_loss(real_output, fake_output)

                gen_losses.append(gen_loss.item())
                disc_losses.append(disc_loss.item())

        return np.mean(gen_losses), np.mean(disc_losses)

    def save_checkpoint(epoch):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'gen_optimizer_state_dict': model.generator_optimizer.state_dict(),
            'disc_optimizer_state_dict': model.discriminator_optimizer.state_dict()
        }, checkpoint_path)
        return checkpoint_path

    # === TRAIN LOOP ===
    with mlflow.start_run(run_name=model.name):
        mlflow.set_tag("model_type", "GAN")
        mlflow.log_params({
            "max_epochs": max_epochs,
            "model_name": model.name,
        })

        start_time = time.time()
        for epoch in range(1, max_epochs + 1):
            print(f"\nEpoch {epoch}/{max_epochs}")
            train_gen_loss, train_disc_loss = train_one_epoch()
            val_gen_loss, val_disc_loss = validate()

            elapsed = time.time() - start_time

            # Log metrics
            mlflow.log_metrics({
                "train_gen_loss": train_gen_loss,
                "train_disc_loss": train_disc_loss,
                "val_gen_loss": val_gen_loss,
                "val_disc_loss": val_disc_loss
            }, step=epoch)

            print(f"[Epoch {epoch}] "
                  f"train_gen_loss={train_gen_loss:.4f}, "
                  f"train_disc_loss={train_disc_loss:.4f}, "
                  f"val_gen_loss={val_gen_loss:.4f}, "
                  f"val_disc_loss={val_disc_loss:.4f}, "
                  f"time={elapsed:.2f}s")

            lr_scheduler.step(val_gen_loss)

            if epoch % saving_freq == 0:
                save_checkpoint(epoch)

            if val_gen_loss < best_val_loss:
                best_val_loss = val_gen_loss
                patience_counter = 0
                best_checkpoint_path = save_checkpoint(epoch)
            else:
                patience_counter += 1
                print(f"  [EarlyStopping] Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("  Early stopping triggered.")
                    break

        if best_checkpoint_path is not None:
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="final_model")

        shutil.rmtree(artifact_dir)
        print(f"Training completed. Final model saved. Local artifacts removed from {artifact_dir}")



import torch.nn.functional as F

def test_gan(model, dataset_test, log_mlflow=True):
    model.generator.eval()

    mse_losses = []

    with torch.no_grad():
        for x_target, y_target in tqdm.tqdm(dataset_test, desc="Testing"):
            x_target, y_target = x_target.to(model.device), y_target.to(model.device)
            y_pred = model.generator(x_target)
            mse_loss = F.mse_loss(y_pred, y_target).item()
            mse_losses.append(mse_loss)

    mean_mse = np.mean(mse_losses)
    print(f"[Test MSE] {mean_mse:.6f}")
    if log_mlflow:
        mlflow.log_metric("test_mse", float(mean_mse))
    return mean_mse


def gen_output(model, x):
    with torch.no_grad():
        x = x.to(model.device)
        y = model.generator(x)
    return y
