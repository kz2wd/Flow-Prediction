
from torch.utils.data import DataLoader, random_split


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

    train_loader = prepare_dataset(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = prepare_dataset(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = prepare_dataset(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def prepare_dataset(dataset, batch_size, num_workers=4, device=None):
    pin_memory = True if device and device.type == 'cuda' else False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True
    )
