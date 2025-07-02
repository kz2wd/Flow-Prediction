from space_exploration.models.UNet import UNet
from space_exploration.training.training import ModelTraining



def test():
    training = ModelTraining("C", "re200-sr05etot", "COMPONENT_NORMALIZE", "Y_ALONG_COMPONENT_NORMALIZE", 4)
    training.run()


def unet_sanity_check():
    import torch
    model = UNet()
    input_channel = torch.rand((1, 3, 64, 64, 64))
    input_wall = torch.rand((1, 3, 64, 1, 64))
    input_wall_full = input_wall.expand(-1, -1, -1, 64, -1)
    X = torch.cat([input_channel, input_wall_full], dim=1)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(X)
    print(outputs.shape)

if __name__ == '__main__':
    test()