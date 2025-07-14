from space_exploration.models.AllModels import ModelReferences
from space_exploration.models.UNet import UNet
from space_exploration.training.training import ModelTraining



def test():
    training = ModelTraining("A", "re200-sr05etot", "COMPONENT_NORMALIZE", "Y_ALONG_COMPONENT_NORMALIZE", 4, name="discri:1e-1")
    training.run()


def unet_sanity_check():
    import torch
    model = UNet(6)
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


def unet_train_test():
    training = ModelTraining("SIMPLE_UNET", "re200-sr05etot", "UNET_ADAPTER64", "Y_ALONG_COMPONENT_NORMALIZE", 8,
                             name="Unet simple")
    training.run()

def wall_decoder_check():
    training = ModelTraining("WALL_DECODER", "re200-sr05etot", "COMPONENT_NORMALIZE", "Y_ALONG_COMPONENT_NORMALIZE", 4, name="wall decoder only")
    training.run()

def tune_discri():
    from space_exploration.beans.training_bean import Training
    from space_exploration.training.training import ModelTraining

    run_id = "a67107ff6b9c4ab6ad48546e70384b88"
    # run_id = "feb60fdfd515407c9d30d2119e3236e7"
    bean = Training.get_training_or_fail(run_id)
    training = ModelTraining.from_training_bean(bean)
    training.load_model()
    training.name = "a67-discri-tuned:1e-2"
    training.run()

def print_stats():
    models = {
        "GAN": ModelReferences.A.model(),
        "Wall Decoder": ModelReferences.WALL_DECODER.model(),
        "UNet": ModelReferences.SIMPLE_UNET.model(),
    }

    for name, model in models.items():
        print(f"model [{name}] statistics")
        model.print_stats()
        print()




if __name__ == '__main__':
    test()
    # tune_discri()
    # unet_sanity_check()
    # unet_train_test()
    # wall_decoder_check()
    # print_stats()