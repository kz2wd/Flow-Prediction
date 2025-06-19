from space_exploration.run_training import ModelTraining



def test():
    training = ModelTraining("A", "re200-sr005etot", "COMPONENT_NORMALIZE", "Y_ALONG_COMPONENT_NORMALIZE", 4, 500)
    training.run()


if __name__ == '__main__':
    test()