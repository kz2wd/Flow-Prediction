from space_exploration.training.training import ModelTraining



def test():
    training = ModelTraining("C", "re200-sr05etot", "COMPONENT_NORMALIZE", "Y_ALONG_COMPONENT_NORMALIZE", 4)
    training.run()


if __name__ == '__main__':
    test()