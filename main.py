from space_exploration.models.implementations.A import A


def test():

    model = A("A-test", "")
    model.train(1, 1, 4, 100)


if __name__ == '__main__':
    test()