from space_exploration.models.A.A03 import A03
from space_exploration.models.C.C04 import C04
from space_exploration.models.C.C05 import C05
from space_exploration.models.C.C06 import C06
from space_exploration.models.C.CBase import CBase


# This is a main
# For now, just used as the root folder of the project


def test():
    model = C04()
    model.train(1, 1, 4)

def test_multiple():
    models = [C04(), C05(), C06()]
    for model in models:
        model.test(5)

if __name__ == '__main__':
    # compare_data()
    # test_multiple()
    test()