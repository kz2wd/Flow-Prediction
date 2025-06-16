from enum import Enum
from space_exploration.models.implementations.A import A as ModelA
from space_exploration.models.implementations.C import C as ModelC

class ModelEnumBase(str, Enum):
    def __new__(cls, value, model):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.model = model
        return obj

    def __str__(self):
        return self.value


class ModelReferences(ModelEnumBase):
    A = ("A", ModelA)
    C = ("C", ModelC)
