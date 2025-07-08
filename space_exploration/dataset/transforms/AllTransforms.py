from enum import Enum

from space_exploration.dataset.transforms.general.component_normalize import ComponentNormalize
from space_exploration.dataset.transforms.general.default_unchanged import DefaultUnchanged
from space_exploration.dataset.transforms.x_only.unet_adapter import UNetAdapter64
from space_exploration.dataset.transforms.y_only.y_along_component_normalizer import YAlongComponentNormalizer

class TransformationEnumBase(str, Enum):
    def __new__(cls, value, transformation):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.transformation = transformation
        return obj

    def __str__(self):
        return self.value


class TransformationReferences(TransformationEnumBase):
    DEFAULT_UNCHANGED = ("DEFAULT_UNCHANGED", DefaultUnchanged)
    Y_ALONG_COMPONENT_NORMALIZE = ("Y_ALONG_COMPONENT_NORMALIZE", YAlongComponentNormalizer)
    COMPONENT_NORMALIZE = ("COMPONENT_NORMALIZE", ComponentNormalize)
    UNET_ADAPTER64 = ("UNET_ADAPTER64", UNetAdapter64)
