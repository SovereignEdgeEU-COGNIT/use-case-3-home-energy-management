import cmath
from abc import ABC, abstractmethod
from typing import Any


def make_current(current: list[float]) -> list[complex]:
    ret = []
    for i in range(3):
        ret.append(current[i] * cmath.exp(complex(0, 120.0 * i * cmath.pi / 180.0)))
    return ret


def complex_dot_product(x: complex, y: complex) -> float:
    return x.real * y.real + x.imag * y.imag


class DeviceUserApi(ABC):
    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def set_params(self, params: dict[str, Any]) -> None:
        pass
