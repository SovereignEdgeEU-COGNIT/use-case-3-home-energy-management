from abc import abstractmethod, ABC
from typing import Any

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)
from home_energy_management.device_simulators.device_utils import complex_dot_product, DeviceUserApi, make_current
from home_energy_management.device_simulators.simple_device import ScheduledDataDevice


class AbstractPV(Device, DeviceUserApi, ABC):
    voltage: list[complex]
    current: list[complex]
    next_update_time: int
    last_energy_update: int
    produced_energy: float

    def __init__(self) -> None:
        self.voltage = [0.0, 0.0, 0.0]
        self.current = [0.0, 0.0, 0.0]
        self.next_update_time = METERSIM_NO_UPDATE_SCHEDULED
        self.last_energy_update = 0
        self.produced_energy = 0.0

    @abstractmethod
    def update_state(self, now: int) -> None:
        """Updates current and nextUpdateTime"""
        pass

    def get_info(self) -> dict[str, Any]:
        self.calculate_energy(self.get_time())
        return {
            "current": self.current,
            "energy_produced": self.produced_energy,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        pass

    def calculate_energy(self, now: int) -> None:
        acc = 0.0
        dt = now - self.last_energy_update
        for i in range(3):
            acc += dt * complex_dot_product(self.voltage[i], -self.current[i])
        self.produced_energy += acc
        self.last_energy_update = now

    def get_energy(self) -> float:
        return self.produced_energy

    def update(self, info: InfoForDevice) -> DeviceResponse:
        self.voltage = info.voltage
        self.calculate_energy(info.now)
        self.update_state(info.now)
        return DeviceResponse(self.current, self.next_update_time)


class ScheduledPV(AbstractPV, ScheduledDataDevice):
    def __init__(
            self,
            update_time: list[int],
            data: list[Any]
    ) -> None:
        AbstractPV.__init__(self)
        ScheduledDataDevice.__init__(self, update_time, data)

    def update_state(self, now: int) -> None:
        power, next_update_time = self.get_state(now)
        self.current = make_current([power / 230., 0, 0])
        self.next_update_time = next_update_time


class LivePV(AbstractPV):
    def __init__(self) -> None:
        AbstractPV.__init__(self)

    def update_state(self, now: int) -> None:
        pass

    def set_state(self, current: list[complex]):
        self.calculate_energy(self.get_time())
        self.current = [complex(x) for x in current]
        self.notify()
