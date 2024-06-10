import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)
from home_energy_management.device_simulators.device_utils import complex_dot_product, DeviceUserApi
from home_energy_management.device_simulators.simple_device import ScheduledDevice


@dataclass
class ElectricVehicle(Device, DeviceUserApi):
    max_power: float  # kW
    max_capacity: float  # kWh
    min_charge_level: float  # %
    charging_switch_level: float  # %
    efficiency: float  # 0-1
    energy_loss: float  # %/s

    # State
    is_available: bool
    get_driving_power: Callable[[int], float]  # kW
    current: list[complex]  # A
    curr_capacity: float  # kWh

    # Mutable params
    max_charge_rate: float  # 0-1
    max_discharge_rate: float  # 0-1
    operation_mode: int  # 0 - away, 1 - charging, 2 - discharging

    # Utils
    last_capacity_update: int
    get_next_update_time: Callable[[int], int]
    voltage: list[complex]  # V

    def update_capacity(self, now: int) -> int:
        dt = now - self.last_capacity_update
        self.last_capacity_update = now

        driving_power = self.get_driving_power(now)
        if driving_power > 0:
            self.curr_capacity -= driving_power * dt / 3600
            self.curr_capacity = max(self.curr_capacity, 0.)
            tt = math.ceil(self.curr_capacity / driving_power * 3600)
            if tt > 0:
                return now + tt
            else:
                return 0

        power = complex_dot_product(self.current[0], self.voltage[0]) / 1000.0  # kW
        self.curr_capacity += power * dt / 3600 * (self.efficiency if power > 0 else 1)
        self.curr_capacity = (1. - self.energy_loss * dt / 100.) * self.curr_capacity
        self.curr_capacity = max(self.curr_capacity, 0)

        if power < 0:
            self.curr_capacity = max(self.curr_capacity, self.min_charge_level / 100 * self.max_capacity)
            tt = math.ceil((self.curr_capacity - self.min_charge_level / 100 * self.max_capacity) / (-power) * 3600)
            if tt > 0:
                return now + tt
            else:
                return 0
        elif power > 0:
            self.curr_capacity = min(self.curr_capacity, self.max_capacity)
            tt = math.ceil((self.max_capacity - self.curr_capacity) / power * 3600)
            if tt > 0:
                return now + tt
            else:
                return 0
        else:
            return -1

    def get_info(self) -> dict[str, Any]:
        now = self.get_time()
        self.update_capacity(now)
        return {
            "max_capacity": self.max_capacity,
            "min_charge_level": self.min_charge_level,
            "curr_charge_level": self.curr_capacity / self.max_capacity * 100,
            "nominal_power": self.max_power,
            "efficiency": self.efficiency,
            "is_available": self.is_available,
            "driving_power": self.get_driving_power(now),
            "next_update_time": self.get_next_update_time(now),
        }

    def set_params(self, params: dict[str, Any]) -> None:
        self.update_capacity(self.get_time())
        self.max_charge_rate = params["InWRte"] / 100.
        self.max_discharge_rate = params["OutWRte"] / 100.
        self.operation_mode = params["StorCtl_Mod"]
        self.notify()

    def adjust_current_to_state(self, info: InfoForDevice) -> None:
        self.current[0] = 0

        if self.get_driving_power(info.now) > 0:
            self.is_available = False
            self.operation_mode = 0
            self.voltage[0] = 0
            return

        self.is_available = True
        self.operation_mode = 1  # TODO - only while no decision from algorithm

        self.voltage = info.voltage
        if info.voltage[0] == 0:
            return

        if self.operation_mode == 2:
            if self.curr_capacity <= self.min_charge_level / 100 * self.max_capacity:
                self.current[0] = 0
            else:
                self.current[0] = self.max_discharge_rate * self.max_power * 1000 / info.voltage[0].real
        if self.operation_mode == 1:
            if self.curr_capacity == self.max_capacity:
                self.current[0] = 0
            else:
                power_reduction = min(
                    self.max_charge_rate,
                    (self.max_capacity - self.curr_capacity) / (
                                self.max_capacity * (1 - self.charging_switch_level / 100))
                )
                self.current[0] = power_reduction * self.max_power * 1000 / info.voltage[0].real

    def update(self, info: InfoForDevice) -> DeviceResponse:
        self.adjust_current_to_state(info)
        return DeviceResponse(self.current, self.update_capacity(info.now))


class AbstractEVDriving(ABC):
    @abstractmethod
    def get_driving_power(self, now: int) -> float:
        pass

    @abstractmethod
    def get_next_update_time(self, now: int) -> int:
        pass


class ScheduledEVDriving(AbstractEVDriving, ScheduledDevice):
    def __init__(
            self,
            config: list[tuple[float, Any]],
            loop: int = 0):
        super().__init__(config, loop)

    def get_driving_power(self, now: int) -> float:
        driving_power, _ = self.get_state(now)
        return driving_power

    def get_next_update_time(self, now: int) -> int:
        _, next_update_time = self.get_state(now)
        return next_update_time

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse([0, 0, 0], METERSIM_NO_UPDATE_SCHEDULED)


class LiveEVDriving(AbstractEVDriving):
    driving_power: float
    next_update_time: int

    def __init__(
            self,
            driving_power: float,
            next_update_time: int = -1):
        self.driving_power = driving_power
        self.next_update_time = next_update_time

    def get_driving_power(self, now: int):
        return self.driving_power

    def get_next_update_time(self, now: int) -> int:
        return self.next_update_time
