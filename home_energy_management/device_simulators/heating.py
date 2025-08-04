from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)
from home_energy_management.device_simulators.device_utils import DeviceUserApi
from home_energy_management.device_simulators.simple_device import ScheduledDevice


class HeatingPreferences(ABC):
    @abstractmethod
    def get_temp(self) -> float:
        pass


class ScheduledHeatingPreferences(HeatingPreferences, ScheduledDevice, Device):
    def __init__(self, config: list[tuple[int, Any]], loop: int = 0):
        super().__init__(config, loop)

    def get_temp(self) -> float:
        temp, _ = self.get_state(self.get_time())
        return float(temp)

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse([0.0, 0.0, 0.0], METERSIM_NO_UPDATE_SCHEDULED)


class LiveHeatingPreferences(HeatingPreferences):
    temperature: float

    def __init__(self, temperature: float):
        self.temperature = float(temperature)

    def get_temp(self) -> float:
        return self.temperature


class TempSensor(DeviceUserApi, Device, ABC):
    @abstractmethod
    def get_temp(self, now: int) -> float:
        pass

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse([0.0, 0.0, 0.0], METERSIM_NO_UPDATE_SCHEDULED)


class LiveTempSensor(TempSensor):
    temp: float

    def __init__(self, temperature: float):
        self.temp = float(temperature)

    def set_temp(self, temperature: float):
        self.temp = float(temperature)

    def get_temp(self, now: int) -> float:
        return self.temp

    def get_info(self) -> dict[str, Any]:
        return {"temperature": self.temp}

    def set_params(self, params: dict[str, Any]) -> None:
        pass


class ScheduledTempSensor(ScheduledDevice, TempSensor):
    def __init__(
            self,
            config: list[tuple[int, Any]],
            loop: int = 0):
        super().__init__(config, loop)

    def get_temp(self, now: int) -> float:
        temp, _ = self.get_state(now)
        return float(temp)

    def get_info(self) -> dict[str, Any]:
        now = self.get_time()
        temp = self.get_temp(now)
        return {"temperature": temp}

    def set_params(self, params: dict[str, Any]) -> None:
        pass


@dataclass
class Heating(Device, DeviceUserApi):
    # Physics
    heat_capacity: float  # J/K
    heating_coefficient: float  # 0-1
    heat_loss_coefficient: float  # W/K

    # Config
    name: str
    temp_window: float  # °C
    heating_devices_power: list[float]  # kW

    # State
    curr_temp: float  # °C
    is_device_switch_on: list[bool]

    # Mutable params
    optimal_temp: float  # °C

    # Utils
    last_temp_update: int
    current: list[complex]  # A
    get_temp_outside: Callable[[int], float]

    def get_info(self) -> dict[str, Any]:
        self.update_temp(self.get_time())
        return {
            "curr_temp": self.curr_temp,
            "optimal_temp": self.optimal_temp,
            "powers_of_heating_devices": self.heating_devices_power,
            "name": self.name,
            "is_device_switch_on": self.is_device_switch_on,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        now = self.get_time()
        self.update_temp(now)
        self.optimal_temp = params["optimal_temp"]
        self.update_state(now)

    def update_temp(self, now: int):
        if now <= self.last_temp_update:
            return

        dt = now - self.last_temp_update
        self.last_temp_update = now

        denergy = (
                self.heating_coefficient
                * sum(is_switch_on * power for is_switch_on, power
                      in zip(self.is_device_switch_on, self.heating_devices_power))
                * 1000 * dt
                - self.heat_loss_coefficient * (self.curr_temp - self.get_temp_outside(now)) * dt
        )  # J
        dtemp = denergy / self.heat_capacity
        self.curr_temp += dtemp

    def update_state(self, now: int):
        self.update_temp(now)

        curr_state = self.is_device_switch_on

        dtemp = self.curr_temp - self.optimal_temp
        if abs(dtemp) > self.temp_window:
            if dtemp < 0:
                self.is_device_switch_on = [True for _ in range(len(self.heating_devices_power))]
            else:
                self.is_device_switch_on = [False for _ in range(len(self.heating_devices_power))]
        if curr_state != self.is_device_switch_on:
            self.notify()

    def update(self, info: InfoForDevice) -> DeviceResponse:
        ret = DeviceResponse([0, 0, 0], METERSIM_NO_UPDATE_SCHEDULED)
        self.current = [0, 0, 0]
        if info.voltage[0] != 0:
            self.current[0] = (
                    sum(is_switch_on * power for is_switch_on, power
                        in zip(self.is_device_switch_on, self.heating_devices_power))
                    * 1000 / info.voltage[0]
            )
        ret.current = self.current
        return ret
