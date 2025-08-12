import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Any, Optional

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)
from home_energy_management.device_simulators.device_utils import complex_dot_product, DeviceUserApi
from home_energy_management.device_simulators.simple_device import ScheduledDevice

EPSILON = 1e-8

@dataclass
class ElectricVehicle(Device, DeviceUserApi):
    max_power: float  # kW
    max_capacity: float  # kWh
    min_charge_level: float  # %
    driving_charge_level: float  # %
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
    operation_mode: int  # 0 - away/idle, 1 - charging, 2 - discharging

    # Utils
    last_capacity_update: int
    voltage: list[complex]  # V

    def set_getter_of_driving_power(self, get_driving_power: Callable[[int], float]):
        self.update_capacity(self.get_time())
        self.get_driving_power = get_driving_power

    def update_capacity(self, now: int) -> int:
        dt = now - self.last_capacity_update
        self.last_capacity_update = now

        driving_power = self.get_driving_power(now)
        if driving_power > 0:
            self.is_available = False
            self.operation_mode = 0
            self.current[0] = 0

            self.curr_capacity -= driving_power * dt / 3600
            self.curr_capacity = max(self.curr_capacity, EPSILON)
            tt = math.ceil(self.curr_capacity / driving_power * 3600)
            if tt > 0:
                return now + tt
            else:
                return 0

        self.is_available = True

        power = complex_dot_product(self.current[0], self.voltage[0]) / 1000.0  # kW
        power_reduction = min(
            1.0,
            (self.max_capacity - self.curr_capacity)
            / (self.max_capacity * (1 - self.charging_switch_level / 100))
        )
        self.curr_capacity += power * dt / 3600 * (self.efficiency * power_reduction if power > 0 else 1)
        self.curr_capacity = (1. - self.energy_loss * dt / 100.) * self.curr_capacity
        self.curr_capacity = max(self.curr_capacity, EPSILON)

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
            return METERSIM_NO_UPDATE_SCHEDULED

    def get_info(self) -> dict[str, Any]:
        now = self.get_time()
        self.update_capacity(now)
        return {
            "max_capacity": self.max_capacity,
            "min_charge_level": self.min_charge_level,
            "driving_charge_level": self.driving_charge_level,
            "charging_switch_level": self.charging_switch_level,
            "curr_charge_level": self.curr_capacity / self.max_capacity * 100,
            "nominal_power": self.max_power,
            "efficiency": self.efficiency,
            "energy_loss": self.energy_loss,
            "is_available": self.is_available,
            "driving_power": self.get_driving_power(now),
        }

    def set_params(self, params: dict[str, Any]) -> None:
        self.update_capacity(self.get_time())
        self.max_charge_rate = params["InWRte"] / 100.
        self.max_discharge_rate = params["OutWRte"] / 100.
        self.operation_mode = params["StorCtl_Mod"]
        self.notify()

    def adjust_current_to_state(self, info: InfoForDevice) -> None:
        self.current[0] = 0

        self.voltage = info.voltage
        if info.voltage[0] == 0:
            return

        if self.operation_mode == 2:
            if self.curr_capacity > self.min_charge_level / 100 * self.max_capacity:
                self.current[0] = self.max_discharge_rate * self.max_power * 1000 / info.voltage[0].real
        if self.operation_mode == 1:
            if self.curr_capacity < self.max_capacity:
                self.current[0] = self.max_charge_rate * self.max_power * 1000 / info.voltage[0].real

    def update(self, info: InfoForDevice) -> DeviceResponse:
        if self.get_driving_power(info.now) == 0.0:
            self.adjust_current_to_state(info)
        return DeviceResponse(self.current, self.update_capacity(info.now))


class EVDriving(Device, DeviceUserApi, ABC):
    @abstractmethod
    def get_driving_power(self, now: int) -> float:
        pass

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse([0.0, 0.0, 0.0], METERSIM_NO_UPDATE_SCHEDULED)


class ScheduledEVDriving(EVDriving, ScheduledDevice):
    def __init__(self, daily_schedule: dict[str, list]):
        time_list = daily_schedule['time']
        value_list = daily_schedule['driving_power']
        config = []
        for time, value in zip(time_list, value_list):
            datetime_time = datetime.strptime(time, "%H:%M").time()
            seconds_from_start = (datetime_time.hour * 60 + datetime_time.minute) * 60
            config.append((seconds_from_start, value))
        EVDriving.__init__(self)
        ScheduledDevice.__init__(self, config, 24 * 3600)

    def get_driving_power(self, now: int) -> float:
        driving_power, _ = self.get_state(now)
        return driving_power

    def get_info(self) -> dict[str, Any]:
        now = self.get_time()
        return {"driving_power": self.get_driving_power(now)}

    def set_params(self, params: dict[str, Any]) -> None:
        pass


class LiveEVDriving(EVDriving):
    driving_power: float

    def __init__(self, driving_power: float,):
        EVDriving.__init__(self)
        self.driving_power = float(driving_power)

    def set_driving_power(self, driving_power: float):
        self.driving_power = float(driving_power)

    def get_driving_power(self, now: int) -> float:
        return self.driving_power

    def get_info(self) -> dict[str, Any]:
        return {"driving_power": self.driving_power}

    def set_params(self, params: dict[str, Any]) -> None:
        pass


class EVDeparturePlans(Device, ABC):
    @abstractmethod
    def get_time_until_departure(self) -> float:
        pass

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse([0.0, 0.0, 0.0], METERSIM_NO_UPDATE_SCHEDULED)


class ScheduledEVDeparturePlans(EVDeparturePlans, ScheduledDevice, Device):
    def __init__(self, daily_schedule: dict[str, list]):
        time_list = daily_schedule['time']
        value_list = daily_schedule['driving_power']
        config = []
        for time, value in zip(time_list, value_list):
            datetime_time = datetime.strptime(time, "%H:%M").time()
            seconds_from_start = (datetime_time.hour * 60 + datetime_time.minute) * 60
            config.append((seconds_from_start, value))
        EVDeparturePlans.__init__(self)
        ScheduledDevice.__init__(self, config, 24 * 3600)

    def get_time_until_departure(self) -> int:
        now = self.get_time()
        _, next_update_time = self.get_state(now)
        update_driving_power = 0.
        while update_driving_power == 0.:
            time = next_update_time
            if next_update_time == METERSIM_NO_UPDATE_SCHEDULED:
                return -1
            update_driving_power, next_update_time = self.get_state(time)
        return time - now


class LiveEVDeparturePlans(EVDeparturePlans):
    next_ev_departure_time: int = METERSIM_NO_UPDATE_SCHEDULED

    def __init__(self, ev_departure_planned_time: Optional[str] = None):
        if ev_departure_planned_time:
            self.next_ev_departure_time = self.__strptime_to_seconds_from_start(ev_departure_planned_time)

    def get_time_until_departure(self) -> int:
        left_time = self.next_ev_departure_time - self.get_time()
        return -1 if (left_time <= 0
                      or self.next_ev_departure_time == METERSIM_NO_UPDATE_SCHEDULED) else left_time

    def update_state(self, ev_departure_planned_time: str):
        next_ev_departure_time = self.__strptime_to_seconds_from_start(ev_departure_planned_time)
        seconds_in_day = timedelta(days=1).total_seconds()
        next_ev_departure_time += (self.get_time() // seconds_in_day) * seconds_in_day
        self.next_ev_departure_time = next_ev_departure_time
        self.notify()

    def __strptime_to_seconds_from_start(self, time_as_string: str) -> int:
        t = datetime.strptime(time_as_string, "%H:%M")
        time_in_seconds = timedelta(hours=t.hour, minutes=t.minute).total_seconds()
        return int(time_in_seconds)
