from dataclasses import dataclass
from typing import Any
import math

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)
from device_utils import complex_dot_product, DeviceUserApi
from photovoltaic import AbstractPV


@dataclass
class Storage(Device, DeviceUserApi):
    max_power: float  # kW
    max_capacity: float  # kWh
    min_charge_level: float  # %
    charging_switch_level: float  # %
    efficiency: float  # 0-1
    energy_loss: float  # %/s

    # State
    current: list[complex]  # A
    curr_capacity: float  # kWh

    # Mutable params
    max_charge_rate: float  # 0-1
    max_discharge_rate: float  # 0-1
    operation_mode: int  # 1 - charging, 2 - discharging

    # Utils
    last_capacity_update: int
    voltage: list[complex]  # V

    def update_capacity(self, now: int) -> int:
        dt = now - self.last_capacity_update
        self.last_capacity_update = now
        power = complex_dot_product(self.current[0], self.voltage[0]) / 1000.0  # kW
        self.curr_capacity += power * dt / 3600 * (self.efficiency if power > 0 else 1)
        self.curr_capacity = (1.0 - self.energy_loss * dt / 100.0) * self.curr_capacity
        self.curr_capacity = max(self.curr_capacity, 0)

        if power < 0:
            self.currCapacity = max(self.currCapacity, self.min_charge_level / 100 * self.max_capacity)
            tt = math.ceil((self.currCapacity - self.min_charge_level / 100 * self.max_capacity) / (-power) * 3600)
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
        self.update_capacity(self.get_time())
        return {
            "max_capacity": self.max_capacity,
            "min_charge_level": self.min_charge_level,
            "curr_charge_level": self.curr_capacity / self.max_capacity * 100,
            "nominal_power": self.max_power,
            "efficiency": self.efficiency,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        self.update_capacity(self.get_time())
        self.max_charge_rate = params["InWRte"] / 100.0
        self.max_discharge_rate = params["OutWRte"] / 100.0
        self.operation_mode = params["StorCtl_Mod"]
        self.notify()

    def adjust_current_to_state(self,
                                info: InfoForDevice,
                                current: list[complex],
                                pv_current: list[complex]) -> None:
        self.update_capacity(info.now)

        self.voltage = info.voltage

        self.current[0] = 0.0

        if info.voltage[0] == 0.0:
            return

        if self.operation_mode == 2 and current[0].real > 0:
            if self.curr_capacity <= self.min_charge_level / 100 * self.max_capacity:
                self.current[0] = 0.0
            else:
                max_current = self.max_discharge_rate * self.max_power * 1000 / info.voltage[0].real
                self.current[0] = -min(current[0].real, max_current)
        if self.operation_mode == 1 and pv_current[0].real < 0:
            if self.curr_capacity == self.max_capacity:
                self.current[0] = 0.0
            else:
                power_reduction = min(
                    self.max_charge_rate,
                    (self.max_capacity - self.curr_capacity)
                    / (self.max_capacity * (1 - self.charging_switch_level / 100))
                )
                max_current = power_reduction * self.max_power * 1000 / info.voltage[0].real
                self.current[0] = min(-pv_current[0].real, max_current)

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse(self.current, self.update_capacity(info.now))
