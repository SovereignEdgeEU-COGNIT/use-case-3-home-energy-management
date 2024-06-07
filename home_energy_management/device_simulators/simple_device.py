from typing import Any

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)


class ScheduledDevice:
    config: list[tuple[int, Any]]
    loop: int

    def __init__(self, config: list[tuple[int, Any]], loop: int = 0):
        self.config = config
        self.loop = loop

    def get_state(self, now: int) -> tuple[Any, int]:
        if self.loop != 0:
            residue = now % self.loop
        else:
            residue = now

        n = len(self.config)
        ret = None

        assert self.loop != 0 or self.config[0][0] == 0  # FIXME:

        for i in range(n - 1):
            if residue >= self.config[i][0] and residue < self.config[i + 1][0]:
                ret = (self.config[i][1], now - residue + self.config[i + 1][0])
        if ret is None:
            if self.loop == 0:
                ret = (self.config[n - 1][1], -1)
            else:
                if residue < self.config[0][0]:
                    ret = (self.config[n - 1][1], now - residue + self.config[0][0])
                else:
                    ret = (self.config[n - 1][1], now - residue + self.loop + self.config[0][0])
        return ret


class SimpleDevice(Device):
    current: list[complex]


class SimpleScheduledDevice(ScheduledDevice, SimpleDevice):
    def __init__(self, config: list[tuple[int, Any]], loop: int = 0):
        super().__init__(config, loop)
        self.current = [0.0, 0.0, 0.0]

    def update(self, info: InfoForDevice) -> DeviceResponse:
        current, next_update_time = self.get_state(info.now)
        self.current = [complex(x) for x in current]
        return DeviceResponse(current, next_update_time)


class SimpleLiveDevice(SimpleDevice):
    def __init__(self) -> None:
        self.current = [0.0, 0.0, 0.0]

    def set_state(self, current: list[complex]):
        self.current = current
        self.notify()

    def update(self, info: InfoForDevice) -> DeviceResponse:
        return DeviceResponse(self.current, METERSIM_NO_UPDATE_SCHEDULED)
