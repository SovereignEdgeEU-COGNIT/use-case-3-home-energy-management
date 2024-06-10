from dataclasses import dataclass

from phoenixsystems.sem.device import (
    Device,
    DeviceResponse,
    InfoForDevice,
    METERSIM_NO_UPDATE_SCHEDULED,
)
from home_energy_management.device_simulators.photovoltaic import AbstractPV
from home_energy_management.device_simulators.storage import Storage


@dataclass
class Gateway(Device):
    devices: list[Device]
    storage: Storage
    pv: AbstractPV

    def init_mgr(self):
        self.pv.mgr = self.mgr
        self.storage.mgr = self.mgr
        for dev in self.devices:
            dev.mgr = self.mgr

    def update(self, info: InfoForDevice) -> DeviceResponse:
        res = DeviceResponse([0.0, 0.0, 0.0], METERSIM_NO_UPDATE_SCHEDULED)
        for dev in self.devices:
            res.accumulate(dev.update(info))

        pv_res = self.pv.update(info)
        res.accumulate(pv_res)

        self.storage.adjust_current_to_state(info, res.current, pv_res.current)
        res.accumulate(self.storage.update(info))
        return res
