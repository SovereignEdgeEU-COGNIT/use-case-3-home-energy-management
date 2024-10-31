# Home Energy Managment: Decision Algorithm and Devices Simulators

This repository contains a decision-making algorithm used to optimize energy management at home and simulators of end 
devices in the local grid. The algorithm retrieves information about all controller settings and current values. Based 
on them, it determines parameters for the next time step and calculates energy transmission in the household.

## Building
Use `pip` to build the library. In `use-case-3-home-energy-managment` directory call:
```bash
pip install use-case-3-home-energy-managment
```

### Decision algorithm
Decision algorithm makes decision based on current values (without any predictions of future). It is assumed it knows 
ideal models of end devices, so can compute energy distributed between them during next time step. Implemented function 
can be used locally or offloaded to COGNIT server. 
Input parameters:
- model_parameters - parameters defining the home energy management model; dict with values for keys: 
heating_delta_temperature, heating_coefficient, heat_loss_coefficient, heat_capacity, delta_charging_power_perc,
- step_timedelta_s - duration of one step in seconds,
- storage_parameters - parameters defining the energy storage model; dict with values for keys: max_capacity, 
min_charge_level, efficiency, nominal_power,
- ev_battery_parameters - parameters defining the EV battery model; dict with values for keys: max_capacity, 
charged_level, efficiency, nominal_power, is_available, time_until_charged,
- room_heating_params_list - parameters defining the heating model for individual rooms; list with dicts, each 
containing values for keys: name, powers_of_heating_devices,
- energy_drawn_from_grid - active energy drawn from the grid in the previous step in kWh,
- energy_returned_to_grid - active energy returned to the grid in the previous step in kWh,
- energy_pv_produced - energy produced by PV matrix in the previous step in kWh,
- temp_outdoor - current outdoor temperature in °C,
- charge_level_of_storage - current charge level of storage in %,
- prev_charge_level_of_storage - charge level of storage before previous step in %,
- heating_status_per_room - statuses of heating devices switches; dict with list of booleans representing status per 
key which is room name,  
- temp_per_room - measured temperature per room in °C.

Returns tuple of variables representing:
- configuration of temperature per room in °C,
- configuration of energy storage (charging and discharging power limits [percent of nominal power], mode of operation),
- configuration of EV battery (charging and discharging power limits [percent of nominal power], mode of operation),
- predicted temperature per room in °C,
- predicted charge level of energy storage in %,
- predicted charge level of EV battery in %,
- predicted energy needed from power grid in kWh.

Simple example of usage of decision algorithm is in `example.py`.

### Devices
Package `device_simulators` includes implementations of class `Device` from library `phoenixsystems-sem` for individual 
end devices in home energy grid.
```Python
class Device(ABC):
    mgr: Any

    def notify(self) -> None:
        """
	Notifies the Simulator that the state has changed and it should poll the device for updates
	"""
        self.mgr.notify()

    def get_time(self) -> int:
        """Get current timestamp from the Simulator"""
        return self.mgr.getTime()

    @abstractmethod
    def update(self, info: InfoForDevice) -> DeviceResponse:
        """The device callback"""
        pass
```

#### Storage
Simple battery storage bidding model with constant charging power limit, as it is suitable for integration in 
optimisation problems. State of energy at step t is described by the following relation:
```math
soe_t = soe_{t-1} + (t * P_ch, t * \eta - \delta t * P_dis, t) * 100 / Cmax
```

#### Heating
In this model, it is assumed that the air in the room heats up directly as a result of the device operating at a 
specific power and efficiency for a given time. The losses are approximated using one parameter. The temperature in the 
room at step t is determined as follows:
```math
T_t = E_t / h
E_t = E_{t-1} + s_t * \eta * P_heat * \delta t + C *(T_{out, t} - T_{t-1}) * \delta t
```

#### Electric Vehicle
The charging/discharging model of an electric car battery when connected to a charger is the same as in the case of a 
storage. However, it can only be used when the car is connected (driving power is set to 0), otherwise the car's battery 
is discharged monotonously according to the assumed load associated with driving the vehicle. For simplicity, the car 
cannot be recharged or stopped outside - when driving power changes to 0, it is interpreted as arriving and connecting 
to the home charger.
