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
There are two versions of decision algorithms. Baseline version makes decision based only on current values using 
predefined scheme of consumption needs and importance. AI version of algorithm uses model trained using PPO learning 
function to optimise parameters of controlled devices for next 24 hours and is based on predictions of energy data. 
It is assumed it knows ideal models of end devices, so can compute energy distributed between them during next time 
step. Implemented functions can be used locally or offloaded to COGNIT server. 
Input parameters:

- `timestamp` - current timestamp, for which decision is made,
- `s3_parameters` - JSON with parameters used for authentication to s3 service and downloading file with trained model 
(for baseline implementation parameter is ignored as not needed),
- `besmart_parameters` - JSON with parameters used for authentication to besmart.energy API and downloading data,
- `home_model_parameters` - JSON with parameters defining the home energy management model; dict with values for keys: 
temp_window, heating_coefficient, heat_loss_coefficient, heat_capacity, delta_charging_power_perc,
- `storage_parameters` - JSON with parameters defining the energy storage model; dict with values for keys: 
max_capacity, min_charge_level, efficiency, nominal_power, curr_charge_level,
- `ev_battery_parameters_per_id` - JSON with parameters defining per EV battery model; dict of dicts with values for 
keys: max_capacity, driving_charge_level, efficiency, nominal_power, is_available, time_until_charged, curr_charge_level,
- `heating_parameters` - JSON with parameters defining the heating model for home; dict with values for keys: name, 
curr_temp, preferred_temp, powers_of_heating_devices, is_device_switch_on,
- `user_preferences` - JSON with user preferences, i.a. cycle_timedelta_s - time duration of one cycle in seconds.

Returns tuple of variables representing:
- configuration of temperature per room in °C,
- configuration of energy storage (charging and discharging power limits [percent of nominal power], mode of operation),
- configuration of EV battery (charging and discharging power limits [percent of nominal power], mode of operation).

Simple example of usage of decision algorithm can be found in `example_with_server.py` and `example_ppo_with_server.py`.

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
