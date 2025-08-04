def make_decision(
        timestamp: float,
        s3_parameters,
        besmart_parameters: str,
        home_model_parameters: str,
        storage_parameters: str,
        ev_battery_parameters_per_id: str,
        heating_parameters: str,
        user_preferences: str,
) -> tuple[float, str, str]:
    """
    The algorithm retrieves information about all controller settings and current values. Based on them, it determines
    parameters for the next cycle.
    Implemented function can be used locally or offloaded to COGNIT server.

    Args:
        timestamp (float): Current timestamp, for which decision is made.
        s3_parameters: For this implementation parameter is ignored as not needed.
        besmart_parameters (str): JSON with parameters used for authentication to besmart API and downloading data;
            dict with values for keys: workspace_key, login, password, pv_generation, energy_consumption,
            temperature_moid.
        home_model_parameters (str): JSON with parameters defining the home energy management model; dict with values
            for keys: temp_window, heating_coefficient, heat_loss_coefficient, heat_capacity,
            delta_charging_power_perc.
        storage_parameters (str): JSON with parameters defining the energy storage model; dict with values for keys:
            max_capacity, min_charge_level, efficiency, nominal_power, curr_charge_level.
        ev_battery_parameters_per_id (str): JSON with parameters defining per EV battery model; dict of dicts with
            values for keys: max_capacity, driving_charge_level, efficiency, nominal_power, is_available,
            time_until_charged, curr_charge_level.
        heating_parameters (str): JSON with parameters defining the heating model for home; dict with values for keys:
            name, curr_temp, preferred_temp, powers_of_heating_devices, is_device_switch_on.
        user_preferences (str): JSON with user preferences, i.a. cycle_timedelta_s - time duration of one cycle in
            seconds.

    Returns:
        Configurations for next cycle in JSONs:
        - configuration of temperature in °C,
        - configuration of energy storage (charging and discharging power limits [percent of nominal power], mode),
        - configuration of EV battery per EV (charging and discharging power limits [percent of nominal power], mode).
    """
    import datetime
    import json
    from typing import Any

    import numpy as np
    import requests

    def authenticate_to_besmart() -> str:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Auth': besmart_parameters['workspace_key'],
        }
        body = {
            "login": besmart_parameters["login"],
            "password": besmart_parameters["password"],
        }
        r = requests.post(
            'https://api.besmart.energy/api/users/token',
            headers=headers,
            json=body
        )
        return r.json()["token"]

    def get_data_from_besmart(cid: int,
                              mid: int,
                              moid: int) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        till_datetime = np.datetime64(state_datetime) + np.timedelta64(cycle_timedelta_s, 's')
        body = [{
            "client_cid": cid,
            "sensor_mid": mid,
            "signal_type_moid": moid,
            "since": int(np.datetime64(state_datetime).astype(int) / 1000),
            "till": int(till_datetime.astype(int) / 1000),
            "get_last": True,
        }]
        res = requests.post(
            'https://api.besmart.energy/api/sensors/signals/data',
            headers=headers, json=body
        )
        return res.json()[0]['data']

    def get_energy_data(identifier: dict[str, int]) -> np.ndarray:
        data = get_data_from_besmart(identifier["cid"],
                                     identifier["mid"],
                                     identifier["moid"], )
        time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
        value = np.array(data['value'])
        origin = np.array(data['origin'])

        pred_value = value[origin == 2]
        pred_time = time[origin == 2]

        if len(pred_value) < 2:
            raise Exception('Not enough data for decision-making')

        return pred_value

    def get_temperature_data() -> np.ndarray:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        sensor_identifier = besmart_parameters["energy_consumption"]
        sensor = requests.get(
            f'https://api.besmart.energy/api/sensors/{sensor_identifier["cid"]}.{sensor_identifier["mid"]}',
            headers=headers,
        ).json()

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Auth': besmart_parameters['workspace_key'],
        }
        till_datetime = np.datetime64(state_datetime) + np.timedelta64(cycle_timedelta_s, 's')
        params = {
            "since": int(np.datetime64(state_datetime).astype(int) / 1000),
            "till": int(till_datetime.astype(int) / 1000),
            'delta_t': cycle_timedelta_s // 60,
            'raw': False,
            'get_last': True,
        }
        res = requests.get(
            f'https://api.besmart.energy/api/weather/{sensor["lat"]}/{sensor["lon"]}/{besmart_parameters["temperature_moid"]}/data',
            headers=headers, params=params
        )
        data = res.json()['data']

        time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
        value = np.array(data['value'])
        origin = np.array(data['origin'])
        estm_value = value[origin == 3]
        estm_time = time[origin == 3]

        if len(estm_value) < 2:
            raise Exception('Not enough data for decision-making')

        return estm_value - 272.15

    def check_heating_conditions(
            heating_params: dict,
            reduction_of_allowed_temp: float = 0.0,
            available_energy: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        energy_for_heating = 0.0  # kWh
        current_temp = heating_params["curr_temp"]
        conf_temp = current_temp
        temp_without_heating = (current_temp - (current_temp - temp_outside_pred)
                                * heat_loss_coeff * cycle_timedelta_s / heat_capacity)
        temp_diff = heating_params["preferred_temp"] - reduction_of_allowed_temp - temp_without_heating
        if temp_diff > 0:
            energy_for_heating = (sum(heating_params["powers_of_heating_devices"])
                                       * cycle_timedelta_s / 3600)  # kWh
            if available_energy and energy_for_heating < available_energy:
                available_energy -= energy_for_heating
            conf_temp += 2 * temp_diff

        return energy_for_heating, conf_temp

    def does_ev_battery_require_charging(ev_battery_params: dict[str, Any]) -> bool:
        charge_level_of_ev_battery = ev_battery_params["curr_charge_level"]
        ev_battery_max_capacity = ev_battery_params["max_capacity"]
        ev_battery_charged_level = ev_battery_params["driving_charge_level"]
        ev_battery_efficiency = ev_battery_params["efficiency"]
        ev_battery_nominal_power = ev_battery_params["nominal_power"]
        time_until_ev_charged = ev_battery_params["time_until_charged"]

        ev_charging_capacity = ((ev_battery_charged_level - charge_level_of_ev_battery)
                                / 100 * ev_battery_max_capacity / ev_battery_efficiency)
        if ev_charging_capacity < 0:
            return False
        min_charging_time = ev_charging_capacity / ev_battery_nominal_power * 3600  # s
        if min_charging_time < time_until_ev_charged - cycle_timedelta_s:
            return False
        return True

    def distribute_needed_energy(
            energy_needed: float,
            energy_pv_produced: float,
            energy_in_storage: float,
    ) -> tuple[float, float, float]:
        if energy_pv_produced > energy_needed:
            return energy_pv_produced - energy_needed, energy_in_storage, 0.0

        if energy_in_storage > (energy_needed - energy_pv_produced):
            return 0.0, energy_in_storage - (energy_needed - energy_pv_produced), 0.0

        return 0.0, 0.0, energy_needed - energy_pv_produced - energy_in_storage


    besmart_parameters = json.loads(besmart_parameters)
    home_model_parameters = json.loads(home_model_parameters)
    storage_parameters = json.loads(storage_parameters)
    ev_battery_parameters_per_id = json.loads(ev_battery_parameters_per_id)
    heating_parameters = json.loads(heating_parameters)
    user_preferences = json.loads(user_preferences)

    state_datetime = datetime.datetime.fromtimestamp(timestamp)
    cycle_timedelta_s = user_preferences["cycle_timedelta_s"]
    cycle_timedelta_min = cycle_timedelta_s // 60
    rounding_minutes = state_datetime.minute % cycle_timedelta_min
    if rounding_minutes > cycle_timedelta_min / 2:
        rounding_minutes = - (cycle_timedelta_min - rounding_minutes)
    state_datetime = datetime.datetime(year=state_datetime.year,
                                       month=state_datetime.month,
                                       day=state_datetime.day,
                                       hour=state_datetime.hour,
                                       minute=state_datetime.minute - rounding_minutes)

    token = authenticate_to_besmart()
    pv_generation_pred = get_energy_data(besmart_parameters["pv_generation"])[0]
    energy_consumption_pred = get_energy_data(besmart_parameters["energy_consumption"])
    energy_consumption_pred = np.diff(energy_consumption_pred)[0]
    temp_outside_pred = get_temperature_data()[0]

    temp_window = home_model_parameters["temp_window"]
    heat_loss_coeff = home_model_parameters["heat_loss_coefficient"]
    heat_capacity = home_model_parameters["heat_capacity"]
    delta_charging_power_perc = home_model_parameters["delta_charging_power_perc"]

    charge_level_of_storage = storage_parameters["curr_charge_level"]
    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_nominal_power = storage_parameters["nominal_power"]

    energy_in_storage = (charge_level_of_storage - storage_min_charge_level) / 100 * storage_max_capacity
    initial_energy_in_storage = energy_in_storage
    temp_configuration = heating_parameters["preferred_temp"]

    (energy_pv_production, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
        energy_needed=energy_consumption_pred,
        energy_pv_produced=pv_generation_pred,
        energy_in_storage=energy_in_storage,
    )

    # Home heating - necessary
    (energy_for_necessary_heating, conf_temp) = check_heating_conditions(
        heating_params=heating_parameters,
        reduction_of_allowed_temp=temp_window,
    )
    if energy_for_necessary_heating > 0:
        temp_configuration = conf_temp
        (energy_pv_production, energy_in_storage, missing_energy) = distribute_needed_energy(
            energy_needed=energy_for_necessary_heating,
            energy_pv_produced=energy_pv_production,
            energy_in_storage=energy_in_storage,
        )
        energy_from_power_grid += missing_energy

    # EV charging - required
    energy_for_ev_charging_per_id = {ev_id: 0.0 for ev_id in ev_battery_parameters_per_id}
    for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items():
        is_ev_available = ev_battery_parameters["is_available"]
        if is_ev_available and does_ev_battery_require_charging(
            ev_battery_params=ev_battery_parameters,
        ):
            charge_level_of_ev_battery = ev_battery_parameters["curr_charge_level"]
            ev_battery_max_capacity = ev_battery_parameters["max_capacity"]
            ev_battery_efficiency = ev_battery_parameters["efficiency"]
            ev_battery_nominal_power = ev_battery_parameters["nominal_power"]
            energy_for_ev_charging = min(
                ev_battery_nominal_power * cycle_timedelta_s / 3600,  # kWh
                (1 - charge_level_of_ev_battery / 100) * ev_battery_max_capacity / ev_battery_efficiency
            )
            (energy_pv_production, energy_in_storage, missing_energy) = distribute_needed_energy(
                energy_needed=energy_for_ev_charging,
                energy_pv_produced=energy_pv_production,
                energy_in_storage=energy_in_storage,
            )
            energy_from_power_grid += missing_energy
            energy_for_ev_charging_per_id[ev_id] = energy_for_ev_charging

    # Home heating - optional
    available_oze_energy = energy_pv_production + energy_in_storage
    energy_for_optional_heating = 0.
    if available_oze_energy > 0.0 and energy_for_necessary_heating == 0:
        (energy_for_optional_heating, conf_temp) = check_heating_conditions(
                heating_params=heating_parameters,
                available_energy=available_oze_energy,
        )
        if energy_for_optional_heating > 0:
            temp_configuration = conf_temp
            (energy_pv_production, energy_in_storage, missing_energy) = distribute_needed_energy(
                energy_needed=energy_for_optional_heating,
                energy_pv_produced=energy_pv_production,
                energy_in_storage=energy_in_storage,
            )
            energy_from_power_grid += missing_energy

    # Home heating - additional
    available_oze_energy = energy_pv_production + max(
        energy_in_storage - home_model_parameters["storage_high_charge_level"] / 100 * storage_max_capacity,
        0.0
    )
    if available_oze_energy > 0.0 and energy_for_optional_heating == 0:
        (energy_for_additional_heating, conf_temp) = check_heating_conditions(
                heating_params=heating_parameters,
                reduction_of_allowed_temp=-temp_window,
                available_energy=available_oze_energy,
        )
        if energy_for_additional_heating > 0:
            temp_configuration = conf_temp
            (energy_pv_production, energy_in_storage, missing_energy) = distribute_needed_energy(
                energy_needed=energy_for_additional_heating,
                energy_pv_produced=energy_pv_production,
                energy_in_storage=energy_in_storage,
            )
            energy_from_power_grid += missing_energy

    # EV charging - optional
    available_oze_energy = energy_pv_production + energy_in_storage
    for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items():
        if (
                ev_battery_parameters["is_available"]
                and energy_for_ev_charging_per_id[ev_id] == 0.0
                and available_oze_energy > 0.0
        ):
            charge_level_of_ev_battery = ev_battery_parameters["curr_charge_level"]
            ev_battery_max_capacity = ev_battery_parameters["max_capacity"]
            ev_battery_efficiency = ev_battery_parameters["efficiency"]
            ev_battery_nominal_power = ev_battery_parameters["nominal_power"]
            energy_for_ev_charging = min(
                available_oze_energy,
                ev_battery_nominal_power * cycle_timedelta_s / 3600,  # kWh
                (1 - charge_level_of_ev_battery / 100) * ev_battery_max_capacity / ev_battery_efficiency  # kWh
            )
            available_oze_energy += - energy_for_ev_charging
            (energy_pv_production, energy_in_storage, missing_energy) = distribute_needed_energy(
                energy_needed=energy_for_ev_charging,
                energy_pv_produced=energy_pv_production,
                energy_in_storage=energy_in_storage,
            )
            energy_from_power_grid += missing_energy
            energy_for_ev_charging_per_id[ev_id] = energy_for_ev_charging


    ev_battery_configuration_per_id = {}
    for ev_id in ev_battery_parameters_per_id:
        ev_battery_configuration = {"InWRte": 0.0, "OutWRte": 0.0}
        energy_for_ev_charging = energy_for_ev_charging_per_id[ev_id]
        if energy_for_ev_charging > 0.0:
            ev_battery_nominal_power = ev_battery_parameters_per_id[ev_id]["nominal_power"]
            ev_battery_configuration["StorCtl_Mod"] = 1
            ev_battery_configuration["InWRte"] = min(
                round(energy_for_ev_charging / (cycle_timedelta_s / 3600) / ev_battery_nominal_power * 100.0
                      + delta_charging_power_perc, 2),
                100.0,
            )
        else:
            ev_battery_configuration["StorCtl_Mod"] = 0
        ev_battery_configuration_per_id[ev_id] = ev_battery_configuration

    # Energy storage
    energy_in_storage += storage_efficiency * energy_pv_production
    energy_in_storage = min(energy_in_storage, storage_max_capacity)
    energy_storage_configuration = {"InWRte": 0.0, "OutWRte": 0.0}
    if energy_in_storage > initial_energy_in_storage:
        energy_storage_configuration["StorCtl_Mod"] = 1
        energy_to_charge_storage = (energy_in_storage - initial_energy_in_storage) / storage_efficiency
        energy_storage_configuration["InWRte"] = min(
            round(energy_to_charge_storage / (cycle_timedelta_s / 3600) / storage_nominal_power * 100.0
                  + delta_charging_power_perc, 2),
            100.0,
        )
    else:
        energy_storage_configuration["StorCtl_Mod"] = 2
        energy_to_discharge_storage = initial_energy_in_storage - energy_in_storage
        energy_storage_configuration["OutWRte"] = min(
            round(energy_to_discharge_storage / (cycle_timedelta_s / 3600) / storage_nominal_power * 100.0
                  + delta_charging_power_perc, 2),
            100.0,
        )

    return (
        temp_configuration,
        json.dumps(energy_storage_configuration),
        json.dumps(ev_battery_configuration_per_id),
    )
