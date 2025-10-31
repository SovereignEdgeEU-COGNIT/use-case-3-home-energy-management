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
        besmart_parameters (str): JSON with parameters used for authentication to besmart.energy API and downloading data;
            dict with values for keys: workspace_key, login, password, pv_generation, energy_consumption,
            temperature_moid.
        home_model_parameters (str): JSON with parameters defining the home energy management model; dict with values
            for keys: temp_window, heating_coefficient, heat_loss_coefficient, heat_capacity.
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

    def get_data_from_besmart(
            cid: int,
            mid: int,
            moid: int,
            is_cumulative: bool = False
    ) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        since_datetime = np.datetime64(int(state_datetime.timestamp()), 's')
        if is_cumulative:
            since_datetime -= np.timedelta64(3600, 's')
        till_datetime = np.datetime64(int(state_datetime.timestamp()), 's') + np.timedelta64(cycle_timedelta_s, 's')
        body = [{
            "client_cid": cid,
            "sensor_mid": mid,
            "signal_type_moid": moid,
            "since": int(since_datetime.astype(int) * 1000),
            "till": int(till_datetime.astype(int) * 1000),
            "get_last": True,
        }]
        res = requests.post(
            'https://api.besmart.energy/api/sensors/signals/data',
            headers=headers, json=body
        )
        return res.json()[0]['data']

    def get_energy_data(
            identifier: dict[str, int],
            is_cumulative: bool = False
    ) -> float:
        data = get_data_from_besmart(identifier["cid"],
                                     identifier["mid"],
                                     identifier["moid"],
                                     is_cumulative)
        value = np.array(data['value'])
        origin = np.array(data['origin'])

        pred_value = value[origin == 2]
        if is_cumulative:
            pred_time = (np.array(data['time']) / 1e3).astype(int)[origin == 2]
            state_index = np.where(pred_time >= state_datetime.timestamp())[0][0]
            pred_value = pred_value[state_index - 1:state_index + 1]
            pred_time = pred_time[state_index - 1:state_index + 1]
            pred_value = np.diff(pred_value) / (np.diff(pred_time) / 3600)

        if len(pred_value) < 1:
            raise Exception('Not enough data for decision-making')

        return pred_value[0]

    def get_temperature_data() -> float:
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

        value = np.array(data['value'])
        origin = np.array(data['origin'])
        estm_value = value[origin == 3]
        if len(estm_value) < 1:
            raise Exception('Not enough data for decision-making')

        return estm_value[0] - 272.15

    def check_heating_conditions(
            heating_params: dict,
            reduction_of_allowed_temp: float = 0.0,
            available_energy: float | None = None,
    ) -> tuple[float, float]:
        energy_for_heating = 0.0  # kWh
        current_temp = heating_params["curr_temp"]
        conf_temp = current_temp
        temp_without_heating = (current_temp - (current_temp - temp_outside_pred)
                                * heat_loss_coeff * cycle_timedelta_s / heat_capacity)
        temp_diff = heating_params["preferred_temp"] - reduction_of_allowed_temp - temp_without_heating
        if temp_diff > 0:
            energy_for_heating = (sum(heating_params["powers_of_heating_devices"])
                                       * cycle_timedelta_s / 3600)  # kWh
            if available_energy is None or energy_for_heating < available_energy:
                conf_temp += 2 * temp_diff
            else:
                energy_for_heating = 0.0

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
    pv_generation_pred = get_energy_data(besmart_parameters["pv_generation"])
    energy_consumption_pred = get_energy_data(besmart_parameters["energy_consumption"], True)
    temp_outside_pred = get_temperature_data()

    temp_window = home_model_parameters["temp_window"]
    heat_loss_coeff = home_model_parameters["heat_loss_coefficient"]
    heat_capacity = home_model_parameters["heat_capacity"]

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
                round(energy_for_ev_charging / (cycle_timedelta_s / 3600) / ev_battery_nominal_power * 100.0, 2),
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
            round(energy_to_charge_storage / (cycle_timedelta_s / 3600) / storage_nominal_power * 100.0, 2),
            100.0,
        )
    else:
        energy_storage_configuration["StorCtl_Mod"] = 2
        energy_to_discharge_storage = initial_energy_in_storage - energy_in_storage
        energy_storage_configuration["OutWRte"] = min(
            round(energy_to_discharge_storage / (cycle_timedelta_s / 3600) / storage_nominal_power * 100.0, 2),
            100.0,
        )

    return (
        temp_configuration,
        json.dumps(energy_storage_configuration),
        json.dumps(ev_battery_configuration_per_id),
    )


def evaluate(
        eval_parameters: str,
        s3_parameters: str,
        besmart_parameters: str,
        home_model_parameters: str,
        storage_parameters: str,
        ev_battery_parameters_per_id: str,
        heating_parameters: str,
        user_preferences: str,
) -> str:
    import datetime
    import json
    from typing import Any

    import numpy as np
    import requests

    def select_action():
        energy_in_storage = (storage_soc - storage_min_charge_level) / 100 * storage_max_capacity
        initial_energy_in_storage = energy_in_storage
        temp_configuration = pref_temperature

        (energy_pv_production, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_consumption,
            energy_pv_produced=pv_generation,
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
        for ev_id, ev_driving_state in ev_driving_state_per_id.items():
            is_ev_available = ev_driving_state["driving_power"] == 0.
            if is_ev_available and does_ev_battery_require_charging(ev_id):
                charge_level_of_ev_battery = ev_soc_per_id[ev_id]
                ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
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
                    ev_driving_state_per_id[ev_id]["driving_power"] == 0.
                    and energy_for_ev_charging_per_id[ev_id] == 0.0
                    and available_oze_energy > 0.0
            ):
                charge_level_of_ev_battery = ev_soc_per_id[ev_id]
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

        ev_charging_power_per_id = []
        for ev_id in ev_id_list:
            energy_for_ev_charging = energy_for_ev_charging_per_id[ev_id]
            if energy_for_ev_charging > 0.0:
                ev_battery_nominal_power = ev_battery_parameters_per_id[ev_id]["nominal_power"]
                ev_charging_power = min(
                    round(energy_for_ev_charging / (cycle_timedelta_s / 3600) / ev_battery_nominal_power * 100.0, 2),
                    100.0,
                ) / 1000. * ev_battery_nominal_power
                ev_charging_power_per_id.append(ev_charging_power)
            else:
                ev_charging_power_per_id.append(0.0)

        # Energy storage
        energy_in_storage += storage_efficiency * energy_pv_production
        energy_in_storage = min(energy_in_storage, storage_max_capacity)
        if energy_in_storage > initial_energy_in_storage:
            energy_to_charge_storage = (energy_in_storage - initial_energy_in_storage) / storage_efficiency
            storage_charging_power = min(
                round(energy_to_charge_storage / (cycle_timedelta_s / 3600) / storage_nominal_power * 100.0, 2),
                100.0,
            ) / 100. * storage_nominal_power
        else:
            energy_to_discharge_storage = initial_energy_in_storage - energy_in_storage
            storage_charging_power = - min(
                round(energy_to_discharge_storage / (cycle_timedelta_s / 3600) / storage_nominal_power * 100.0, 2),
                100.0,
            ) / 100. * storage_nominal_power

        return (
            temp_configuration,
            storage_charging_power,
            ev_charging_power_per_id
        )

    def check_heating_conditions(
            heating_params: dict,
            reduction_of_allowed_temp: float = 0.0,
            available_energy: float | None = None,
    ) -> tuple[float, float]:
        energy_for_heating = 0.0  # kWh
        conf_temp = temp_inside
        temp_without_heating = (temp_inside - (temp_inside - temp_outside)
                                * heat_loss_coefficient * cycle_timedelta_s / heat_capacity)
        temp_diff = pref_temperature - reduction_of_allowed_temp - temp_without_heating
        if temp_diff > 0:
            energy_for_heating = (sum(heating_params["powers_of_heating_devices"])
                                       * cycle_timedelta_s / 3600)  # kWh
            if available_energy is None or energy_for_heating < available_energy:
                conf_temp += 2 * temp_diff
            else:
                energy_for_heating = 0.0

        return energy_for_heating, conf_temp

    def does_ev_battery_require_charging(ev_id: int) -> bool:
        charge_level_of_ev_battery = ev_soc_per_id[ev_id]
        ev_battery_params = ev_battery_parameters_per_id[ev_id]
        ev_battery_max_capacity = ev_battery_params["max_capacity"]
        ev_battery_charged_level = ev_battery_params["driving_charge_level"]
        ev_battery_efficiency = ev_battery_params["efficiency"]
        ev_battery_nominal_power = ev_battery_params["nominal_power"]
        time_until_ev_charged = ev_driving_state_per_id[ev_id]["time_till_departure"]

        ev_charging_capacity = ((ev_battery_charged_level - charge_level_of_ev_battery)
                                / 100 * ev_battery_max_capacity / ev_battery_efficiency)
        if ev_charging_capacity < 0:
            return False
        min_charging_time = ev_charging_capacity / ev_battery_nominal_power * 3600  # s
        if min_charging_time < time_until_ev_charged.seconds - cycle_timedelta_s:
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

    def get_ev_driving_state(ev_driving_schedule: dict[str, Any]) -> dict[str, Any]:
        ev_driving_time = ev_driving_schedule["time"]
        ev_schedule_ind = np.where(time >= ev_driving_time)[0][-1]
        ev_driving_power = ev_driving_schedule["driving_power"][ev_schedule_ind]

        next_driving_power_arr = np.array(ev_driving_schedule["driving_power"][ev_schedule_ind + 1:]
                                          + ev_driving_schedule["driving_power"][:ev_schedule_ind + 1])
        next_driving_time_arr = np.concatenate((ev_driving_time[ev_schedule_ind + 1:],
                                                ev_driving_time[:ev_schedule_ind + 1]))
        next_ev_departure_time = next_driving_time_arr[np.where(next_driving_power_arr > 0.)[0][0]]
        next_ev_departure_timestamp = datetime.datetime.strptime(
            f"{next_ev_departure_time.hour}:{next_ev_departure_time.minute}", "%H:%M")
        if next_ev_departure_time < time:
            next_ev_departure_timestamp = next_ev_departure_timestamp + datetime.timedelta(days=1)
        time_till_ev_departure = (next_ev_departure_timestamp
                                  - datetime.datetime.strptime(f"{time.hour}:{time.minute}", "%H:%M"))

        return {
            "driving_power": ev_driving_power,
            "time_till_departure": time_till_ev_departure,
        }

    def get_preferred_temperature() -> float:
        pref_temp_schedule_ind = np.where(time >= pref_temp_schedule_time)[0][-1]
        return pref_temp_schedule["temp"][pref_temp_schedule_ind]

    def get_reward(
            controlled_consumption_t: float,
            temp_inside_t: float,
            storage_soc_t: float,
            ev_soc_per_id_t: dict[int, float],
            dt: int
    ) -> tuple[float, float]:
        energy_balance = pv_generation - energy_consumption - controlled_consumption_t
        temperature_error = max(np.abs(temp_inside_t - pref_temperature) - temp_window, 0.)
        storage_soc_error = (max(storage_soc_t - 100., 0.)
                             + max(storage_min_charge_level - storage_soc_t, 0.)
                             ) / 100. * storage_max_capacity

        ev_soc_error = 0
        for ev_id in ev_id_list:
            ev_driving_state = ev_driving_state_per_id[ev_id]
            ev_driving_power = ev_driving_state["driving_power"]
            time_till_ev_departure = ev_driving_state["time_till_departure"].seconds
            ev_soc_t = ev_soc_per_id_t[ev_id]
            if ev_driving_power == 0.:
                ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
                ev_min_charge_level = ev_battery_parameters["min_charge_level"]
                ev_max_capacity = ev_battery_parameters["max_capacity"]
                ev_soc_error += (max(ev_soc_t - 100., 0.)
                                 + max(ev_min_charge_level - ev_soc_t, 0.)
                                 ) / 100. * ev_max_capacity
                if time_till_ev_departure <= dt:
                    ev_driving_charge_level = ev_battery_parameters["driving_charge_level"]
                    ev_soc_error += max(ev_driving_charge_level - ev_soc_t, 0.) / 100. * ev_max_capacity


        energy_balance_reward = - energy_reward_coeff * np.abs(energy_balance)
        temperature_reward = - temp_reward_coeff * temperature_error
        storage_reward = - storage_reward_coeff * storage_soc_error
        ev_reward = - ev_reward_coeff * ev_soc_error
        return (
            energy_balance_reward + temperature_reward + storage_reward + ev_reward,
            np.abs(energy_balance)
        )

    def step(
            actions: tuple[float, ...],
            temp_inside_t: float,
            storage_soc_t: float,
            ev_soc_per_id_t: dict[int, float],
            dt: int
    ) -> tuple[float, float, float, float, dict, bool]:
        temp_setting = actions[0]
        storage_charging_power = actions[1]
        ev_charging_power_list = actions[2]

        delta_temp = temp_inside_t - temp_setting
        if abs(delta_temp) > temp_window:
            next_is_heating_on = delta_temp < 0
        else:
            next_is_heating_on = is_heating_on
        heating_energy = (
                heating_coefficient * next_is_heating_on * heating_devices_power * 1000 * dt
                - heat_loss_coefficient * (temp_inside_t - temp_outside) * dt
        )  # J
        delta_temp = heating_energy / heat_capacity
        next_temp_inside = temp_inside_t + delta_temp
        heating_consumption = next_is_heating_on * heating_devices_power * dt / 3600

        storage_power_reduction = min(1.0, max(epsilon, (100. - storage_soc_t) / (100. - storage_charging_switch_level)))
        delta_capacity = storage_charging_power * dt / 3600 * (
            storage_efficiency * storage_power_reduction if storage_charging_power > 0 else 1.)
        next_storage_soc = storage_soc_t + delta_capacity / storage_max_capacity * 100.0
        next_storage_soc = (1.0 - storage_energy_loss * dt / 100.0) * next_storage_soc
        next_storage_soc = min(max(next_storage_soc, epsilon), 100.0)
        real_delta_capacity = (next_storage_soc - storage_soc_t) / 100. * storage_max_capacity
        storage_consumption = real_delta_capacity / (
                storage_power_reduction * storage_efficiency if storage_charging_power > 0 else 1.)

        next_ev_soc_per_id = {}
        ev_consumption = 0.
        for ev_id, ev_charging_power in zip(ev_id_list, ev_charging_power_list):
            ev_driving_power = ev_driving_state_per_id[ev_id]["driving_power"]
            ev_soc_t = ev_soc_per_id_t[ev_id]
            ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
            ev_charging_switch_level = ev_battery_parameters["charging_switch_level"]
            ev_efficiency = ev_battery_parameters["efficiency"]
            ev_max_capacity = ev_battery_parameters["max_capacity"]
            ev_energy_loss = ev_battery_parameters["energy_loss"]
            if ev_driving_power == 0.:
                ev_power_reduction = min(1.0, max(epsilon, (100. - ev_soc_t) / (100. - ev_charging_switch_level)))
                delta_capacity = ev_charging_power * dt / 3600 * ev_efficiency * ev_power_reduction
                next_ev_soc = ev_soc_t + delta_capacity / ev_max_capacity * 100.0
                next_ev_soc = (1.0 - ev_energy_loss * dt / 100.0) * next_ev_soc
                next_ev_soc = min(max(next_ev_soc, epsilon), 100.0)
                real_delta_capacity = (next_ev_soc - ev_soc_t) / 100. * ev_max_capacity
                ev_consumption = real_delta_capacity / (ev_power_reduction * ev_efficiency)
            else:
                next_ev_soc = ev_soc_t - ev_driving_power * dt / 3600 / ev_max_capacity * 100.0
                next_ev_soc = max(next_ev_soc, epsilon)
                ev_consumption = 0.
            next_ev_soc_per_id[ev_id] = next_ev_soc

        controlled_consumption = heating_consumption + storage_consumption + ev_consumption
        reward_t, energy_balance_t = get_reward(
            controlled_consumption,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc_per_id,
            dt
        )

        return (
            reward_t,
            energy_balance_t,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc_per_id,
            next_is_heating_on,
        )

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

    def get_data_from_besmart(
            cid: int,
            mid: int,
            moid: int,
            is_cumulative: bool = False
    ) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        since = int(besmart_parameters["since"]) - cycle_timedelta_s
        if is_cumulative:
            since -= cycle_timedelta_s
        body = [{
            "client_cid": cid,
            "sensor_mid": mid,
            "signal_type_moid": moid,
            "since": since * 1000,
            "till": int(besmart_parameters["till"]) * 1000,
            "get_last": True,
        }]
        res = requests.post(
            'https://api.besmart.energy/api/sensors/signals/data',
            headers=headers, json=body
        )
        return res.json()[0]['data']

    def get_energy_data(
            identifier: dict[str, int],
            is_cumulative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        data = get_data_from_besmart(identifier["cid"],
                                     identifier["mid"],
                                     identifier["moid"],
                                     is_cumulative)
        time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
        value = np.array(data['value'])
        origin = np.array(data['origin'])

        real_value = value[origin == 1]
        real_time = time[origin == 1]
        pred_value = value[origin == 2]
        pred_time = time[origin == 2]

        try:
            real_value, real_time = validate_data(real_time, real_value, is_cumulative)
            pred_value, pred_time = validate_data(pred_time, pred_value, is_cumulative)
        except ValueError:
            raise Exception('Not enough data for training')

        if is_cumulative:
            real_value = np.diff(real_value) / (np.diff(real_time.astype(int)) / 60)
            pred_value = np.diff(pred_value) / (np.diff(pred_time.astype(int)) / 60)

        return real_value, pred_value

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
        params = {
            "since": (int(besmart_parameters["since"]) - cycle_timedelta_s) * 1000,
            "till": int(besmart_parameters["till"]) * 1000,
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
        try:
            pred_value, _ = validate_data(estm_time, estm_value)
        except ValueError:
            raise Exception('Not enough data for training')

        return pred_value - 272.15

    def validate_data(
            time: np.ndarray,
            value: np.ndarray,
            is_cumulative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        since = np.datetime64(int(besmart_parameters["since"]), "s")
        if is_cumulative:
            since -= np.timedelta64(cycle_timedelta_s, 's')
        expected_time = np.arange(since,
                                  np.datetime64(int(besmart_parameters["till"]), "s"),
                                  np.timedelta64(cycle_timedelta_s, 's')).astype('datetime64[m]')
        missing_time = np.array([t for t in expected_time if t not in time])
        num_missing = len(missing_time)
        if num_missing > 0:
            new_time = np.concatenate((time, missing_time))
            new_value = np.concatenate((value, np.array(len(missing_time) * [np.nan])))
            ind = np.argsort(new_time)
            new_time = new_time[ind]
            new_value = new_value[ind]
            missing_data_mask = np.isnan(new_value)
            sequences_last_indexes = np.append(np.where(missing_data_mask[1:] != missing_data_mask[:-1]),
                                               len(missing_data_mask) - 1)
            sequences_lengths = np.diff(np.append(-1, sequences_last_indexes))
            gap_lengths = sequences_lengths[missing_data_mask[sequences_last_indexes]]
            if np.any(gap_lengths > 2):
                raise ValueError
            new_value = np.interp(new_time.astype('float64'),
                                  new_time[~missing_data_mask].astype('float64'),
                                  new_value[~missing_data_mask])
        else:
            new_value = value.copy()
            new_time = time.copy()
        if len(new_time) > len(expected_time):
            new_value = np.array([v for t, v in zip(new_time, new_value) if t in expected_time])

        return new_value, expected_time

    epsilon = 1e-8
    eval_parameters = json.loads(eval_parameters)
    besmart_parameters = json.loads(besmart_parameters)
    home_model_parameters = json.loads(home_model_parameters)
    storage_parameters = json.loads(storage_parameters)
    ev_battery_parameters_per_id = json.loads(ev_battery_parameters_per_id) if (
            ev_battery_parameters_per_id != json.dumps(None)) else {}
    heating_parameters = json.loads(heating_parameters)
    user_preferences = json.loads(user_preferences)

    energy_reward_coeff = eval_parameters["energy_reward_coeff"]
    temp_reward_coeff = eval_parameters["temp_reward_coeff"]
    storage_reward_coeff = eval_parameters["storage_reward_coeff"]
    ev_reward_coeff = eval_parameters["ev_reward_coeff"]

    heating_coefficient = home_model_parameters["heating_coefficient"]
    heat_loss_coefficient = home_model_parameters["heat_loss_coefficient"]
    heat_capacity = home_model_parameters["heat_capacity"]
    temp_window = home_model_parameters["temp_window"]

    heating_devices_power = sum(heating_parameters["powers_of_heating_devices"])

    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_charging_switch_level = storage_parameters["charging_switch_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_energy_loss = storage_parameters["energy_loss"]
    storage_nominal_power = storage_parameters["nominal_power"]

    ev_driving_schedule_per_id = user_preferences["ev_driving_schedule"]
    pref_temp_schedule = user_preferences["pref_temp_schedule"]
    pref_temp_schedule_time = np.array([datetime.datetime.strptime(t, "%H:%M").time()
                                        for t in pref_temp_schedule["time"]])
    cycle_timedelta_s = user_preferences["cycle_timedelta_s"]

    ev_id_list = list(ev_battery_parameters_per_id.keys())
    ev_id_list.sort()
    for ev_driving_schedule_dict in ev_driving_schedule_per_id.values():
        ev_driving_schedule_dict["time"] = np.array([datetime.datetime.strptime(t, "%H:%M").time()
                                                     for t in ev_driving_schedule_dict["time"]])

    timestamps = np.arange(np.datetime64(int(besmart_parameters["since"]), "s"),
                           np.datetime64(int(besmart_parameters["till"]), "s"),
                           datetime.timedelta(seconds=cycle_timedelta_s))

    token = authenticate_to_besmart()
    pv_generation_real, pv_generation_pred = get_energy_data(besmart_parameters["pv_generation"])
    energy_consumption_real, energy_consumption_pred = get_energy_data(besmart_parameters["energy_consumption"], True)
    temp_outside_pred = get_temperature_data()

    storage_soc = round(np.random.uniform(storage_min_charge_level, 100.), 2)
    ev_soc_per_id = {
        ev_id: round(np.random.uniform(ev_battery_parameters["min_charge_level"], 100.), 2)
        for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items()
    }
    is_heating_on = bool(np.random.randint(2))
    number_of_cycles = datetime.timedelta(days=1) // datetime.timedelta(seconds=cycle_timedelta_s)
    remainder_cycles = len(timestamps) % number_of_cycles
    if remainder_cycles != 0:
        timestamps = timestamps[:-remainder_cycles]

    reward_list = []
    energy_balance_list = []
    for i, timestamp in enumerate(timestamps):
        ts = (timestamp - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).time()

        pv_generation = pv_generation_pred[i]
        energy_consumption = energy_consumption_pred[i]
        temp_outside = temp_outside_pred[i]
        pref_temperature = get_preferred_temperature()
        if i == 0:
            temp_inside = pref_temperature + round(np.random.uniform(- temp_window, temp_window), 2)
        ev_driving_state_per_id = {
            ev_id: get_ev_driving_state(ev_driving_schedule_per_id[ev_id]) for ev_id in ev_id_list
        }

        action = select_action()

        pv_generation = pv_generation_real[i]
        energy_consumption = energy_consumption_real[i]

        # Receive state and reward from environment.
        reward, energy_balance_abs, temp_inside, storage_soc, ev_soc_per_id, is_heating_on = step(
            action, temp_inside, storage_soc, ev_soc_per_id, cycle_timedelta_s
        )
        reward_list.append(reward)
        energy_balance_list.append(energy_balance_abs)

    reward_array = np.array(reward_list).reshape(number_of_cycles, -1)
    energy_balance_array = np.array(energy_balance_list).reshape(number_of_cycles, -1)
    return json.dumps(
        {
            'mean_reward': float(np.mean(np.sum(reward_array, axis=0))),
            'mean_energy_balance': float(np.mean(np.sum(energy_balance_array, axis=0))),
        }
    )
