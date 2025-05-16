from typing import Any


def make_decision(
        timestamp: float,
        s3_parameters,
        besmart_parameters: dict[str, Any],
        home_model_parameters: dict[str, float],
        storage_parameters: dict[str, float],
        ev_battery_parameters: dict[str, float | int],
        room_heating_params_list: list[dict],
        cycle_timedelta_s: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    The algorithm retrieves information about all controller settings and current values. Based on them, it determines
    parameters for the next cycle.
    Implemented function can be used locally or offloaded to COGNIT server.

    Args:
        timestamp (float): Current timestamp, for which decision is made.
        s3_parameters: For this implementation parameter is ignored as not needed.
        besmart_parameters (dict[str, Any]): Parameters used for authentication to besmart API and downloading data;
            dict with values for keys: workspace_key, login, password, pv_generation, energy_consumption,
            temperature_moid.
        home_model_parameters (dict[str, float]): Parameters defining the home energy management model; dict with values
            for keys: heating_delta_temperature, heating_coefficient, heat_loss_coefficient, heat_capacity,
            delta_charging_power_perc.
        storage_parameters (dict[str, float]): Parameters defining the energy storage model; dict with values for keys:
            max_capacity, min_charge_level, efficiency, nominal_power, curr_charge_level.
        ev_battery_parameters (dict[str, float]): Parameters defining the EV battery model; dict with values for keys:
            max_capacity, driving_charge_level, efficiency, nominal_power, is_available, time_until_charged,
            curr_charge_level.
        room_heating_params_list (list[dict]): Parameters defining the heating model for individual rooms; list with
            dicts, each containing values for keys: name, curr_temp, preferred_temp, powers_of_heating_devices,
            is_device_switch_on.
        cycle_timedelta_s (int): Time duration of one cycle in seconds.

    Returns:
        Configurations for next cycle:
        - configuration of temperature per room in °C,
        - configuration of energy storage (charging and discharging power limits [percent of nominal power], mode),
        - configuration of EV battery (charging and discharging power limits [percent of nominal power], mode).
    """
    import datetime

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
                                     identifier["moid"],)
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
            heating_params_list: list[dict],
            reduction_of_allowed_temp: float = 0.0,
            available_energy: float | None = None,
    ) -> tuple[float, dict[str, float], list[dict]]:
        energy_for_heating = 0.0  # kWh
        conf_temp_per_room = {}
        remaining_rooms_list = []
        for room_heating_params in heating_params_list:
            room_name = room_heating_params["name"]
            room_temp = room_heating_params["curr_temp"]
            room_temp_without_heating = (room_temp - (room_temp - temp_outside_pred)
                                         * heat_loss_coeff * cycle_timedelta_s / heat_capacity)
            temp_diff = room_heating_params["preferred_temp"] - reduction_of_allowed_temp - room_temp_without_heating
            if temp_diff > 0:
                energy_for_room_heating = (sum(room_heating_params["powers_of_heating_devices"])
                                           * cycle_timedelta_s / 3600)  # kWh
                if available_energy:
                    if energy_for_room_heating > available_energy:
                        remaining_rooms_list.append(room_heating_params)
                        continue
                    else:
                        available_energy -= energy_for_room_heating
                conf_temp_per_room[room_name] = room_temp + 2 * temp_diff
                energy_for_heating += energy_for_room_heating
            else:
                remaining_rooms_list.append(room_heating_params)

        return energy_for_heating, conf_temp_per_room, remaining_rooms_list

    def does_ev_battery_require_charging(
            charge_level_of_ev_battery: float,
            ev_battery_charged_level: float,
            ev_battery_max_capacity: float,
            ev_battery_efficiency: float,
            ev_battery_nominal_power: float,
            time_until_ev_charged: int,
    ) -> bool:
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


    state_datetime = datetime.datetime.fromtimestamp(timestamp)
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

    delta_temp = home_model_parameters["heating_delta_temperature"]
    heat_loss_coeff = home_model_parameters["heat_loss_coefficient"]
    heat_capacity = home_model_parameters["heat_capacity"]
    delta_charging_power_perc = home_model_parameters["delta_charging_power_perc"]

    charge_level_of_storage = storage_parameters["curr_charge_level"]
    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_nominal_power = storage_parameters["nominal_power"]

    charge_level_of_ev_battery = ev_battery_parameters["curr_charge_level"]
    ev_battery_max_capacity = ev_battery_parameters["max_capacity"]
    ev_battery_charged_level = ev_battery_parameters["driving_charge_level"]
    ev_battery_efficiency = ev_battery_parameters["efficiency"]
    ev_battery_nominal_power = ev_battery_parameters["nominal_power"]
    is_ev_available = ev_battery_parameters["is_available"]
    time_until_ev_charged = ev_battery_parameters["time_until_charged"]

    energy_in_storage = (charge_level_of_storage - storage_min_charge_level) / 100 * storage_max_capacity
    initial_energy_in_storage = energy_in_storage

    (energy_pv_production, energy_in_storage, missing_energy) = distribute_needed_energy(
        energy_needed=energy_consumption_pred,
        energy_pv_produced=pv_generation_pred,
        energy_in_storage=energy_in_storage,
    )

    # Home heating - necessary
    (energy_for_necessary_heating, conf_per_room, remaining_rooms_list) = check_heating_conditions(
        heating_params_list=room_heating_params_list,
        reduction_of_allowed_temp=delta_temp,
    )
    temp_per_room_configuration = conf_per_room
    (energy_pv_production, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
        energy_needed=energy_for_necessary_heating,
        energy_pv_produced=energy_pv_production,
        energy_in_storage=energy_in_storage,
    )
    energy_from_power_grid += missing_energy

    # EV charging - required
    energy_for_ev_charging = 0.0
    if is_ev_available and does_ev_battery_require_charging(
            charge_level_of_ev_battery=charge_level_of_ev_battery,
            ev_battery_charged_level=ev_battery_charged_level,
            ev_battery_max_capacity=ev_battery_max_capacity,
            ev_battery_efficiency=ev_battery_efficiency,
            ev_battery_nominal_power=ev_battery_nominal_power,
            time_until_ev_charged=time_until_ev_charged,
    ):
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

    # Home heating - optional
    available_oze_energy = energy_pv_production + energy_in_storage
    if available_oze_energy > 0.0 and len(remaining_rooms_list) > 0:
        (energy_for_optional_heating, conf_per_room, remaining_rooms_list) = (
            check_heating_conditions(
                heating_params_list=remaining_rooms_list,
                available_energy=available_oze_energy,
            )
        )
        temp_per_room_configuration.update(conf_per_room)
        (energy_pv_production, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_for_optional_heating,
            energy_pv_produced=energy_pv_production,
            energy_in_storage=energy_in_storage,
        )

    # Home heating - additional
    available_oze_energy = energy_pv_production + max(
        energy_in_storage - home_model_parameters["storage_high_charge_level"] / 100 * storage_max_capacity,
        0.0
    )
    if available_oze_energy > 0.0 and len(remaining_rooms_list) > 0:
        (energy_for_additional_heating, conf_per_room, remaining_rooms_list) = (
            check_heating_conditions(
                heating_params_list=remaining_rooms_list,
                reduction_of_allowed_temp=-delta_temp,
                available_energy=available_oze_energy,
            )
        )
        temp_per_room_configuration.update(conf_per_room)
        (energy_pv_production, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_for_additional_heating,
            energy_pv_produced=energy_pv_production,
            energy_in_storage=energy_in_storage,
        )

    for room in remaining_rooms_list:
        temp_per_room_configuration[room["name"]] = room["preferred_temp"]

    # Electric vehicle
    available_oze_energy = energy_pv_production + energy_in_storage
    if is_ev_available and energy_for_ev_charging == 0.0 and available_oze_energy > 0.0:
        energy_for_ev_charging = min(
            available_oze_energy,
            ev_battery_nominal_power * cycle_timedelta_s / 3600,  # kWh
            (1 - charge_level_of_ev_battery / 100) * ev_battery_max_capacity / ev_battery_efficiency  # kWh
        )
        (energy_pv_production, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_for_ev_charging,
            energy_pv_produced=energy_pv_production,
            energy_in_storage=energy_in_storage,
        )

    ev_battery_configuration = {"InWRte": 0.0, "OutWRte": 0.0}
    if energy_for_ev_charging > 0.0:
        ev_battery_configuration["StorCtl_Mod"] = 1
        ev_battery_configuration["InWRte"] = min(
            round(energy_for_ev_charging / (cycle_timedelta_s / 3600) / ev_battery_nominal_power * 100.0
                  + delta_charging_power_perc, 2),
            100.0,
        )
    else:
        ev_battery_configuration["StorCtl_Mod"] = 0

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
        temp_per_room_configuration,
        energy_storage_configuration,
        ev_battery_configuration,
    )
