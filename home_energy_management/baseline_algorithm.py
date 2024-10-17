def make_decision(
        timestamp,
        trained_model,
        home_model_parameters: dict[str, float],
        storage_parameters: dict[str, float],
        ev_battery_parameters: dict[str, float | int],
        room_heating_params_list: list[dict],
        energy_pv_production: float,
        uncontrolled_energy_consumption: float,
        temp_outside: float,
        cycle_timedelta_s: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    The algorithm retrieves information about all controller settings and current values. Based on them, it determines
    parameters for the next cycle.
    Implemented function can be used locally or offloaded to COGNIT server.

    Args:
        timestamp: For this function it is ignored as not needed.
        trained_model: For this function it is ignored as not needed.
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
        uncontrolled_energy_consumption (float): Prediction of uncontrolled energy consumption for cycle in kWh.
        energy_pv_production (float): Prediction of energy production from PV matrix for cycle in kWh.
        temp_outside (float): Current outdoor temperature in °C.
        cycle_timedelta_s (int): Time duration of one cycle in seconds.

    Returns:
        Configurations for next cycle:
        - configuration of temperature per room in °C,
        - configuration of energy storage (charging and discharging power limits [percent of nominal power], mode),
        - configuration of EV battery (charging and discharging power limits [percent of nominal power], mode).
    """
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
            room_temp_without_heating = (room_temp - (room_temp - temp_outside)
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
        energy_needed=uncontrolled_energy_consumption,
        energy_pv_produced=energy_pv_production,
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
