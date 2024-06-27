def run_one_step(
        model_parameters: dict[str, float],
        step_timedelta_s: int,
        storage_parameters: dict[str, float],
        ev_battery_parameters: dict[str, float | int],
        room_heating_params_list: list[dict],
        energy_drawn_from_grid: float,
        energy_returned_to_grid: float,
        energy_pv_produced: float,
        temp_outdoor: float,
        charge_level_of_storage: float,
        prev_charge_level_of_storage: float,
        charge_level_of_ev_battery: float,
        prev_charge_level_of_ev_battery: float,
        heating_status_per_room: dict[str, list[bool]],
        temp_per_room: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], float, float, float]:
    """
    The algorithm retrieves information about all controller settings and current values. Based on them, it determines
    parameters for the next time step and calculates energy transmission in the household.
    Implemented function can be used locally or offloaded to COGNIT server.

    Args:
        model_parameters (dict[str, float]): Parameters defining the home energy management model; dict with values for
            keys: heating_delta_temperature, heating_coefficient, heat_loss_coefficient, heat_capacity,
            delta_charging_power_perc.
        step_timedelta_s (int): Time in seconds between each step of the simulation.
        storage_parameters (dict[str, float]): Parameters defining the energy storage model; dict with values for keys:
            max_capacity, min_charge_level, efficiency, nominal_power.
        ev_battery_parameters (dict[str, float]): Parameters defining the EV battery model; dict with values for keys:
            max_capacity, charged_level, efficiency, nominal_power, is_available, time_until_charged.
        room_heating_params_list (list[dict]): Parameters defining the heating model for individual rooms; list with
            dicts, each containing values for keys: name, powers_of_heating_devices.
        energy_drawn_from_grid (float): Active energy drawn from the grid in the previous step in kWh.
        energy_returned_to_grid (float): Active energy returned to the grid in the previous step in kWh.
        energy_pv_produced (float): Energy produced by PV matrix in the previous step in kWh.
        temp_outdoor (float): Current outdoor temperature in °C.
        charge_level_of_storage (float): Current charge level of storage in %.
        prev_charge_level_of_storage (float): Charge level of storage before previous step in %.
        charge_level_of_ev_battery (float): Current charge level of EV battery in %.
        prev_charge_level_of_ev_battery (float): Charge level of EV battery before previous step in %.
        heating_status_per_room (dict[str, list[bool]]): Statuses of heating devices switches; dict with list of
            booleans representing status per key which is room name.
        temp_per_room (dict[str, float]): Measured temperature per room in °C.

    Returns:
        Configurations and predicted state after cycle:
        - configuration of temperature per room in °C,
        - configuration of energy storage (charging and discharging power limits [percent of nominal power], mode),
        - configuration of EV battery (charging and discharging power limits [percent of nominal power], mode),
        - predicted temperature per room in °C,
        - predicted charge level of energy storage in %,
        - predicted charge level of EV battery in %,
        - predicted energy needed from power grid in kWh.
    """

    def check_heating_conditions(
            rooms_list: list[dict],
            temp_per_room: dict[str, float],
            temp_outdoor: float,
            reduction_of_allowed_temp: float = 0.0,
            available_energy: float | None = None,
    ) -> tuple[float, dict[str, float], dict[str, float], list[dict]]:
        energy_for_heating = 0.0  # kWh
        conf_temp_per_room = {}
        new_temp_per_room = {}
        remaining_rooms_list = []
        for room_heating_params in rooms_list:
            room_name = room_heating_params["name"]
            room_temp = temp_per_room[room_name]
            temp_diff = room_heating_params["preferred_temp"] - reduction_of_allowed_temp - room_temp
            if temp_diff > 0:
                energy_for_room_heating = (sum(room_heating_params["powers_of_heating_devices"])
                                           * step_timedelta_s / 3600)  # kWh
                if available_energy:
                    if energy_for_room_heating > available_energy:
                        remaining_rooms_list.append(room_heating_params)
                        continue
                    else:
                        available_energy -= energy_for_room_heating
                conf_temp_per_room[room_name] = room_temp + 2 * temp_diff
                new_temp_per_room[room_name] = room_temp + (
                        heating_coeff * energy_for_room_heating * 3.6e6  # J
                        - (room_temp - temp_outdoor) * heat_loss_coeff * step_timedelta_s  # J
                ) / heat_capacity
                energy_for_heating += energy_for_room_heating
            else:
                remaining_rooms_list.append(room_heating_params)

        return energy_for_heating, conf_temp_per_room, new_temp_per_room, remaining_rooms_list

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
        if min_charging_time < time_until_ev_charged - step_timedelta_s:
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

    delta_temp = model_parameters["heating_delta_temperature"]
    heating_coeff = model_parameters["heating_coefficient"]
    heat_loss_coeff = model_parameters["heat_loss_coefficient"]
    heat_capacity = model_parameters["heat_capacity"]
    delta_charging_power_perc = model_parameters["delta_charging_power_perc"]

    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_nominal_power = storage_parameters["nominal_power"]

    ev_battery_max_capacity = ev_battery_parameters["max_capacity"]
    ev_battery_charged_level = ev_battery_parameters["charged_level"]
    ev_battery_efficiency = ev_battery_parameters["efficiency"]
    ev_battery_nominal_power = ev_battery_parameters["nominal_power"]
    is_ev_available = ev_battery_parameters["is_available"]
    time_until_ev_charged = ev_battery_parameters["time_until_charged"]

    energy_consumption = (
            energy_drawn_from_grid
            + energy_pv_produced
            - energy_returned_to_grid
            - (charge_level_of_storage - prev_charge_level_of_storage) / 100 * storage_max_capacity
            - (charge_level_of_ev_battery - prev_charge_level_of_ev_battery) / 100 * ev_battery_max_capacity
    )
    energy_in_storage = (charge_level_of_storage - storage_min_charge_level) / 100 * storage_max_capacity
    initial_energy_in_storage = energy_in_storage

    total_power_used_for_heating = 0.0
    for room_heating_params in room_heating_params_list:
        total_power_used_for_heating += sum(
            power * status
            for power, status in zip(
                room_heating_params["powers_of_heating_devices"], heating_status_per_room[room_heating_params["name"]]
            )
        )
    energy_consumption_without_heating = max(energy_consumption
                                             - total_power_used_for_heating * step_timedelta_s / 3600, 0.0)

    (energy_pv_produced, energy_in_storage, missing_energy) = distribute_needed_energy(
        energy_needed=energy_consumption_without_heating,
        energy_pv_produced=energy_pv_produced,
        energy_in_storage=energy_in_storage,
    )

    # Home heating - necessary
    (energy_for_necessary_heating, conf_per_room, temp_per_room_pred, remaining_rooms_list) = check_heating_conditions(
        rooms_list=room_heating_params_list,
        temp_per_room=temp_per_room,
        temp_outdoor=temp_outdoor,
        reduction_of_allowed_temp=delta_temp,
    )
    temp_per_room_configuration = conf_per_room
    next_temp_per_room = temp_per_room_pred
    (energy_pv_produced, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
        energy_needed=energy_for_necessary_heating,
        energy_pv_produced=energy_pv_produced,
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
            ev_battery_nominal_power * step_timedelta_s / 3600,  # kWh
            (1 - charge_level_of_ev_battery / 100) * ev_battery_max_capacity / ev_battery_efficiency
        )
        (energy_pv_produced, energy_in_storage, missing_energy) = distribute_needed_energy(
            energy_needed=energy_for_ev_charging,
            energy_pv_produced=energy_pv_produced,
            energy_in_storage=energy_in_storage,
        )
        energy_from_power_grid += missing_energy

    # Home heating - optional
    available_oze_energy = energy_pv_produced + energy_in_storage
    if available_oze_energy > 0.0 and len(remaining_rooms_list) > 0:
        (energy_for_optional_heating, conf_per_room, temp_per_room_pred, remaining_rooms_list) = (
            check_heating_conditions(
                rooms_list=remaining_rooms_list,
                temp_per_room=temp_per_room,
                temp_outdoor=temp_outdoor,
                available_energy=available_oze_energy,
            )
        )
        temp_per_room_configuration.update(conf_per_room)
        next_temp_per_room.update(temp_per_room_pred)
        (energy_pv_produced, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_for_optional_heating,
            energy_pv_produced=energy_pv_produced,
            energy_in_storage=energy_in_storage,
        )

    # Home heating - additional
    available_oze_energy = energy_pv_produced + max(
        energy_in_storage - model_parameters["storage_high_charge_level"] / 100 * storage_max_capacity,
        0.0
    )
    if available_oze_energy > 0.0 and len(remaining_rooms_list) > 0:
        (energy_for_additional_heating, conf_per_room, temp_per_room_pred, remaining_rooms_list) = (
            check_heating_conditions(
                rooms_list=remaining_rooms_list,
                temp_per_room=temp_per_room,
                temp_outdoor=temp_outdoor,
                reduction_of_allowed_temp=-delta_temp,
                available_energy=available_oze_energy,
            )
        )
        temp_per_room_configuration.update(conf_per_room)
        next_temp_per_room.update(temp_per_room_pred)
        (energy_pv_produced, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_for_additional_heating,
            energy_pv_produced=energy_pv_produced,
            energy_in_storage=energy_in_storage,
        )

    for room in remaining_rooms_list:
        room_name = room["name"]
        temp_per_room_configuration[room_name] = room["preferred_temp"]
        room_temp = temp_per_room[room_name]
        next_temp_per_room[room_name] = (
                room_temp - (room_temp - temp_outdoor) * heat_loss_coeff * step_timedelta_s / heat_capacity
        )

    # Electric vehicle
    available_oze_energy = energy_pv_produced + energy_in_storage
    if is_ev_available and energy_for_ev_charging == 0.0 and available_oze_energy > 0.0:
        energy_for_ev_charging = min(
            available_oze_energy,
            ev_battery_nominal_power * step_timedelta_s / 3600,  # kWh
            (1 - charge_level_of_ev_battery / 100) * ev_battery_max_capacity / ev_battery_efficiency  # kWh
        )
        (energy_pv_produced, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
            energy_needed=energy_for_ev_charging,
            energy_pv_produced=energy_pv_produced,
            energy_in_storage=energy_in_storage,
        )

    next_charge_level_of_ev_battery = (
            charge_level_of_ev_battery
            + ev_battery_efficiency * energy_for_ev_charging / ev_battery_max_capacity * 100.
    )
    ev_battery_configuration = {"InWRte": 0.0, "OutWRte": 0.0}
    if energy_for_ev_charging > 0.0:
        ev_battery_configuration["StorCtl_Mod"] = 1
        ev_battery_configuration["InWRte"] = min(
            round(energy_for_ev_charging / (step_timedelta_s / 3600) / ev_battery_nominal_power * 100.0
                  + delta_charging_power_perc, 2),
            100.0,
        )
    else:
        ev_battery_configuration["StorCtl_Mod"] = 0

    # Energy storage
    energy_in_storage += storage_efficiency * energy_pv_produced
    next_charge_level_of_storage = min(
        energy_in_storage / storage_max_capacity * 100.0 + storage_min_charge_level, 100.0
    )
    energy_storage_configuration = {"InWRte": 0.0, "OutWRte": 0.0}
    if energy_in_storage > initial_energy_in_storage:
        energy_storage_configuration["StorCtl_Mod"] = 1
        energy_to_charge_storage = (energy_in_storage - initial_energy_in_storage) / storage_efficiency
        energy_storage_configuration["InWRte"] = min(
            round(energy_to_charge_storage / (step_timedelta_s / 3600) / storage_nominal_power * 100.0
                  + delta_charging_power_perc, 2),
            100.0,
        )
    else:
        energy_storage_configuration["StorCtl_Mod"] = 2
        energy_to_discharge_storage = initial_energy_in_storage - energy_in_storage
        energy_storage_configuration["OutWRte"] = min(
            round(energy_to_discharge_storage / (step_timedelta_s / 3600) / storage_nominal_power * 100.0
                  + delta_charging_power_perc, 2),
            100.0,
        )

    return (
        temp_per_room_configuration,
        energy_storage_configuration,
        ev_battery_configuration,
        next_temp_per_room,
        next_charge_level_of_storage,
        next_charge_level_of_ev_battery,
        energy_from_power_grid,
    )
