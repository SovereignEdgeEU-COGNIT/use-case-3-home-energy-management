def run_one_step(
        model_parameters: dict[str, float],
        step_timedelta_s: int,
        storage_parameters: dict[str, float],
        room_heating_params_list: list[dict],
        energy_drawn_from_grid: float,
        energy_returned_to_grid: float,
        energy_pv_produced: float,
        temp_outdoor: float,
        charge_level_of_storage: float,
        prev_charge_level_of_storage: float,
        heating_status_per_room: dict[str, list[bool]],
        temp_per_room: dict[str, float]
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float]:
    """
    The naive version of decision algorithm - it determines parameters for the next time step based on devices standard
    functioning without any optimization.
    Implemented function can be used locally or offloaded to COGNIT server.

    Args:
        model_parameters (dict[str, float]): Parameters defining the home energy management model; dict with values for
            keys: heating_delta_temperature, heating_coefficient, heat_loss_coefficient, heat_capacity,
            storage_delta_power_perc.
        step_timedelta_s (int): Time in seconds between each step of the simulation.
        storage_parameters (dict[str, float]): Parameters defining the energy storage model; dict with values for keys:
            max_capacity, min_charge_level, efficiency, nominal_power.
        room_heating_params_list (list[dict]): Parameters defining the heating model for individual rooms; list with
            dicts, each containing values for keys: name, powers_of_heating_devices.
        energy_drawn_from_grid (float): Active energy drawn from the grid in the previous step in kWh.
        energy_returned_to_grid (float): Active energy returned to the grid in the previous step in kWh.
        energy_pv_produced (float): Energy produced by PV matrix in the previous step in kWh.
        temp_outdoor (float): Current outdoor temperature in °C.
        charge_level_of_storage (float): Current charge level of storage in %.
        prev_charge_level_of_storage (float): Charge level of storage before previous step in %.
        heating_status_per_room (dict[str, list[bool]]): Statuses of heating devices switches; dict with list of
            booleans representing status per key which is room name.
        temp_per_room (dict[str, float]): Measured temperature per room in °C.

    Returns:
        Configurations and predicted state after cycle:
        - configuration of temperature per room
        - configuration of energy storage (charging and discharging power limits [percent of nominal power], mode)
        - predicted temperature per room
        - predicted charge level of energy storage
        - predicted energy needed from power grid.
    """

    def check_heating_conditions(
            rooms_list: list[dict],
            heating_status_per_room: dict[str, list[bool]],
            temp_per_room: dict[str, float],
            temp_outdoor: float,
            delta_temp: float
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        energy_for_heating = 0.  # kWh
        conf_temp_per_room = {}
        new_temp_per_room = {}
        for room_heating_params in rooms_list:
            room_name = room_heating_params["name"]
            room_temp = temp_per_room[room_name]
            status_list = heating_status_per_room[room_name]
            preferred_temp = room_heating_params["preferred_temp"]
            temp_diff = abs(preferred_temp - room_temp)
            if temp_diff > delta_temp:
                if room_temp > preferred_temp:
                    status_list = [False] * len(status_list)
                else:
                    status_list = [True] * len(status_list)

            energy_for_room_heating = sum(status * power for status, power
                                          in zip(status_list, room_heating_params["powers_of_heating_devices"]))
            conf_temp_per_room[room_name] = preferred_temp
            new_temp_per_room[room_name] = room_temp + (
                    heating_coeff * energy_for_room_heating * 3.6e6  # J
                    - (room_temp - temp_outdoor) * heat_loss_coeff * step_timedelta_s  # J
            ) / heat_capacity
            energy_for_heating += energy_for_room_heating

        return energy_for_heating, conf_temp_per_room, new_temp_per_room

    def distribute_needed_energy(
            energy_needed: float,
            energy_pv_produced: float,
            energy_in_storage: float
    ) -> tuple[float, float, float]:
        if energy_pv_produced > energy_needed:
            return energy_pv_produced - energy_needed, energy_in_storage, 0.

        if energy_in_storage > (energy_needed - energy_pv_produced):
            return 0., energy_in_storage - (energy_needed - energy_pv_produced), 0.

        return 0., 0., energy_needed - energy_pv_produced - energy_in_storage

    delta_temp = model_parameters["heating_delta_temperature"]
    heating_coeff = model_parameters["heating_coefficient"]
    heat_loss_coeff = model_parameters["heat_loss_coefficient"]
    heat_capacity = model_parameters["heat_capacity"]

    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]

    energy_consumption = energy_drawn_from_grid + energy_pv_produced - energy_returned_to_grid - (
            charge_level_of_storage - prev_charge_level_of_storage) / 100 * storage_max_capacity
    energy_in_storage = (charge_level_of_storage - storage_min_charge_level) / 100 * storage_max_capacity
    initial_energy_in_storage = energy_in_storage

    total_power_used_for_heating = 0.
    for room_heating_params in room_heating_params_list:
        total_power_used_for_heating += sum(power * status for power, status
                                            in zip(room_heating_params["powers_of_heating_devices"],
                                                   heating_status_per_room[room_heating_params["name"]]))
    energy_consumption_without_heating = max(energy_consumption
                                             - total_power_used_for_heating * step_timedelta_s / 3600, 0.)

    (energy_for_heating, temp_per_room_configuration, next_temp_per_room) = check_heating_conditions(
        rooms_list=room_heating_params_list,
        heating_status_per_room=heating_status_per_room,
        temp_per_room=temp_per_room,
        temp_outdoor=temp_outdoor,
        delta_temp=delta_temp
    )
    (energy_pv_produced, energy_in_storage, energy_from_power_grid) = distribute_needed_energy(
        energy_needed=energy_consumption_without_heating + energy_for_heating,
        energy_pv_produced=energy_pv_produced,
        energy_in_storage=energy_in_storage
    )

    # Energy storage
    energy_in_storage += energy_pv_produced
    next_charge_level_of_storage = min(
        energy_in_storage / storage_max_capacity * 100.0 + storage_min_charge_level, 100.0
    )
    energy_storage_configuration = {"InWRte": 100.0, "OutWRte": 100.0}
    if energy_in_storage > initial_energy_in_storage:
        energy_storage_configuration["StorCtl_Mod"] = 1
    else:
        energy_storage_configuration["StorCtl_Mod"] = 2

    return (temp_per_room_configuration,
            energy_storage_configuration,
            next_temp_per_room,
            next_charge_level_of_storage,
            energy_from_power_grid)
