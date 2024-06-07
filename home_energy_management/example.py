import logging

from home_energy_management.decision_algo import run_one_step

logging.basicConfig(level=logging.INFO)

DELTA_TEMP = 0.75
HEATING_COEFF = 0.98
HEAT_LOSS_COEFF = 300
HEAT_CAPACITY = 3.6e7

model_parameters = {
    "heating_delta_temperature": DELTA_TEMP,
    "heating_coefficient": HEATING_COEFF,
    "heat_loss_coefficient": HEAT_LOSS_COEFF,
    "heat_capacity": HEAT_CAPACITY,
    "storage_high_charge_level": 90.,
    "storage_delta_power_perc": 5.,
}

storage_parameters = {
    "max_capacity": 24.,
    "min_charge_level": 40.,
    "efficiency": 0.98,
    "nominal_power": 6.4
}

room_heating_params_list = [{
    "name": "room",
    "powers_of_heating_devices": [8.0, 8.0],
    "preferred_temp": 20.0,
}]

energy_pv_produced_list = [0., 0., 0., 0., 0., 0., 0., 0.3, 0.7, 1.6, 9.2, 11.5,
                           12., 10., 8., 5., 1.8, 0.5, 0., 0., 0., 0., 0., 0.]
energy_drawn_list = [1.4, 1.4, 0.9, 0.9, 0.9, 1.4, 1.8, 1.8, 1.6, 0.9, 1.2, 0.9,
                     0.2, 0.9, 1.4, 1.6, 1.8, 2.1, 2.1, 1.8, 1.8, 1.6, 1.4, 1.4]
energy_returned_list = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.,
                        1., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
temp_outdoor_list = [10, 10, 10, 6, 6, 6, 8, 8, 8, 10, 10, 13,
                     13, 16, 16, 16, 18, 18, 16, 16, 13, 13, 10, 10]

temp_per_room = {
    "room": 19.
}
heating_status_per_room = {
    "room": [False, False]
}
charge_level_of_storage = 50.
prev_charge_level_of_storage = 50.

i = 0
for (energy_drawn_from_grid,
     energy_returned_to_grid,
     energy_pv_produced,
     temp_outdoor,
     ) in zip(
    energy_drawn_list,
    energy_returned_list,
    energy_pv_produced_list,
    temp_outdoor_list,
):
    i += 1
    logging.info(f'------- step = {i} -------')

    (configuration_of_temp_per_room,
     configuration_of_energy_storage,
     next_step_temp_per_room,
     next_step_charge_level_of_storage,
     predicted_energy_from_power_grid,) = run_one_step(model_parameters=model_parameters,
                                                       step_timedelta_s=3600,
                                                       storage_parameters=storage_parameters,
                                                       room_heating_params_list=room_heating_params_list,
                                                       energy_drawn_from_grid=energy_drawn_from_grid,
                                                       energy_returned_to_grid=energy_returned_to_grid,
                                                       energy_pv_produced=energy_pv_produced,
                                                       temp_outdoor=temp_outdoor,
                                                       charge_level_of_storage=charge_level_of_storage,
                                                       prev_charge_level_of_storage=prev_charge_level_of_storage,
                                                       heating_status_per_room=heating_status_per_room,
                                                       temp_per_room=temp_per_room)

    logging.info(f'{configuration_of_temp_per_room = }')
    logging.info(f'{configuration_of_energy_storage = }')
    logging.info(f'{next_step_temp_per_room = }')
    logging.info(f'{next_step_charge_level_of_storage = }')
    logging.info(f'{predicted_energy_from_power_grid = }')

    for room_heating_params in room_heating_params_list:
        name = room_heating_params["name"]
        temp = temp_per_room[name]
        preferred_temp = room_heating_params["preferred_temp"]
        was_switch_on = heating_status_per_room[name][0]
        is_switch_on = was_switch_on if abs(preferred_temp - temp) < 0.75 else (preferred_temp > temp)
        heating_status_per_room[name] = len(room_heating_params["powers_of_heating_devices"]) * [is_switch_on]

    temp_per_room = next_step_temp_per_room
    prev_charge_level_of_storage = charge_level_of_storage
    charge_level_of_storage = next_step_charge_level_of_storage
