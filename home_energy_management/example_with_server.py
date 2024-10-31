import logging
import time

from cognit import (
    EnergySchedulingPolicy,
    FaaSState,
    ServerlessRuntimeConfig,
    ServerlessRuntimeContext,
)

from decision_algo import run_one_step

logging.basicConfig(level=logging.INFO)

model_parameters = {
    "heating_delta_temperature": 0.75,
    "heating_coefficient": 0.98,
    "heat_loss_coefficient": 300.,
    "heat_capacity": 3.6e7,
    "delta_charging_power_perc": 5.0,
    "storage_high_charge_level": 90.0,
}

storage_parameters = {
    "nominal_power": 12.8,  # (kW)
    "max_capacity": 24.0,  # (kWh)
    "min_charge_level": 20.0,  # (%)
    "efficiency": 0.98,
}

ev_parameters = {
    "nominal_power": 6.9,  # (kW)
    "max_capacity": 40.0,  # (kWh)
    "charged_level": 80.0,  # (%)
    "efficiency": 0.98,
    "is_available": True,
    "time_until_charged": 3 * 3600,
}

room_heating_params_list = [{
    "name": "room",
    "preferred_temp": 21.0,  # (°C)
    "powers_of_heating_devices": [8.0, 8.0]
}]

heating_status_per_room = {
    "room": [False],
}

temp_per_room = {
    "room": 19.0
}

# Sanity check that algo runs locally
(configuration_of_temp_per_room,
 configuration_of_energy_storage,
 configuration_of_ev_battery,
 next_step_temp_per_room,
 next_step_charge_level_of_storage,
 next_step_charge_level_of_ev_battery,
 predicted_energy_from_power_grid,) = run_one_step(model_parameters=model_parameters,
                                                   step_timedelta_s=3600,
                                                   storage_parameters=storage_parameters,
                                                   ev_battery_parameters=ev_parameters,
                                                   room_heating_params_list=room_heating_params_list,
                                                   energy_drawn_from_grid=0.,
                                                   energy_returned_to_grid=2.1,
                                                   energy_pv_produced=3.7,
                                                   temp_outdoor=15.,
                                                   charge_level_of_storage=50.,
                                                   prev_charge_level_of_storage=50.,
                                                   charge_level_of_ev_battery=50.,
                                                   prev_charge_level_of_ev_battery=50.,
                                                   heating_status_per_room=heating_status_per_room,
                                                   temp_per_room=temp_per_room)

logging.info(f'{configuration_of_temp_per_room = }')
logging.info(f'{configuration_of_energy_storage = }')
logging.info(f'{configuration_of_ev_battery = }')
logging.info(f'{next_step_temp_per_room = }')
logging.info(f'{next_step_charge_level_of_storage = }')
logging.info(f'{next_step_charge_level_of_ev_battery = }')
logging.info(f'{predicted_energy_from_power_grid = }')


sr_conf = ServerlessRuntimeConfig()
sr_conf.name = "Smart Energy Meter Serverless Runtime"
sr_conf.scheduling_policies = [EnergySchedulingPolicy(50)]
sr_conf.faas_flavour = "Energy"

try:
    runtime = ServerlessRuntimeContext(config_path="./cognit.yml")
    runtime.create(sr_conf)
except Exception as e:
    print("Error in config file content: {}".format(e))
    exit(1)

while runtime.status != FaaSState.RUNNING:
    time.sleep(1)

print("COGNIT Serverless Runtime ready!")

time.sleep(15)


offloadCtx = runtime.call_async(
    run_one_step,
    model_parameters,
    3600,
    storage_parameters,
    ev_parameters,
    room_heating_params_list,
    0.,
    2.1,
    3.7,
    15.,
    50.,
    50.,
    50.,
    50.,
    heating_status_per_room,
    temp_per_room
)
time.sleep(2)

print("Status: ", offloadCtx)

status = runtime.wait(offloadCtx.exec_id, 20)

if status.res != None:
    print(status.res.res)
else:
    print(status)

runtime.delete()
