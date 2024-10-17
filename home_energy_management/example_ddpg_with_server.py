import logging
import time

from cognit import device_runtime

from ddpg_algorithm import get_actor, make_decision

logging.basicConfig(level=logging.INFO)

REQS_INIT = {
      "FLAVOUR": "EnergyV2__7GB",
      "MIN_ENERGY_RENEWABLE_USAGE": 50,
}


num_states = 7
num_actions = 3

lower_bounds = [7., -12.8, 0.]
upper_bounds = [30., 12.8, 6.9]

home_model_parameters = {
    "min_temp_setting": 7.,
    "max_temp_setting": 30.,
}
storage_parameters = {
    "nominal_power": 12.8,  # (kW)
    "curr_charge_level": 50.0,
}
ev_battery_parameters = {
    "nominal_power": 6.9,  # (kW)
    "time_until_charged": 3 * 3600,  # s
    "curr_charge_level": 50.0,
}
room_heating_params_list = [{
    "name": "room",
    "curr_temp": 19.0,  # (°C)
}]

pv_generation = 3.7
uncontrolled_consumption = 1.6
temp_outside = 15.

actor_model = get_actor(num_states, lower_bounds, upper_bounds)
actor_model.load_weights("hems_actor.weights.h5")

model_weights = []
for layer_weights in actor_model.get_weights():
    model_weights.append(layer_weights.tolist())

action = make_decision(
    trained_model=actor_model,
    home_model_parameters=home_model_parameters,
    storage_parameters=storage_parameters,
    ev_battery_parameters=ev_battery_parameters,
    room_heating_params_list=room_heating_params_list,
    pv_generation=pv_generation,
    uncontrolled_consumption=uncontrolled_consumption,
    temp_outside=temp_outside,
    cycle_timedelta_s=3600,
)

logging.info(f'{action = }')

runtime = device_runtime.DeviceRuntime("cognit.yml")
runtime.init(REQS_INIT)

logging.info("COGNIT Serverless Runtime ready!")

start_time = time.perf_counter()
return_code, result = runtime.call(
    make_decision,
    actor_model,
    home_model_parameters,
    storage_parameters,
    ev_battery_parameters,
    room_heating_params_list,
    pv_generation,
    uncontrolled_consumption,
    temp_outside,
    3600,
)
end_time = time.perf_counter()

logging.info("Status code: " + str(return_code))
logging.info("Func result: " + str(result))
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")
