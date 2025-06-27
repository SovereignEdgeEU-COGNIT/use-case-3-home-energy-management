import datetime
import json
import logging
import time

from cognit import device_runtime

from ppo_algorithm import make_decision, training_function

logging.basicConfig(level=logging.INFO)

REQS_INIT = {
    "FLAVOUR": "EnergyV2__16GB_1CPU",
    "MIN_ENERGY_RENEWABLE_USAGE": 50,
}

S3_PARAMETERS = {
    "endpoint_url": "https://s3.sovereignedge.eu/",
    "bucket_name": "uc3-test-bucket",
    "model_filename": "files/onnx_model_from_cognit.onnx",
    "access_key_id": "eXuxY2Gt4bI8PTScQ9gz",
    "secret_access_key": "RtomdwwpoN7tkQe6ZZSPZTGScvQ0GtEwVhObreo4",
}

BESMART_PARAMETERS = {
    "workspace_key": "wubbalubbadubdub",
    "login": "cognit_demo",
    "password": "CognitDemo2025!",
    "pv_generation": {
        "cid": 68,
        "mid": 84,
        "moid": 70,
    },
    "energy_consumption": {
        "cid": 68,
        "mid": 83,
        "moid": 32,
    },
    "temperature_moid": 139,
    "since": datetime.datetime.fromisoformat('2023-03-15').timestamp(),
    "till": datetime.datetime.fromisoformat('2023-06-15').timestamp(),
}

cycle_timedelta_s = 3600

runtime = device_runtime.DeviceRuntime("cognit.yml")
runtime.init(REQS_INIT)

logging.info("COGNIT Serverless Runtime ready!")

ENERGY_REWARD_COEFFICIENT = .3
TEMP_REWARD_COEFFICIENT = 2.
STORAGE_REWARD_COEFFICIENT = .8
EV_REWARD_COEFFICIENT = .8

data_directory = 'data/'
model_filename = 'model_scripted_from_cognit.pt'

number_of_episodes = 8000
# Learning rate for actor-critic models
critic_lr = 0.001
actor_lr = 0.001

gamma = 0.2  # Discount factor for future rewards
lambda_ = 0.95
K_epochs = 10
eps_clip = 0.2
min_action_std = 0.1
action_std_decay_freq = 1 / 50
action_std_decay_rate = 0.01
update_epoch = 10

train_parameters = {
    "num_episodes": number_of_episodes,
    "critic_lr": critic_lr,
    "actor_lr": actor_lr,
    "gamma": gamma,  # Discount factor for future rewards
    "lambda_": lambda_,
    "num_epochs": K_epochs,
    "eps_clip": eps_clip,
    "min_action_std": min_action_std,
    "action_std_decay_freq": action_std_decay_freq,
    "action_std_decay_rate": action_std_decay_rate,
    "update_epoch": update_epoch,
    "action_std_init": 0.6,
    "batch_size": 64,
    "energy_reward_coeff": ENERGY_REWARD_COEFFICIENT,
    "temp_reward_coeff": TEMP_REWARD_COEFFICIENT,
    "storage_reward_coeff": STORAGE_REWARD_COEFFICIENT,
    "ev_reward_coeff": EV_REWARD_COEFFICIENT,
    "debug_mode": True,
}

home_model_parameters = {
    "min_temp_setting": 17.,  # (°C)
    "max_temp_setting": 24.,  # (°C)
    "heating_delta_temperature": 0.75,  # (°C)
    "heating_coefficient": 0.98,
    "heat_loss_coefficient": 300.,
    "heat_capacity": 3.6e7,
    "ev_driving_schedule": {
        "time": ["0:00", "8:00", "15:00", "20:00", "22:00"],
        "driving_power": [0., 5., 0., 8., 0.],
    },
    "pref_temp_schedule": {
        "time": ["0:00", "7:00", "9:00", "17:00", "23:00"],
        "temp": [18., 20, 18., 21., 19.],
    },
}
storage_parameters = {
    "nominal_power": 12.8,  # (kW)
    "max_capacity": 24.0,  # (kWh)
    "min_charge_level": 10.0,  # (%)
    "charging_switch_level": 75.0,  # (%)
    "efficiency": 0.85,
    "energy_loss": 0.
}
ev_battery_parameters = {
    "nominal_power": 6.9,  # (kW)
    "max_capacity": 40.0,  # (kWh)
    "min_charge_level": 10.0,  # (%)
    "driving_charge_level": 80.0,  # (%)
    "charging_switch_level": 75.0,  # (%)
    "efficiency": 0.85,
    "energy_loss": 0.
}
room_heating_params_list = [{
    "name": "room",
    "preferred_temp": 21.0,  # (°C)
    "powers_of_heating_devices": [8.0, 8.0],  # (kW)
}]


logging.info(" --> Local run training")
start_time = time.perf_counter()
result = training_function(
    json.dumps(train_parameters),
    json.dumps(S3_PARAMETERS),
    json.dumps(BESMART_PARAMETERS),
    json.dumps(home_model_parameters),
    json.dumps(storage_parameters),
    json.dumps(ev_battery_parameters),
    json.dumps(room_heating_params_list[0]),
    cycle_timedelta_s,
)
end_time = time.perf_counter()
logging.info(f"Func result: {result = }")
logging.info(f"Execution time ({number_of_episodes} episodes): {(end_time - start_time):.6f} seconds")


logging.info(" --> COGNIT run training")
start_time = time.perf_counter()
return_code, result = runtime.call(
    training_function,
    json.dumps(train_parameters),
    json.dumps(S3_PARAMETERS),
    json.dumps(BESMART_PARAMETERS),
    json.dumps(home_model_parameters),
    json.dumps(storage_parameters),
    json.dumps(ev_battery_parameters),
    json.dumps(room_heating_params_list[0]),
    cycle_timedelta_s,
)
end_time = time.perf_counter()

logging.info(f"Status code: {return_code}")
logging.info(f"Func result: {result}")
logging.info(f"Execution time ({number_of_episodes} episodes): {(end_time - start_time):.6f} seconds")


timestamp = datetime.datetime.fromisoformat('2023-06-16 05:00:00')
storage_parameters["curr_charge_level"] = 50.0  # (%)
ev_battery_parameters["curr_charge_level"] = 50.0  # (%)
ev_battery_parameters["is_available"] = True
ev_battery_parameters["time_until_charged"] = 3 * 3600  # (s)
room_heating_params_list[0]["curr_temp"] = 19.0  # (°C)


logging.info(" --> Local run predict 1")
start_time = time.perf_counter()
action = make_decision(
    timestamp=timestamp.timestamp(),
    s3_parameters=json.dumps(S3_PARAMETERS),
    besmart_parameters=json.dumps(BESMART_PARAMETERS),
    home_model_parameters=json.dumps(home_model_parameters),
    storage_parameters=json.dumps(storage_parameters),
    ev_battery_parameters=json.dumps(ev_battery_parameters),
    room_heating_params_list=json.dumps(room_heating_params_list),
    cycle_timedelta_s=cycle_timedelta_s,
)
end_time = time.perf_counter()
logging.info(f"Func result 1: {action = }")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

logging.info(" --> Local run predict 2")
start_time = time.perf_counter()
action = make_decision(
    timestamp=timestamp.timestamp(),
    s3_parameters=json.dumps(S3_PARAMETERS),
    besmart_parameters=json.dumps(BESMART_PARAMETERS),
    home_model_parameters=json.dumps(home_model_parameters),
    storage_parameters=json.dumps(storage_parameters),
    ev_battery_parameters=json.dumps(ev_battery_parameters),
    room_heating_params_list=json.dumps(room_heating_params_list),
    cycle_timedelta_s=cycle_timedelta_s,
)
end_time = time.perf_counter()
logging.info(f"Func result 2: {action = }")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")


logging.info(" --> COGNIT run predict 1")
start_time = time.perf_counter()
return_code, result = runtime.call(
    make_decision,
    timestamp.timestamp(),
    json.dumps(S3_PARAMETERS),
    json.dumps(BESMART_PARAMETERS),
    json.dumps(home_model_parameters),
    json.dumps(storage_parameters),
    json.dumps(ev_battery_parameters),
    json.dumps(room_heating_params_list),
    cycle_timedelta_s,
)
end_time = time.perf_counter()

logging.info(f"Status code: {return_code}")
logging.info(f"Func result 1: {result}")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

logging.info(" --> COGNIT run predict 2")
start_time = time.perf_counter()
return_code, result = runtime.call(
    make_decision,
    timestamp.timestamp(),
    json.dumps(S3_PARAMETERS),
    json.dumps(BESMART_PARAMETERS),
    json.dumps(home_model_parameters),
    json.dumps(storage_parameters),
    json.dumps(ev_battery_parameters),
    json.dumps(room_heating_params_list),
    cycle_timedelta_s,
)
end_time = time.perf_counter()

logging.info(f"Status code: {return_code}")
logging.info(f"Func result 2: {result}")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")
