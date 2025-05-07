import datetime
import logging
import time

import numpy as np
import pandas as pd
from cognit import device_runtime

from ppo_algorithm import make_decision, training_function

logging.basicConfig(level=logging.INFO)

REQS_INIT = {
    "FLAVOUR": "EnergyV2__Service_persistent",
    "MIN_ENERGY_RENEWABLE_USAGE": 50,
}

S3_PARAMETERS = {
    "endpoint_url": "https://s3.sovereignedge.eu/",
    "bucket_name": "uc3-test-bucket",
    "model_filename": "files/onnx_model_from_cognit.onnx",
}

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

home_model_parameters = {
    "min_temp_setting": 17.,  # (°C)
    "max_temp_setting": 24.,  # (°C)
    "heating_delta_temperature": 0.75,  # (°C)
    "heating_coefficient": 0.98,
    "heat_loss_coefficient": 300.,
    "heat_capacity": 3.6e7,
    "ev_driving_schedule": {
        "hour": [0., 8., 15., 20., 22.],
        "driving_power": [0., 5., 0., 8., 0.],
    },
    "pref_temp_schedule": {
        "hour": [0., 7., 9., 17., 23.],
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

pv_generation_df  = pd.read_csv(data_directory + 'pv_generation.csv', index_col=0)
pv_generation_df.index = pd.to_datetime(pv_generation_df.index)
pv_generation_pred_df  = pd.read_csv(data_directory + 'pv_generation_prediction.csv', index_col=0)
pv_generation_pred_df.index = pd.to_datetime(pv_generation_pred_df.index)
uncontrolled_consumption_df  = pd.read_csv(data_directory + 'uncontrolled_consumption.csv', index_col=0)
uncontrolled_consumption_df.index = pd.to_datetime(uncontrolled_consumption_df.index)
uncontrolled_consumption_pred_df  = pd.read_csv(data_directory + 'uncontrolled_consumption_prediction.csv', index_col=0)
uncontrolled_consumption_pred_df.index = pd.to_datetime(uncontrolled_consumption_pred_df.index)
temp_outside_df  = pd.read_csv(data_directory + 'temp_outside.csv', index_col=0)
temp_outside_df.index = pd.to_datetime(temp_outside_df.index)
temp_outside_df['value'] = temp_outside_df['value'].values - 272.15
temp_outside_pred_df  = pd.read_csv(data_directory + 'temp_outside_prediction.csv', index_col=0)
temp_outside_pred_df.index = pd.to_datetime(temp_outside_pred_df.index)
temp_outside_pred_df['value'] = temp_outside_pred_df['value'].values - 272.15

since_data = np.datetime64('2023-03-15')
till_train_data = np.datetime64('2023-06-15')
till_data = np.datetime64('2023-09-15')

pv_generation_train = pv_generation_df.loc[since_data: till_train_data, ['value']]
pv_generation_test = pv_generation_df.loc[till_train_data: till_data, ['value']]
pv_generation_pred_train = pv_generation_pred_df.loc[since_data: till_train_data, ['value']]
pv_generation_pred_test = pv_generation_pred_df.loc[till_train_data: till_data, ['value']]
uncontrolled_consumption_train = uncontrolled_consumption_df.loc[since_data: till_train_data, ['value']]
uncontrolled_consumption_test = uncontrolled_consumption_df.loc[till_train_data: till_data, ['value']]
uncontrolled_consumption_pred_train = uncontrolled_consumption_pred_df.loc[since_data: till_train_data, ['value']]
uncontrolled_consumption_pred_test = uncontrolled_consumption_pred_df.loc[till_train_data: till_data, ['value']]
temp_outside_train = temp_outside_df.loc[since_data: till_train_data, ['value']]
temp_outside_test = temp_outside_df.loc[till_train_data: till_data, ['value']]
temp_outside_pred_train = temp_outside_pred_df.loc[since_data: till_train_data, ['value']]
temp_outside_pred_test = temp_outside_pred_df.loc[till_train_data: till_data, ['value']]


logging.info(" --> Local run training")
start_time = time.perf_counter()
result = training_function(
    {
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
    },
    S3_PARAMETERS,
    home_model_parameters,
    storage_parameters,
    ev_battery_parameters,
    room_heating_params_list[0],
    3600,
    pv_generation_train.index.hour.values,
    pv_generation_train['value'].values,
    pv_generation_pred_train['value'].values,
    uncontrolled_consumption_train['value'].values,
    uncontrolled_consumption_pred_train['value'].values,
    temp_outside_train['value'].values,
    temp_outside_pred_train['value'].values,
)
end_time = time.perf_counter()
logging.info(f"Func result: {result = }")
logging.info(f"Execution time ({number_of_episodes} episodes): {(end_time - start_time):.6f} seconds")


logging.info(" --> COGNIT run training")
start_time = time.perf_counter()
return_code, result = runtime.call(
    training_function,
    {
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
    },
    S3_PARAMETERS,
    home_model_parameters,
    storage_parameters,
    ev_battery_parameters,
    room_heating_params_list[0],
    3600,
    pv_generation_train.index.hour.values,
    pv_generation_train['value'].values,
    pv_generation_pred_train['value'].values,
    uncontrolled_consumption_train['value'].values,
    uncontrolled_consumption_pred_train['value'].values,
    temp_outside_train['value'].values,
    temp_outside_pred_train['value'].values,
)
end_time = time.perf_counter()

logging.info(f"Status code: {return_code}")
logging.info(f"Func result: {result}")
logging.info(f"Execution time ({number_of_episodes} episodes): {(end_time - start_time):.6f} seconds")


timestamp = datetime.datetime.fromisoformat('2025-03-21 05:00:00')
pv_generation = 3.7
uncontrolled_consumption = 1.6
temp_outside = 15.
storage_parameters["curr_charge_level"] = 50.0  # (%)
ev_battery_parameters["curr_charge_level"] = 50.0  # (%)
ev_battery_parameters["is_available"] = True
ev_battery_parameters["time_until_charged"] = 3 * 3600  # (s)
room_heating_params_list[0]["curr_temp"] = 19.0  # (°C)


logging.info(" --> Local run predict 1")
start_time = time.perf_counter()
action = make_decision(
    timestamp=timestamp.timestamp(),
    s3_parameters=S3_PARAMETERS,
    home_model_parameters=home_model_parameters,
    storage_parameters=storage_parameters,
    ev_battery_parameters=ev_battery_parameters,
    room_heating_params_list=room_heating_params_list,
    pv_generation=pv_generation,
    uncontrolled_consumption=uncontrolled_consumption,
    temp_outside=temp_outside,
    cycle_timedelta_s=3600,
)
end_time = time.perf_counter()
logging.info(f"Func result 1: {action = }")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

logging.info(" --> Local run predict 2")
start_time = time.perf_counter()
action = make_decision(
    timestamp=timestamp.timestamp(),
    s3_parameters=S3_PARAMETERS,
    home_model_parameters=home_model_parameters,
    storage_parameters=storage_parameters,
    ev_battery_parameters=ev_battery_parameters,
    room_heating_params_list=room_heating_params_list,
    pv_generation=pv_generation,
    uncontrolled_consumption=uncontrolled_consumption,
    temp_outside=temp_outside,
    cycle_timedelta_s=3600,
)
end_time = time.perf_counter()
logging.info(f"Func result 2: {action = }")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")


logging.info(" --> COGNIT run predict 1")
start_time = time.perf_counter()
return_code, result = runtime.call(
    make_decision,
    timestamp.timestamp(),
    S3_PARAMETERS,
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

logging.info(f"Status code: {return_code}")
logging.info(f"Func result 1: {result}")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

logging.info(" --> COGNIT run predict 2")
start_time = time.perf_counter()
return_code, result = runtime.call(
    make_decision,
    timestamp.timestamp(),
    S3_PARAMETERS,
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

logging.info(f"Status code: {return_code}")
logging.info(f"Func result 2: {result}")
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")
