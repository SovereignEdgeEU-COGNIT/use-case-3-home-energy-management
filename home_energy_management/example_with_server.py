import datetime
import json
import logging
import time

from cognit import device_runtime

from access_config import BESMART_PARAMETERS
from baseline_algorithm import make_decision, evaluate

logging.basicConfig(level=logging.INFO)

REQS_INIT = {
    "FLAVOUR": "Energy",
    "GEOLOCATION": {
        "latitude": 59.3294,
        "longitude": 18.0687,
    },
}

timestamp = datetime.datetime.fromisoformat('2023-06-16 05:00:00')

ENERGY_REWARD_COEFFICIENT = .3
TEMP_REWARD_COEFFICIENT = 3.
STORAGE_REWARD_COEFFICIENT = .8
EV_REWARD_COEFFICIENT = .8

eval_parameters = {
    "history_timedelta_days": 90,
    "energy_reward_coeff": ENERGY_REWARD_COEFFICIENT,
    "temp_reward_coeff": TEMP_REWARD_COEFFICIENT,
    "storage_reward_coeff": STORAGE_REWARD_COEFFICIENT,
    "ev_reward_coeff": EV_REWARD_COEFFICIENT,
}

model_parameters = {
    "temp_window": 0.75,
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
    "charging_switch_level": 75.0,  # (%)
    "efficiency": 0.98,
    "curr_charge_level": 50.0,
    "energy_loss": 0.
}

ev_battery_parameters = {
    0: {
        "nominal_power": 6.9,  # (kW)
        "max_capacity": 40.0,  # (kWh)
        "min_charge_level": 10.0,  # (%)
        "driving_charge_level": 80.0,  # (%)
        "charging_switch_level": 75.0,  # (%)
        "efficiency": 0.85,
        "energy_loss": 0.,
        "is_available": True,
        "time_until_charged": 3 * 3600,
        "curr_charge_level": 50.0,
    },
    1: {
        "nominal_power": 5.5,  # (kW)
        "max_capacity": 32.0,  # (kWh)
        "min_charge_level": 10.0,  # (%)
        "driving_charge_level": 90.0,  # (%)
        "charging_switch_level": 75.0,  # (%)
        "efficiency": 0.85,
        "energy_loss": 0.,
        "is_available": True,
        "time_until_charged": 2 * 3600,
        "curr_charge_level": 60.0,
    },
}

heating_parameters = {
    "curr_temp": 19.0,  # (°C)
    "preferred_temp": 21.0,  # (°C)
    "powers_of_heating_devices": [8.0, 8.0],
    "is_device_switch_on": [False, False],
}

user_preferences = {
    "ev_driving_schedule": {
        0: {
            "time": ["0:00", "8:00", "15:00", "20:00", "22:00"],
            "driving_power": [0., 5., 0., 8., 0.],
        },
        1: {
            "time": ["0:00", "7:00", "9:00", "14:00", "16:00", "18:00", "19:00"],
            "driving_power": [0., 4., 0., 4., 0., 6., 0.],
        }
    },
    "pref_temp_schedule": {
        "time": ["0:00", "7:00", "9:00", "17:00", "23:00"],
        "temp": [18., 20, 18., 21., 19.],
    },
    "cycle_timedelta_s": 3600,
}

BESMART_PARAMETERS["till"] = (timestamp - datetime.timedelta(seconds=user_preferences["cycle_timedelta_s"])).timestamp()
BESMART_PARAMETERS["since"] = (BESMART_PARAMETERS["till"]
                               - datetime.timedelta(days=eval_parameters["history_timedelta_days"]).total_seconds())

# Sanity check that algo runs locally
result = evaluate(
    eval_parameters=json.dumps(eval_parameters),
    s3_parameters=None,
    besmart_parameters=json.dumps(BESMART_PARAMETERS),
    home_model_parameters=json.dumps(model_parameters),
    storage_parameters=json.dumps(storage_parameters),
    ev_battery_parameters_per_id=json.dumps(ev_battery_parameters),
    heating_parameters=json.dumps(heating_parameters),
    user_preferences=json.dumps(user_preferences),
)
logging.info("Func result: " + str(result))

result = make_decision(
    timestamp=timestamp.timestamp(),
    s3_parameters=None,
    besmart_parameters=json.dumps(BESMART_PARAMETERS),
    home_model_parameters=json.dumps(model_parameters),
    storage_parameters=json.dumps(storage_parameters),
    ev_battery_parameters_per_id=json.dumps(ev_battery_parameters),
    heating_parameters=json.dumps(heating_parameters),
    user_preferences=json.dumps(user_preferences),
)

logging.info("Func result: " + str(result))


runtime = device_runtime.DeviceRuntime("cognit.yml")
runtime.init(REQS_INIT)

logging.info("COGNIT Serverless Runtime ready!")

logging.info(" --> COGNIT run baseline evaluation")
start_time = time.perf_counter()
result = runtime.call(
    evaluate,
    json.dumps(eval_parameters),
    None,
    json.dumps(BESMART_PARAMETERS),
    json.dumps(model_parameters),
    json.dumps(storage_parameters),
    json.dumps(ev_battery_parameters),
    json.dumps(heating_parameters),
    json.dumps(user_preferences),
)
end_time = time.perf_counter()

logging.info("Func result: " + str(result))
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

logging.info(" --> COGNIT run baseline decision")
start_time = time.perf_counter()
result = runtime.call(
    make_decision,
    timestamp.timestamp(),
    None,
    json.dumps(BESMART_PARAMETERS),
    json.dumps(model_parameters),
    json.dumps(storage_parameters),
    json.dumps(ev_battery_parameters),
    json.dumps(heating_parameters),
    json.dumps(user_preferences),
)
end_time = time.perf_counter()

logging.info("Func result: " + str(result))
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")
