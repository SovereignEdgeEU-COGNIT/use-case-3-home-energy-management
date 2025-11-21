import datetime
import json
import logging
import random
import time

from cognit import device_runtime
import pandas as pd

from home_energy_management.baseline_algorithm import evaluate as evaluate_baseline
from home_energy_management.ppo_algorithm import train, evaluate

logging.basicConfig(level=logging.INFO)
sem_id = 123456

min_timestamp = datetime.datetime.fromisoformat('2023-06-01')
max_timestamp = datetime.datetime.fromisoformat('2025-02-01')

for _ in range(10):
    timestamp = min_timestamp + random.random() * (max_timestamp - min_timestamp)
    timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

    config_dir = '/home/agnieszka/repositories/cognit/use-case-3/scenario/'
    with open(config_dir + "config_local.json", "r") as f:
        config = json.load(f)

    num_simulation_days = datetime.timedelta(days=config["NUM_SIMULATION_DAYS"])
    besmart_access_parameters = config["BESMART_PARAMETERS"]
    s3_parameters = config["S3_PARAMETERS"]
    train_parameters = config["TRAIN_PARAMETERS"]
    eval_parameters = config["EVAL_PARAMETERS"]

    with open(config_dir + f"{sem_id}.json", "r") as f:
        sem_config = json.load(f)

    cycle_timedelta_s = sem_config["USER_APP_CYCLE_LENGTH"]
    initial_state = sem_config["INITIAL_STATE"]
    storage_config = sem_config["STORAGE_CONFIG"]
    ev_config = sem_config["EV_CONFIG"]
    heating_config = sem_config["HEATING_CONFIG"]
    home_model_parameters = sem_config["MODEL_PARAMETERS"]
    home_model_parameters.update(heating_config)
    user_preferences = sem_config["USER_PREFERENCES"]
    besmart_parameters = sem_config["BESMART_PARAMETERS"]

    s3_parameters["model_filename"] = s3_parameters["model_filename"].format(sem_id)
    besmart_parameters.update(besmart_access_parameters)
    heating_config["powers_of_heating_devices"] = heating_config["heating_devices_power"]
    storage_config["nominal_power"] = storage_config["max_power"]
    for ev_config_per_id in ev_config.values():
        ev_config_per_id["nominal_power"] = ev_config_per_id["max_power"]
    user_preferences["cycle_timedelta_s"] = cycle_timedelta_s

    REQS_INIT = {
        "FLAVOUR": "EnergyTorch",
        "MIN_ENERGY_RENEWABLE_USAGE": 75,
        "MAX_LATENCY": 600,
        "MAX_FUNCTION_EXECUTION_TIME": 15.0,
        "GEOLOCATION": {
            "latitude": 52.19,
            "longitude": 21.05
        }
    }

    runtime = device_runtime.DeviceRuntime(config_dir + "../cognit.yml")
    runtime.init(REQS_INIT)

    logging.info("COGNIT Serverless Runtime ready!")

    besmart_parameters_train = besmart_parameters.copy()
    besmart_parameters_train["till"] = (timestamp - datetime.timedelta(seconds=cycle_timedelta_s)).timestamp()
    besmart_parameters_train["since"] = (
            besmart_parameters_train["till"]
            - datetime.timedelta(days=train_parameters["history_timedelta_days"]).total_seconds()
    )

    besmart_parameters_evaluate = besmart_parameters.copy()
    besmart_parameters_evaluate["since"] = timestamp.timestamp()
    besmart_parameters_evaluate["till"] = (
            besmart_parameters_evaluate["since"]
            + datetime.timedelta(days=eval_parameters["history_timedelta_days"]).total_seconds()
    )


    logging.info(f"{sem_id = }")
    logging.info(f"{timestamp = }")

    mean_reward_list = []
    mean_energy_balance_list = []
    for i in range(10):
        logging.info(f"Iteration {i}")

        training_done = False
        while not training_done:
            logging.info(" --> COGNIT run training")
            start_time = time.perf_counter()
            train_result = runtime.call(
                train,
                json.dumps(train_parameters),
                json.dumps(s3_parameters),
                json.dumps(besmart_parameters_train),
                json.dumps(home_model_parameters),
                json.dumps(storage_config),
                json.dumps(ev_config),
                json.dumps(heating_config),
                json.dumps(user_preferences),
                timeout=600,
            )
            end_time = time.perf_counter()
            logging.info(f"Func result: {train_result.res}")
            logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")
            training_done = train_result.res

        evaluation_done = False
        while not evaluation_done:
            logging.info(" --> COGNIT run evaluation")
            start_time = time.perf_counter()
            eval_result = runtime.call(
                evaluate,
                json.dumps(eval_parameters),
                json.dumps(s3_parameters),
                json.dumps(besmart_parameters_evaluate),
                json.dumps(home_model_parameters),
                json.dumps(storage_config),
                json.dumps(ev_config),
                json.dumps(heating_config),
                json.dumps(user_preferences),
            )
            end_time = time.perf_counter()
            try:
                metrics = json.loads(eval_result.res)
                logging.info(f"Func result: {metrics}")
                logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

                mean_reward_list.append(metrics["mean_reward"])
                mean_energy_balance_list.append(metrics["mean_energy_balance"])
                evaluation_done = True
            except TypeError:
                evaluation_done = False

    metrics_summary_df = pd.DataFrame.from_dict({
        "sem_id": sem_id,
        "timestamp": timestamp,
        "mean_reward": mean_reward_list,
        "mean_energy_balance": mean_energy_balance_list,
    })
    metrics_summary_df.to_csv('evaluation_ppo.csv', index=False, mode='a', header=False)


    mean_reward_list = []
    mean_energy_balance_list = []
    for i in range(10):
        logging.info(f"Iteration {i}")

        evaluation_done = False
        while not evaluation_done:
            logging.info(" --> COGNIT run baseline evaluation")
            start_time = time.perf_counter()
            result = runtime.call(
                evaluate_baseline,
                json.dumps(eval_parameters),
                None,
                json.dumps(besmart_parameters_evaluate),
                json.dumps(home_model_parameters),
                json.dumps(storage_config),
                json.dumps(ev_config),
                json.dumps(heating_config),
                json.dumps(user_preferences),
            )
            end_time = time.perf_counter()

            try:
                metrics = json.loads(result.res)
                logging.info(f"Func result: {metrics}")
                logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")

                mean_reward_list.append(metrics["mean_reward"])
                mean_energy_balance_list.append(metrics["mean_energy_balance"])
                evaluation_done = True
            except TypeError:
                evaluation_done = False

    metrics_summary_df = pd.DataFrame.from_dict({
        "sem_id": sem_id,
        "timestamp": timestamp,
        "mean_reward": mean_reward_list,
        "mean_energy_balance": mean_energy_balance_list,
    })
    metrics_summary_df.to_csv('evaluation_baseline.csv', index=False, mode='a', header=False)
