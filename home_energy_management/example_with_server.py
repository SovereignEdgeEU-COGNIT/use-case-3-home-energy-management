import datetime
import logging
import time

from cognit import device_runtime

from baseline_algorithm import make_decision

logging.basicConfig(level=logging.INFO)

REQS_INIT = {
    "FLAVOUR": "EnergyV2",
    "MIN_ENERGY_RENEWABLE_USAGE": 50,
}

timestamp = datetime.datetime.fromisoformat('2025-03-21 05:00:00')

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
    "since": datetime.datetime.fromisoformat('2023-03-15'),
    "till": datetime.datetime.fromisoformat('2023-06-15'),
}

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
    "curr_charge_level": 50.0,
}

ev_parameters = {
    "nominal_power": 6.9,  # (kW)
    "max_capacity": 40.0,  # (kWh)
    "driving_charge_level": 80.0,  # (%)
    "efficiency": 0.98,
    "is_available": True,
    "time_until_charged": 3 * 3600,
    "curr_charge_level": 50.0,
}

room_heating_params_list = [{
    "name": "room",
    "curr_temp": 19.0,  # (°C)
    "preferred_temp": 21.0,  # (°C)
    "powers_of_heating_devices": [8.0, 8.0],
    "is_device_switch_on": [False, False],
}]


# Sanity check that algo runs locally
(configuration_of_temp_per_room,
 configuration_of_energy_storage,
 configuration_of_ev_battery,) = make_decision(timestamp=timestamp.timestamp(),
                                               s3_parameters=None,
                                               besmart_parameters=BESMART_PARAMETERS,
                                               home_model_parameters=model_parameters,
                                               storage_parameters=storage_parameters,
                                               ev_battery_parameters=ev_parameters,
                                               room_heating_params_list=room_heating_params_list,
                                               cycle_timedelta_s=3600,)

logging.info(f'{configuration_of_temp_per_room = }')
logging.info(f'{configuration_of_energy_storage = }')
logging.info(f'{configuration_of_ev_battery = }')


runtime = device_runtime.DeviceRuntime("cognit.yml")
runtime.init(REQS_INIT)

logging.info("COGNIT Serverless Runtime ready!")

start_time = time.perf_counter()
return_code, result = runtime.call(
    make_decision,
    timestamp.timestamp(),
    None,
    model_parameters,
    storage_parameters,
    ev_parameters,
    room_heating_params_list,
    3.7,
    1.6,
    15.,
    3600,
)
end_time = time.perf_counter()

logging.info("Status code: " + str(return_code))
logging.info("Func result: " + str(result))
logging.info(f"Execution time: {(end_time - start_time):.6f} seconds")
