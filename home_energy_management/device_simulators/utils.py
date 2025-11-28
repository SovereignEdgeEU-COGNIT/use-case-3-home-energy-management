from typing import Any

import numpy as np
import requests
from home_energy_management.device_simulators.heating import ScheduledTempSensor
from home_energy_management.device_simulators.photovoltaic import ScheduledPV
from home_energy_management.device_simulators.simple_device import ScheduledDataDevice, SimpleScheduledDevice


def get_data_from_besmart(
        besmart_parameters: dict[str, Any],
        token: str,
        signal_type: str,
        is_cumulative: bool = False
) -> dict:
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    since = int(besmart_parameters["since"])
    if is_cumulative:
        since += -3600
    identifier = besmart_parameters[signal_type]
    body = [{
        "client_cid": identifier["cid"],
        "sensor_mid": identifier["mid"],
        "signal_type_moid": identifier["moid"],
        "since": since * 1000,
        "till": int(besmart_parameters["till"]) * 1000,
        "get_last": True,
    }]
    res = requests.post(
        'https://api.besmart.energy/api/sensors/signals/data',
        headers=headers, json=body
    )
    return res.json()[0]['data']


def get_energy_data(
        besmart_parameters: dict[str, Any],
        token: str,
        signal_type: str,
        is_cumulative: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    data = get_data_from_besmart(besmart_parameters, token, signal_type, is_cumulative)
    time = (np.array(data['time']) / 1e3).astype(int) - int(besmart_parameters["since"])
    value = np.array(data['value'])
    origin = np.array(data['origin'])

    real_value = value[origin == 1] * 1e3
    real_time = time[origin == 1]
    if is_cumulative:
        real_value = np.diff(real_value) / (np.diff(real_time) / 3600)
        real_time = real_time[1:]

    return real_time, real_value


def get_temperature_data(
        besmart_parameters: dict[str, Any],
        token: str
) -> tuple[np.ndarray, np.ndarray]:
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    sensor_identifier = besmart_parameters["energy_consumption"]
    sensor = requests.get(
        f'https://api.besmart.energy/api/sensors/{sensor_identifier["cid"]}.{sensor_identifier["mid"]}',
        headers=headers,
    ).json()

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Auth': besmart_parameters['workspace_key'],
    }
    params = {
        "since": int(besmart_parameters["since"]) * 1000,
        "till": int(besmart_parameters["till"]) * 1000,
        'delta_t': 60,
        'raw': False,
        'get_last': True,
    }
    res = requests.get(
        f'https://api.besmart.energy/api/weather/{sensor["lat"]}/{sensor["lon"]}/{besmart_parameters["temperature_moid"]}/data',
        headers=headers, params=params
    )
    data = res.json()['data']

    time = (np.array(data['time']) / 1e3).astype(int) - int(besmart_parameters["since"])
    value = np.array(data['value'])
    origin = np.array(data['origin'])
    estm_value = value[origin == 3] - 272.15
    estm_time = time[origin == 3]

    return estm_time, estm_value


def prepare_device_simulator_from_data(
        besmart_parameters,
        signal_type: str,
) -> ScheduledDataDevice:
    token = besmart_parameters["token"]
    if signal_type == "energy_consumption":
        time, data = get_energy_data(besmart_parameters, token, signal_type, True)
        return SimpleScheduledDevice(time.tolist(), data.tolist())
    if signal_type == "pv_generation":
        time, data = get_energy_data(besmart_parameters, token, signal_type, False)
        return ScheduledPV(time.tolist(), data.tolist())
    if signal_type == "temperature":
        time, data = get_temperature_data(besmart_parameters, token)
        return ScheduledTempSensor(time.tolist(), data.tolist())
    return None
