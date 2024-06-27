import logging
import time

logging.basicConfig(level=logging.INFO)


temp_per_room = {
    "p1_01_pokoj_przy_schodach": 23.5,
    "p1_02_lazienka_przy_schodach": 21.8,
    "p1_03_duza_sypialnia": 23.5,
    "p1_04_lazienka_w_duzej_sypialni": 24.5,
    "p1_05_mala_sypialnia_1": 23.1,
    "p1_06_lazienka_malej_sypialni_1": 24.25,
    "p1_07_mala_sypialnia_2": 22.6,
    "p1_08_lazienka_malej_sypialni_2": 23.6,
    "p1_09_pokoj_nad_holem": 22.0,
    "p1_10_korytarz": 23.8,
    "p1_11_lazienka_nad_holem": 22.9,
    "p0_01_wiatrolap": 24.2,
    "p0_02_hol": 25.9,
    "p0_03_lazienka": 25.9,
    "p0_04_maly_pokoj": 26.0,
    "p0_05_gabinet": 25.9,
    "p0_06_salon": 24.3,
    "p0_07_ogrod_zimowy": 23.2,
    "p-1_01_hol": 22.8,
    "p-1_02_korytarz": 26.8,
    "p-1_03_pralnia": 25.7,
    "p-1_04_skladzik": 25.8,
    "p-1_06_fitness": 19.1,
    "p-1_07_lazienka": 22.3,
    "p-1_08_pokoj_rekreacyjny": 25.4,
    "p-1_09_kotłownia": 27.1,
}

data = {
    "max_capacity_of_storage": 12,
    "room_heating_params_list": [
        {
            "name": "p1_01_pokoj_przy_schodach",
            "powers_of_heating_devices": [2000],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {
            "name": "p1_02_lazienka_przy_schodach",
            "powers_of_heating_devices": [500],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {
            "name": "p1_03_duza_sypialnia",
            "powers_of_heating_devices": [2000, 1000],
            "is_heated": True,
            "temp_optimal": 19.25,
        },
        {
            "name": "p1_04_lazienka_w_duzej_sypialni",
            "powers_of_heating_devices": [2500],
            "is_heated": True,
            "temp_optimal": 21.5,
        },
        {
            "name": "p1_05_mala_sypialnia_1",
            "powers_of_heating_devices": [2000, 1000],
            "is_heated": True,
            "temp_optimal": 19.5,
        },
        {
            "name": "p1_06_lazienka_malej_sypialni_1",
            "powers_of_heating_devices": [800],
            "is_heated": True,
            "temp_optimal": 21.5,
        },
        {
            "name": "p1_07_mala_sypialnia_2",
            "powers_of_heating_devices": [1500],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {
            "name": "p1_08_lazienka_malej_sypialni_2",
            "powers_of_heating_devices": [500],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {
            "name": "p1_09_pokoj_nad_holem",
            "powers_of_heating_devices": [1500],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {"name": "p1_10_korytarz", "powers_of_heating_devices": [1000, 2000], "is_heated": False, "temp_optimal": 19.0},
        {
            "name": "p1_11_lazienka_nad_holem",
            "powers_of_heating_devices": [500],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {"name": "p0_01_wiatrolap", "powers_of_heating_devices": [800], "is_heated": True, "temp_optimal": 18.7},
        {"name": "p0_02_hol", "powers_of_heating_devices": [2900], "is_heated": True, "temp_optimal": 21.0},
        {"name": "p0_03_lazienka", "powers_of_heating_devices": [440], "is_heated": True, "temp_optimal": 20.5},
        {"name": "p0_04_maly_pokoj", "powers_of_heating_devices": [1800], "is_heated": True, "temp_optimal": 21.0},
        {"name": "p0_05_gabinet", "powers_of_heating_devices": [2800], "is_heated": False, "temp_optimal": 19.0},
        {
            "name": "p0_06_salon",
            "powers_of_heating_devices": [2800, 2000, 2000, 2000],
            "is_heated": True,
            "temp_optimal": 20.5,
        },
        {
            "name": "p0_07_ogrod_zimowy",
            "powers_of_heating_devices": [2500, 2500],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {"name": "p-1_01_hol", "powers_of_heating_devices": [2800], "is_heated": False, "temp_optimal": 19.0},
        {"name": "p-1_02_korytarz", "powers_of_heating_devices": [1500], "is_heated": False, "temp_optimal": 19.0},
        {"name": "p-1_03_pralnia", "powers_of_heating_devices": [1000], "is_heated": False, "temp_optimal": 19.0},
        {"name": "p-1_04_skladzik", "powers_of_heating_devices": [1000], "is_heated": False, "temp_optimal": 19.0},
        {"name": "p-1_06_fitness", "powers_of_heating_devices": [2800, 2800], "is_heated": False, "temp_optimal": 19.0},
        {"name": "p-1_07_lazienka", "powers_of_heating_devices": [740], "is_heated": True, "temp_optimal": 21.5},
        {
            "name": "p-1_08_pokoj_rekreacyjny",
            "powers_of_heating_devices": [2700],
            "is_heated": False,
            "temp_optimal": 19.0,
        },
        {"name": "p-1_09_kotłownia", "powers_of_heating_devices": [1000], "is_heated": False, "temp_optimal": 19.0},
    ],
    "min_charge_level_of_storage": 45,
    "high_charge_level_of_storage": 90,
    "cycle_timedelta_s": 300,
    "delta_temp": 0.75,
    "heating_coeff": 2.5e-05,
    "heat_loss_coeff": 2.5e-05,
}

# Sanity check that algo runs locally
(
    configuration_of_temp_per_room,
    if_to_charge_storage,
    next_step_temp_per_room,
    next_step_charge_level_of_storage,
    predicted_energy_from_power_grid,
) = run_one_step(data, 7, 2, 10, 68, temp_per_room)

logging.info(f"{configuration_of_temp_per_room = }")
logging.info(f"{if_to_charge_storage = }")
logging.info(f"{next_step_temp_per_room = }")
logging.info(f"{next_step_charge_level_of_storage = }")
logging.info(f"{predicted_energy_from_power_grid = }")


sr_conf = ServerlessRuntimeConfig()
sr_conf.name = "Smart Energy Meter Serverless Runtime"
sr_conf.scheduling_policies = [EnergySchedulingPolicy(50)]

try:
    runtime = ServerlessRuntimeContext(config_path="./cognit.yml")
    runtime.create(sr_conf)
except Exception as e:
    print("Error in config file content: {}".format(e))
    exit(1)

while runtime.status != FaaSState.RUNNING:
    time.sleep(1)

print("COGNIT Serverless Runtime ready!")

time.sleep(45)


offloadCtx = runtime.call_async(run_one_step, data, 7, 2, 10, 68, temp_per_room)
time.sleep(2)

print("Status: ", offloadCtx)

status = runtime.wait(offloadCtx.exec_id, 20)

if status.res != None:
    print(status.res.res)
else:
    print(status)

runtime.delete()
