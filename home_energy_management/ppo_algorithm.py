def make_decision(
        timestamp: float,
        s3_parameters: str,
        besmart_parameters: str,
        home_model_parameters: str,
        storage_parameters: str,
        ev_battery_parameters_per_id: str,
        heating_parameters: str,
        user_preferences: str,
) -> tuple[float, str, str, str]:
    import datetime
    import json
    from io import BytesIO

    import botocore
    import boto3
    import numpy as np
    import onnx
    import torch
    import requests
    from onnx2torch import convert

    def select_action(
            actor_model: torch.fx.graph_module.GraphModule,
            tensor_state: torch.Tensor,
            lower_bounds: list[float],
            upper_bounds: list[float]
    ) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.FloatTensor(tensor_state).to(device)
            tensor_action = actor_model(tensor_state)

        tensor_action = tensor_action.detach().cpu().numpy().flatten()
        tensor_action[0] = tensor_action[0] * (upper_bounds[0] - lower_bounds[0]) / 2 + (
                upper_bounds[0] + lower_bounds[0]) / 2
        tensor_action[1] = tensor_action[1] * upper_bounds[1]
        for i in range(len(ev_id_list)):
            tensor_action[2 + i] = (tensor_action[2 + i] + 1) * upper_bounds[2 + i] / 2
        tensor_action = np.clip(tensor_action, np.array(lower_bounds), np.array(upper_bounds))

        return tensor_action

    def get_data_from_besmart(
            cid: int,
            mid: int,
            moid: int,
            is_cumulative: bool = False
    ) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        since_datetime = np.datetime64(int(state_datetime.timestamp()), 's')
        if is_cumulative:
            since_datetime -= np.timedelta64(3600, 's')
        till_datetime = np.datetime64(int(state_datetime.timestamp()), 's') + np.timedelta64(cycle_timedelta_s, 's')
        body = [{
            "client_cid": cid,
            "sensor_mid": mid,
            "signal_type_moid": moid,
            "since": int(since_datetime.astype(int) * 1000),
            "till": int(till_datetime.astype(int) * 1000),
            "get_last": True,
        }]
        res = requests.post(
            'https://api.besmart.energy/api/sensors/signals/data',
            headers=headers, json=body
        )
        if res.status_code == 200:
            return res.json()[0]['data']
        return res.status_code

    def get_energy_data(
            identifier: dict[str, int],
            is_cumulative: bool = False
    ) -> float:
        data = get_data_from_besmart(identifier["cid"],
                                     identifier["mid"],
                                     identifier["moid"],
                                     is_cumulative)
        try:
            value = np.array(data['value'])
            origin = np.array(data['origin'])
        except Exception as e:
            raise Exception(f'{e} - besmart returned HTTP {data}')

        pred_value = value[origin == 2]
        if is_cumulative:
            pred_time = (np.array(data['time']) / 1e3).astype(int)[origin == 2]
            state_index = np.where(pred_time >= state_datetime.timestamp())[0][0]
            pred_value = pred_value[state_index - 1:state_index + 1]
            pred_time = pred_time[state_index - 1:state_index + 1]
            pred_value = np.diff(pred_value) / (np.diff(pred_time) / 3600)

        if len(pred_value) < 1:
            raise Exception(
                'Not enough energy data for decision-making '
                f'(cid: {identifier["cid"]}, mid: {identifier["mid"]}, moid: {identifier["moid"]})'
            )

        return pred_value[0]

    def get_temperature_data() -> float:
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
        till_datetime = np.datetime64(state_datetime) + np.timedelta64(cycle_timedelta_s, 's')
        params = {
            "since": int(np.datetime64(state_datetime).astype(int) / 1000),
            "till": int(till_datetime.astype(int) / 1000),
            'delta_t': cycle_timedelta_s // 60,
            'raw': False,
            'get_last': True,
        }
        res = requests.get(
            f'https://api.besmart.energy/api/weather/{sensor["lat"]}/{sensor["lon"]}/{besmart_parameters["temperature_moid"]}/data',
            headers=headers, params=params
        )
        if res.status_code == 200:
            data = res.json()['data']
        else:
            raise Exception(f'Besmart returned HTTP {res.status_code}')

        value = np.array(data['value'])
        origin = np.array(data['origin'])
        estm_value = value[origin == 3]
        if len(estm_value) < 1:
            raise Exception(
                f'Not enough temperature data for decision-making (lat: {sensor["lat"]}, lon: {sensor["lon"]})'
            )

        return estm_value[0] - 272.15


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s3_parameters = json.loads(s3_parameters)
    besmart_parameters = json.loads(besmart_parameters)
    home_model_parameters = json.loads(home_model_parameters)
    storage_parameters = json.loads(storage_parameters)
    ev_battery_parameters_per_id = json.loads(ev_battery_parameters_per_id) if (
            ev_battery_parameters_per_id != json.dumps(None)) else {}
    heating_parameters = json.loads(heating_parameters)
    user_preferences = json.loads(user_preferences)

    state_datetime = datetime.datetime.fromtimestamp(timestamp)
    cycle_timedelta_s = user_preferences["cycle_timedelta_s"]
    cycle_timedelta_min = cycle_timedelta_s // 60
    rounding_minutes = state_datetime.minute % cycle_timedelta_min
    if rounding_minutes > cycle_timedelta_min / 2:
        rounding_minutes = - (cycle_timedelta_min - rounding_minutes)
    state_datetime = datetime.datetime(year=state_datetime.year,
                                       month=state_datetime.month,
                                       day=state_datetime.day,
                                       hour=state_datetime.hour,
                                       minute=state_datetime.minute)
    state_datetime = state_datetime - datetime.timedelta(minutes=rounding_minutes)

    min_temp_setting = home_model_parameters["min_temp_setting"]
    max_temp_setting = home_model_parameters["max_temp_setting"]
    storage_max_charging_power = storage_parameters["nominal_power"]
    storage_soc = storage_parameters["curr_charge_level"]
    ev_id_list = list(ev_battery_parameters_per_id.keys())
    ev_id_list.sort()
    temp_inside = heating_parameters["curr_temp"]
    pref_temp = heating_parameters["preferred_temp"]
    temp_window = home_model_parameters["temp_window"]

    lower_bounds = [min_temp_setting, - storage_max_charging_power] + len(ev_id_list) * [0.]
    upper_bounds = [max_temp_setting, storage_max_charging_power] + [
        ev_battery_parameters_per_id[ev_id]["nominal_power"] for ev_id in ev_id_list]

    token = besmart_parameters["token"]
    pv_generation_pred = get_energy_data(besmart_parameters["pv_generation"])
    energy_consumption_pred = get_energy_data(besmart_parameters["energy_consumption"], True)
    temp_outside_pred = get_temperature_data()

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_parameters["endpoint_url"],
            aws_access_key_id=s3_parameters["access_key_id"],
            aws_secret_access_key=s3_parameters["secret_access_key"],
        )
        stream = BytesIO()
        s3_client.download_fileobj(Bucket=s3_parameters["bucket_name"], Key=s3_parameters["model_filename"], Fileobj=stream)
        stream.seek(0)
        onnx_model = onnx.load_model_from_string(stream.getvalue())
        model = convert(onnx_model)

        stream = BytesIO()
        state_filename = '.'.join(s3_parameters["model_filename"].split('.')[:-1]) + "_state_range.json"
        s3_client.download_fileobj(Bucket=s3_parameters["bucket_name"], Key=state_filename, Fileobj=stream)
        stream.seek(0)
        state_range = json.loads(stream.read().decode("utf-8"))
        pv_generation_range = state_range["pv_generation"]
        energy_consumption_range = state_range["energy_consumption"]
        temp_outside_range = state_range["temperature"]
    except botocore.exceptions.ClientError:
        raise Exception(f'Error loading trained model ({s3_parameters["model_filename"]})')

    state = (
        (state_datetime.hour + state_datetime.minute / 60) / 24,
        (pv_generation_pred - pv_generation_range[0]) / (pv_generation_range[1] - pv_generation_range[0]),
        (energy_consumption_pred - energy_consumption_range[0]) / (energy_consumption_range[1] - energy_consumption_range[0]),
        (temp_inside - min_temp_setting) / max_temp_setting,
        (temp_inside - (pref_temp - temp_window)) / (max_temp_setting - min_temp_setting),
        (pref_temp + temp_window - temp_inside) / (max_temp_setting - min_temp_setting),
        (temp_outside_pred - temp_outside_range[0]) / (temp_outside_range[1] - temp_outside_range[0]),
        storage_soc / 100,
    )
    for ev_id in ev_id_list:
        ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
        is_ev_available = ev_battery_parameters["is_available"]
        hours_till_ev_departure = ev_battery_parameters["time_until_charged"] / 3600
        ev_soc = ev_battery_parameters["curr_charge_level"]
        state += (
            is_ev_available,
            hours_till_ev_departure / 24,
            ev_soc / 100,
        )

    action = select_action(
        model,
        torch.tensor(state, dtype=torch.float).unsqueeze(0),
        lower_bounds,
        upper_bounds,
    )
    temp_setting = action[0]
    storage_charging_power = action[1]
    ev_charging_power_list = action[2:]

    storage_params = {"InWRte": 0.0, "OutWRte": 0.0}
    if storage_charging_power > 0:
        storage_params["InWRte"] = storage_charging_power / storage_max_charging_power * 100.
        storage_params["StorCtl_Mod"] = 1
    else:
        storage_params["OutWRte"] = - storage_charging_power / storage_max_charging_power * 100.
        storage_params["StorCtl_Mod"] = 2

    ev_params_per_id = {}
    for ev_id, ev_charging_power in zip(ev_id_list, ev_charging_power_list):
        ev_max_charging_power = ev_battery_parameters_per_id[ev_id]["nominal_power"]
        ev_params = {"InWRte": 0.0, "OutWRte": 0.0}
        if ev_charging_power > 0:
            ev_params["InWRte"] = ev_charging_power / ev_max_charging_power * 100.
            ev_params["StorCtl_Mod"] = 1
        else:
            ev_params["StorCtl_Mod"] = 0
        ev_params_per_id[ev_id] = ev_params

    return (
        temp_setting,
        json.dumps(storage_params),
        json.dumps(ev_params_per_id)
    )


def train(
        train_parameters: str,
        s3_parameters: str,
        besmart_parameters: str,
        home_model_parameters: str,
        storage_parameters: str,
        ev_battery_parameters_per_id: str,
        heating_parameters: str,
        user_preferences: str,
) -> bool:
    import datetime
    import json
    import logging
    import math
    from io import BytesIO
    from typing import Any

    import boto3
    import numpy as np
    import torch
    import requests
    from torch import nn
    from torch.distributions import MultivariateNormal
    from torch.utils.data import TensorDataset, DataLoader

    def get_state(index: int) -> tuple[float, ...]:
        state = (
            (time.hour + time.minute / 60) / 24,
            pv_generation_pred_list[index] / pv_generation_max,
            energy_consumption_pred_list[index] / energy_consumption_max,
            (temp_inside - min_temp_setting) / max_temp_setting,
            (temp_inside - (pref_temperature - temp_window)) / (max_temp_setting - min_temp_setting),
            (pref_temperature + temp_window - temp_inside) / (max_temp_setting - min_temp_setting),
            (temp_outside_list[index] - temp_outside_min) / (temp_outside_max - temp_outside_min),
            storage_soc / 100,
        )

        for ev_id in ev_id_list:
            ev_driving_state = ev_driving_state_per_id[ev_id]
            ev_driving_power = ev_driving_state["driving_power"]
            hours_till_ev_departure = ev_driving_state["time_till_departure"].seconds / 3600
            state += (
                float(ev_driving_power == 0.),
                hours_till_ev_departure / 24,
                ev_soc_per_id[ev_id] / 100,
            )

        return state

    def get_ev_driving_state(ev_driving_schedule: dict[str, Any]) -> dict[str, Any]:
        ev_driving_time = ev_driving_schedule["time"]
        ev_schedule_ind = np.where(time >= ev_driving_time)[0][-1]
        ev_driving_power = ev_driving_schedule["driving_power"][ev_schedule_ind]

        next_driving_power_arr = np.array(ev_driving_schedule["driving_power"][ev_schedule_ind + 1:]
                                          + ev_driving_schedule["driving_power"][:ev_schedule_ind + 1])
        next_driving_time_arr = np.concatenate((ev_driving_time[ev_schedule_ind + 1:],
                                                ev_driving_time[:ev_schedule_ind + 1]))
        next_ev_departure_time = next_driving_time_arr[np.where(next_driving_power_arr > 0.)[0][0]]
        next_ev_departure_timestamp = datetime.datetime.strptime(
            f"{next_ev_departure_time.hour}:{next_ev_departure_time.minute}", "%H:%M")
        if next_ev_departure_time < time:
            next_ev_departure_timestamp = next_ev_departure_timestamp + datetime.timedelta(days=1)
        time_till_ev_departure = (next_ev_departure_timestamp
                                  - datetime.datetime.strptime(f"{time.hour}:{time.minute}", "%H:%M"))

        return {
            "driving_power": ev_driving_power,
            "time_till_departure": time_till_ev_departure,
        }

    def get_preferred_temperature() -> float:
        pref_temp_schedule_ind = np.where(time >= pref_temp_schedule_time)[0][-1]
        return pref_temp_schedule["temp"][pref_temp_schedule_ind]


    def get_reward(
            controlled_consumption_t: float,
            temp_inside_t: float,
            storage_soc_t: float,
            ev_soc_per_id_t: dict[int, float],
            dt: int
    ) -> float:
        energy_balance = pv_generation - energy_consumption - controlled_consumption_t
        temperature_error = max(np.abs(temp_inside_t - pref_temperature) - temp_window, 0.)
        storage_soc_error = (max(storage_soc_t - 100., 0.)
                             + max(storage_min_charge_level - storage_soc_t, 0.)
                             ) / 100. * storage_max_capacity

        ev_soc_error = 0
        for ev_id in ev_id_list:
            ev_driving_state = ev_driving_state_per_id[ev_id]
            ev_driving_power = ev_driving_state["driving_power"]
            time_till_ev_departure = ev_driving_state["time_till_departure"].seconds
            ev_soc_t = ev_soc_per_id_t[ev_id]
            if ev_driving_power == 0.:
                ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
                ev_min_charge_level = ev_battery_parameters["min_charge_level"]
                ev_max_capacity = ev_battery_parameters["max_capacity"]
                ev_soc_error += (max(ev_soc_t - 100., 0.)
                                 + max(ev_min_charge_level - ev_soc_t, 0.)
                                 ) / 100. * ev_max_capacity
                if time_till_ev_departure <= dt:
                    ev_driving_charge_level = ev_battery_parameters["driving_charge_level"]
                    ev_soc_error += max(ev_driving_charge_level - ev_soc_t, 0.) / 100. * ev_max_capacity


        energy_balance_reward = - energy_reward_coeff * np.abs(energy_balance)
        temperature_reward = - temp_reward_coeff * temperature_error
        storage_reward = - storage_reward_coeff * storage_soc_error
        ev_reward = - ev_reward_coeff * ev_soc_error
        return energy_balance_reward + temperature_reward + storage_reward + ev_reward

    def step(
            actions: tuple[float, ...],
            temp_inside_t: float,
            storage_soc_t: float,
            ev_soc_per_id_t: dict[int, float],
            dt: int
    ) -> tuple[float, float, float, dict, bool]:
        temp_setting = actions[0]
        storage_charging_power = actions[1]
        ev_charging_power_list = actions[2:]

        delta_temp = temp_inside_t - temp_setting
        if abs(delta_temp) > temp_window:
            next_is_heating_on = delta_temp < 0
        else:
            next_is_heating_on = is_heating_on
        heating_energy = (
                heating_coefficient * next_is_heating_on * heating_devices_power * 1000 * dt
                - heat_loss_coefficient * (temp_inside_t - temp_outside) * dt
        )  # J
        delta_temp = heating_energy / heat_capacity
        next_temp_inside = temp_inside_t + delta_temp
        heating_consumption = next_is_heating_on * heating_devices_power * dt / 3600

        storage_power_reduction = min(1.0, max(epsilon, (100. - storage_soc_t) / (100. - storage_charging_switch_level)))
        delta_capacity = storage_charging_power * dt / 3600 * (
            storage_efficiency * storage_power_reduction if storage_charging_power > 0 else 1.)
        next_storage_soc = storage_soc_t + delta_capacity / storage_max_capacity * 100.0
        next_storage_soc = (1.0 - storage_energy_loss * dt / 100.0) * next_storage_soc
        next_storage_soc = min(max(next_storage_soc, epsilon), 100.0)
        real_delta_capacity = (next_storage_soc - storage_soc_t) / 100. * storage_max_capacity
        storage_consumption = real_delta_capacity / (
                storage_power_reduction * storage_efficiency if storage_charging_power > 0 else 1.)

        next_ev_soc_per_id = {}
        ev_consumption = 0.
        for ev_id, ev_charging_power in zip(ev_id_list, ev_charging_power_list):
            ev_driving_power = ev_driving_state_per_id[ev_id]["driving_power"]
            ev_soc_t = ev_soc_per_id_t[ev_id]
            ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
            ev_charging_switch_level = ev_battery_parameters["charging_switch_level"]
            ev_efficiency = ev_battery_parameters["efficiency"]
            ev_max_capacity = ev_battery_parameters["max_capacity"]
            ev_energy_loss = ev_battery_parameters["energy_loss"]
            if ev_driving_power == 0.:
                ev_power_reduction = min(1.0, max(epsilon, (100. - ev_soc_t) / (100. - ev_charging_switch_level)))
                delta_capacity = ev_charging_power * dt / 3600 * ev_efficiency * ev_power_reduction
                next_ev_soc = ev_soc_t + delta_capacity / ev_max_capacity * 100.0
                next_ev_soc = (1.0 - ev_energy_loss * dt / 100.0) * next_ev_soc
                next_ev_soc = min(max(next_ev_soc, epsilon), 100.0)
                real_delta_capacity = (next_ev_soc - ev_soc_t) / 100. * ev_max_capacity
                ev_consumption = real_delta_capacity / (ev_power_reduction * ev_efficiency)
            else:
                next_ev_soc = ev_soc_t - ev_driving_power * dt / 3600 / ev_max_capacity * 100.0
                next_ev_soc = max(next_ev_soc, epsilon)
                ev_consumption = 0.
            next_ev_soc_per_id[ev_id] = next_ev_soc

        controlled_consumption = heating_consumption + storage_consumption + ev_consumption
        reward_t = get_reward(
            controlled_consumption,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc_per_id,
            dt
        )

        return (
            reward_t,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc_per_id,
            next_is_heating_on,
        )

    class RolloutBuffer:
        def __init__(self):
            self.actions = []
            self.states = []
            self.logprobs = []
            self.rewards = []
            self.state_values = []
            self.is_terminals = []

        def clear(self):
            del self.actions[:]
            del self.states[:]
            del self.logprobs[:]
            del self.rewards[:]
            del self.state_values[:]
            del self.is_terminals[:]

    class ActorCritic(nn.Module):
        def __init__(self):
            super(ActorCritic, self).__init__()

            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds

            # actor
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
            # critic
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 1)
            )

        def set_action_std(self, new_action_std: float):
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

        def forward(self):
            raise NotImplementedError

        def act(
                self,
                tensor_state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            action_mean = self.actor(tensor_state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            tensor_action = dist.sample()
            tensor_action = torch.clip(tensor_action, -1, 1)

            action_logprob = dist.log_prob(tensor_action)
            state_val = self.critic(tensor_state)
            return tensor_action.detach(), action_logprob.detach(), state_val.detach()

        def evaluate(
                self,
                tensor_state: torch.Tensor,
                tensor_action: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            action_mean = self.actor(tensor_state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action_logprobs = dist.log_prob(tensor_action)
            dist_entropy = dist.entropy()
            state_values = self.critic(tensor_state)

            return action_logprobs, state_values, dist_entropy

    def select_action(tensor_state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.FloatTensor(tensor_state).to(device)
            tensor_action, action_logprob, state_val = policy_old.act(tensor_state)

        buffer.states.append(tensor_state)
        buffer.actions.append(tensor_action)
        buffer.logprobs.append(action_logprob)
        buffer.state_values.append(state_val)

        tensor_action = tensor_action.detach().cpu().numpy().flatten()
        tensor_action[0] = tensor_action[0] * (upper_bounds[0] - lower_bounds[0]) / 2 + (
                    upper_bounds[0] + lower_bounds[0]) / 2
        tensor_action[1] = tensor_action[1] * upper_bounds[1]
        for i in range(len(ev_id_list)):
            tensor_action[2 + i] = (tensor_action[2 + i] + 1) * upper_bounds[2 + i] / 2
        tensor_action = np.clip(tensor_action, np.array(lower_bounds), np.array(upper_bounds))

        return tensor_action

    def update():
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32).to(device)
        advantages, target_values = advantage(rewards, buffer.is_terminals, old_state_values)

        dataset = TensorDataset(old_states, old_actions, old_logprobs, old_state_values, advantages, target_values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimize policy for K epochs
        for _ in range(number_of_epochs):
            for old_states, old_actions, old_logprobs, old_state_values, advantages, target_values in dataloader:
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * mse_loss(state_values, target_values)

                # take gradient step
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

        # Copy new weights into old policy
        policy_old.load_state_dict(policy.state_dict())
        buffer.clear()

    def advantage(
            rewards: torch.Tensor,
            done: list,
            values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros(len(rewards), dtype=torch.float)
        last_advantage = 0
        last_value = values[-1]
        for t in reversed(range(len(rewards))):
            mask = 1.0 - done[t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[t] + gamma * last_value - values[t]
            last_advantage = delta + gamma * lambda_ * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        target_values = advantages + values

        return advantages, target_values

    def decay_action_std(current_action_std: float) -> float:
        new_action_std = current_action_std - action_std_decay_rate
        new_action_std = round(new_action_std, 4)
        if new_action_std <= min_action_std:
            new_action_std = min_action_std
        policy.set_action_std(new_action_std)
        policy_old.set_action_std(new_action_std)

        return new_action_std

    def get_data_from_besmart(
            cid: int,
            mid: int,
            moid: int,
            is_cumulative: bool = False
    ) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        since = int(besmart_parameters["since"]) - cycle_timedelta_s
        if is_cumulative:
            since -= cycle_timedelta_s
        body = [{
            "client_cid": cid,
            "sensor_mid": mid,
            "signal_type_moid": moid,
            "since": since * 1000,
            "till": int(besmart_parameters["till"]) * 1000,
            "get_last": True,
        }]
        res = requests.post(
            'https://api.besmart.energy/api/sensors/signals/data',
            headers=headers, json=body
        )
        if res.status_code == 200:
            return res.json()[0]['data']
        return res.status_code

    def get_energy_data(
            identifier: dict[str, int],
            is_cumulative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        data = get_data_from_besmart(identifier["cid"],
                                     identifier["mid"],
                                     identifier["moid"],
                                     is_cumulative)
        try:
            time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
            value = np.array(data['value'])
            origin = np.array(data['origin'])
        except Exception as e:
            raise Exception(f'{e} - besmart returned HTTP {data}')

        real_value = value[origin == 1]
        real_time = time[origin == 1]
        pred_value = value[origin == 2]
        pred_time = time[origin == 2]

        try:
            real_value, real_time = validate_data(real_time, real_value, is_cumulative)
            pred_value, pred_time = validate_data(pred_time, pred_value, is_cumulative)
        except ValueError:
            raise Exception(
                'Not enough energy data for training '
                f'(cid: {identifier["cid"]}, mid: {identifier["mid"]}, moid: {identifier["moid"]})'
            )

        if is_cumulative:
            real_value = np.diff(real_value) / (np.diff(real_time.astype(int)) / 60)
            pred_value = np.diff(pred_value) / (np.diff(pred_time.astype(int)) / 60)

        return real_value, pred_value

    def get_temperature_data() -> np.ndarray:
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
            "since": (int(besmart_parameters["since"]) - cycle_timedelta_s) * 1000,
            "till": int(besmart_parameters["till"]) * 1000,
            'delta_t': cycle_timedelta_s // 60,
            'raw': False,
            'get_last': True,
        }
        res = requests.get(
            f'https://api.besmart.energy/api/weather/{sensor["lat"]}/{sensor["lon"]}/{besmart_parameters["temperature_moid"]}/data',
            headers=headers, params=params
        )
        if res.status_code == 200:
            data = res.json()['data']
        else:
            raise Exception(f'Besmart returned HTTP {res.status_code}')

        time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
        value = np.array(data['value'])
        origin = np.array(data['origin'])
        estm_value = value[origin == 3]
        estm_time = time[origin == 3]
        try:
            pred_value, _ = validate_data(estm_time, estm_value)
        except ValueError:
            raise Exception(
                f'Not enough temperature data for training (lat: {sensor["lat"]}, lon: {sensor["lon"]})'
            )

        return pred_value - 272.15

    def validate_data(
            time: np.ndarray,
            value: np.ndarray,
            is_cumulative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        since = np.datetime64(int(besmart_parameters["since"]), "s")
        if is_cumulative:
            since -= np.timedelta64(cycle_timedelta_s, 's')
        expected_time = np.arange(since,
                                  np.datetime64(int(besmart_parameters["till"]), "s"),
                                  np.timedelta64(cycle_timedelta_s, 's')).astype('datetime64[m]')
        missing_time = np.array([t for t in expected_time if t not in time])
        num_missing = len(missing_time)
        if num_missing > 0:
            new_time = np.concatenate((time, missing_time))
            new_value = np.concatenate((value, np.array(len(missing_time) * [np.nan])))
            ind = np.argsort(new_time)
            new_time = new_time[ind]
            new_value = new_value[ind]
            missing_data_mask = np.isnan(new_value)
            sequences_last_indexes = np.append(np.where(missing_data_mask[1:] != missing_data_mask[:-1]),
                                               len(missing_data_mask) - 1)
            sequences_lengths = np.diff(np.append(-1, sequences_last_indexes))
            gap_lengths = sequences_lengths[missing_data_mask[sequences_last_indexes]]
            if np.any(gap_lengths > 2):
                raise ValueError
            new_value = np.interp(new_time.astype('float64'),
                                  new_time[~missing_data_mask].astype('float64'),
                                  new_value[~missing_data_mask])
        else:
            new_value = value.copy()
            new_time = time.copy()
        if len(new_time) > len(expected_time):
            new_value = np.array([v for t, v in zip(new_time, new_value) if t in expected_time])

        return new_value, expected_time


    epsilon = 1e-8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_parameters = json.loads(train_parameters)
    s3_parameters = json.loads(s3_parameters)
    besmart_parameters = json.loads(besmart_parameters)
    home_model_parameters = json.loads(home_model_parameters)
    storage_parameters = json.loads(storage_parameters)
    ev_battery_parameters_per_id = json.loads(ev_battery_parameters_per_id) if (
            ev_battery_parameters_per_id != json.dumps(None)) else {}
    heating_parameters = json.loads(heating_parameters)
    user_preferences = json.loads(user_preferences)

    number_of_episodes = train_parameters["num_episodes"]
    lr_critic = train_parameters["critic_lr"]
    lr_actor = train_parameters["actor_lr"]
    gamma = train_parameters["gamma"]
    lambda_ = train_parameters["lambda_"]
    number_of_epochs = train_parameters["num_epochs"]
    eps_clip = train_parameters["eps_clip"]
    min_action_std = train_parameters["min_action_std"]
    action_std_decay_freq = math.floor(train_parameters["action_std_decay_freq"] * number_of_episodes)
    action_std_decay_rate = train_parameters["action_std_decay_rate"]
    update_epoch = train_parameters["update_epoch"]
    action_std_init = train_parameters["action_std_init"]
    batch_size = train_parameters["batch_size"]
    energy_reward_coeff = train_parameters["energy_reward_coeff"]
    temp_reward_coeff = train_parameters["temp_reward_coeff"]
    storage_reward_coeff = train_parameters["storage_reward_coeff"]
    ev_reward_coeff = train_parameters["ev_reward_coeff"]

    heating_coefficient = home_model_parameters["heating_coefficient"]
    heat_loss_coefficient = home_model_parameters["heat_loss_coefficient"]
    heat_capacity = home_model_parameters["heat_capacity"]
    temp_window = home_model_parameters["temp_window"]
    min_temp_setting = home_model_parameters["min_temp_setting"]
    max_temp_setting = home_model_parameters["max_temp_setting"]

    heating_devices_power = sum(heating_parameters["powers_of_heating_devices"])

    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_charging_switch_level = storage_parameters["charging_switch_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_energy_loss = storage_parameters["energy_loss"]
    storage_nominal_power = storage_parameters["nominal_power"]

    ev_driving_schedule_per_id = user_preferences["ev_driving_schedule"]
    pref_temp_schedule = user_preferences["pref_temp_schedule"]
    pref_temp_schedule_time = np.array([datetime.datetime.strptime(t, "%H:%M").time()
                                        for t in pref_temp_schedule["time"]])
    cycle_timedelta_s = user_preferences["cycle_timedelta_s"]

    ev_id_list = list(ev_battery_parameters_per_id.keys())
    ev_id_list.sort()
    for ev_driving_schedule_dict in ev_driving_schedule_per_id.values():
        ev_driving_schedule_dict["time"] = np.array([datetime.datetime.strptime(t, "%H:%M").time()
                                                     for t in ev_driving_schedule_dict["time"]])

    lower_bounds = [min_temp_setting, - storage_nominal_power] + len(ev_id_list) * [0.]
    upper_bounds = [max_temp_setting, storage_nominal_power] + [
        ev_battery_parameters_per_id[ev_id]["nominal_power"] for ev_id in ev_id_list]
    state_dim = 8 + 3 * len(ev_id_list)
    action_dim = len(lower_bounds)

    policy = ActorCritic()
    policy_old = ActorCritic()
    buffer = RolloutBuffer()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW([
        {'params': policy.actor.parameters(), 'lr': lr_actor},
        {'params': policy.critic.parameters(), 'lr': lr_critic}
    ])
    action_std = action_std_init

    timestamps = np.arange(np.datetime64(int(besmart_parameters["since"]), "s"),
                           np.datetime64(int(besmart_parameters["till"]), "s"),
                           datetime.timedelta(seconds=cycle_timedelta_s))

    token = besmart_parameters["token"]
    pv_generation_real, pv_generation_pred = get_energy_data(besmart_parameters["pv_generation"])
    energy_consumption_real, energy_consumption_pred = get_energy_data(besmart_parameters["energy_consumption"], True)
    temp_outside_pred = get_temperature_data()

    pv_generation_max = np.max(pv_generation_pred)
    energy_consumption_max = np.max(energy_consumption_pred)
    temp_outside_min = np.min(temp_outside_pred)
    temp_outside_max = np.max(temp_outside_pred)

    ep_reward_list = []
    number_of_cycles = datetime.timedelta(days=1) // datetime.timedelta(seconds=cycle_timedelta_s)
    max_train_index = len(timestamps) - number_of_cycles
    train_indexes = np.random.randint(max_train_index, size=(max_train_index,))
    for ep in range(number_of_episodes):
        episode_start_index = train_indexes[ep % len(train_indexes)]
        episode_end_index = episode_start_index + number_of_cycles + 1

        timestamps_list = timestamps[episode_start_index:episode_end_index]
        pv_generation_real_list = pv_generation_real[episode_start_index:episode_end_index]
        pv_generation_pred_list = pv_generation_pred[episode_start_index:episode_end_index]
        energy_consumption_real_list = energy_consumption_real[episode_start_index:episode_end_index]
        energy_consumption_pred_list = energy_consumption_pred[episode_start_index:episode_end_index]
        temp_outside_list = temp_outside_pred[episode_start_index:episode_end_index]

        ts = (timestamps_list[0] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).time()
        pref_temperature = get_preferred_temperature()
        temp_inside = pref_temperature + np.random.uniform(- temp_window, temp_window)
        storage_soc = np.random.uniform(storage_min_charge_level, 100.)
        ev_soc_per_id = {
            ev_id: np.random.uniform(ev_battery_parameters["min_charge_level"], 100.)
            for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items()
        }
        is_heating_on = bool(np.random.randint(2))

        episodic_reward = 0
        for i_cycle in range(number_of_cycles):
            ts = (timestamps_list[i_cycle] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            time = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).time()
            ev_driving_state_per_id = {
                ev_id: get_ev_driving_state(ev_driving_schedule_per_id[ev_id]) for ev_id in ev_id_list
            }
            pref_temperature = get_preferred_temperature()
            state = get_state(index=i_cycle)
            action = select_action(torch.tensor(state, dtype=torch.float).unsqueeze(0))

            pv_generation = pv_generation_real_list[i_cycle]
            energy_consumption = energy_consumption_real_list[i_cycle]
            temp_outside = temp_outside_list[i_cycle]

            # Receive state and reward from environment.
            reward, temp_inside, storage_soc, ev_soc_per_id, is_heating_on = step(
                action, temp_inside, storage_soc, ev_soc_per_id, cycle_timedelta_s
            )
            buffer.rewards.append(reward)
            if i_cycle == number_of_cycles - 1:
                buffer.is_terminals.append(1)
            else:
                buffer.is_terminals.append(0)
            episodic_reward += reward

        if ep % action_std_decay_freq == 0:
            action_std = decay_action_std(action_std)
        if ep % update_epoch == 0:
            update()

        ep_reward_list.append(episodic_reward)
        logging.debug(f"Episode * {ep} * Avg Reward is ==> {np.mean(ep_reward_list[-100:])} " + f"* Std {action_std}")

    example_state = get_state(index=0)
    example_inputs = (torch.FloatTensor(torch.tensor(example_state, dtype=torch.float).unsqueeze(0).to(device)), )
    tmp_stream = BytesIO()
    torch.onnx.export(
        policy_old.actor,
        example_inputs,
        tmp_stream,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'}}
    )
    tmp_stream.seek(0)

    s3 = boto3.resource(
        "s3",
        endpoint_url=s3_parameters["endpoint_url"],
        aws_access_key_id=s3_parameters["access_key_id"],
        aws_secret_access_key=s3_parameters["secret_access_key"],
    )
    bucket = s3.Bucket(s3_parameters["bucket_name"])
    bucket.put_object(Key=s3_parameters["model_filename"], Body=tmp_stream.getvalue())

    state_range = {
        "energy_consumption": [0.0, energy_consumption_max],
        "pv_generation": [0.0, pv_generation_max],
        "temperature": [temp_outside_min, temp_outside_max],
    }
    state_filename = '.'.join(s3_parameters["model_filename"].split('.')[:-1]) + "_state_range.json"
    bucket.put_object(Key=state_filename, Body=(bytes(json.dumps(state_range).encode('UTF-8'))))

    return True


def evaluate(
        eval_parameters: str,
        s3_parameters: str,
        besmart_parameters: str,
        home_model_parameters: str,
        storage_parameters: str,
        ev_battery_parameters_per_id: str,
        heating_parameters: str,
        user_preferences: str,
) -> str | bool:
    import datetime
    import json
    from io import BytesIO
    from typing import Any

    import botocore
    import boto3
    import numpy as np
    import onnx
    import torch
    import requests
    from onnx2torch import convert

    def select_action(
            actor_model: torch.fx.graph_module.GraphModule,
            tensor_state: torch.Tensor,
            lower_bounds: list[float],
            upper_bounds: list[float]
    ) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.FloatTensor(tensor_state).to(device)
            tensor_action = actor_model(tensor_state)

        tensor_action = tensor_action.detach().cpu().numpy().flatten()
        tensor_action[0] = tensor_action[0] * (upper_bounds[0] - lower_bounds[0]) / 2 + (
                upper_bounds[0] + lower_bounds[0]) / 2
        tensor_action[1] = tensor_action[1] * upper_bounds[1]
        for i in range(len(ev_id_list)):
            tensor_action[2 + i] = (tensor_action[2 + i] + 1) * upper_bounds[2 + i] / 2
        tensor_action = np.clip(tensor_action, np.array(lower_bounds), np.array(upper_bounds))

        return tensor_action

    def get_state(index: int) -> tuple[float, ...]:
        state = (
            (time.hour + time.minute / 60) / 24,
            (pv_generation_pred[index] - pv_generation_range[0]) / (pv_generation_range[1] - pv_generation_range[0]),
            (energy_consumption_pred[index] - energy_consumption_range[0]) / (
                    energy_consumption_range[1] - energy_consumption_range[0]),
            (temp_inside - min_temp_setting) / max_temp_setting,
            (temp_inside - (pref_temperature - temp_window)) / (max_temp_setting - min_temp_setting),
            (pref_temperature + temp_window - temp_inside) / (max_temp_setting - min_temp_setting),
            (temp_outside_pred[index] - temp_outside_range[0]) / (temp_outside_range[1] - temp_outside_range[0]),
            storage_soc / 100,
        )

        for ev_id in ev_id_list:
            ev_driving_state = ev_driving_state_per_id[ev_id]
            ev_driving_power = ev_driving_state["driving_power"]
            hours_till_ev_departure = ev_driving_state["time_till_departure"].seconds / 3600
            state += (
                float(ev_driving_power == 0.),
                hours_till_ev_departure / 24,
                ev_soc_per_id[ev_id] / 100,
            )

        return state

    def get_ev_driving_state(ev_driving_schedule: dict[str, Any]) -> dict[str, Any]:
        ev_driving_time = ev_driving_schedule["time"]
        ev_schedule_ind = np.where(time >= ev_driving_time)[0][-1]
        ev_driving_power = ev_driving_schedule["driving_power"][ev_schedule_ind]

        next_driving_power_arr = np.array(ev_driving_schedule["driving_power"][ev_schedule_ind + 1:]
                                          + ev_driving_schedule["driving_power"][:ev_schedule_ind + 1])
        next_driving_time_arr = np.concatenate((ev_driving_time[ev_schedule_ind + 1:],
                                                ev_driving_time[:ev_schedule_ind + 1]))
        next_ev_departure_time = next_driving_time_arr[np.where(next_driving_power_arr > 0.)[0][0]]
        next_ev_departure_timestamp = datetime.datetime.strptime(
            f"{next_ev_departure_time.hour}:{next_ev_departure_time.minute}", "%H:%M")
        if next_ev_departure_time < time:
            next_ev_departure_timestamp = next_ev_departure_timestamp + datetime.timedelta(days=1)
        time_till_ev_departure = (next_ev_departure_timestamp
                                  - datetime.datetime.strptime(f"{time.hour}:{time.minute}", "%H:%M"))

        return {
            "driving_power": ev_driving_power,
            "time_till_departure": time_till_ev_departure,
        }

    def get_preferred_temperature() -> float:
        pref_temp_schedule_ind = np.where(time >= pref_temp_schedule_time)[0][-1]
        return pref_temp_schedule["temp"][pref_temp_schedule_ind]

    def get_reward(
            controlled_consumption_t: float,
            temp_inside_t: float,
            storage_soc_t: float,
            ev_soc_per_id_t: dict[int, float],
            dt: int
    ) -> tuple[float, float]:
        energy_balance = pv_generation - energy_consumption - controlled_consumption_t
        temperature_error = max(np.abs(temp_inside_t - pref_temperature) - temp_window, 0.)
        storage_soc_error = (max(storage_soc_t - 100., 0.)
                             + max(storage_min_charge_level - storage_soc_t, 0.)
                             ) / 100. * storage_max_capacity

        ev_soc_error = 0
        for ev_id in ev_id_list:
            ev_driving_state = ev_driving_state_per_id[ev_id]
            ev_driving_power = ev_driving_state["driving_power"]
            time_till_ev_departure = ev_driving_state["time_till_departure"].seconds
            ev_soc_t = ev_soc_per_id_t[ev_id]
            if ev_driving_power == 0.:
                ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
                ev_min_charge_level = ev_battery_parameters["min_charge_level"]
                ev_max_capacity = ev_battery_parameters["max_capacity"]
                ev_soc_error += (max(ev_soc_t - 100., 0.)
                                 + max(ev_min_charge_level - ev_soc_t, 0.)
                                 ) / 100. * ev_max_capacity
                if time_till_ev_departure <= dt:
                    ev_driving_charge_level = ev_battery_parameters["driving_charge_level"]
                    ev_soc_error += max(ev_driving_charge_level - ev_soc_t, 0.) / 100. * ev_max_capacity


        energy_balance_reward = - energy_reward_coeff * np.abs(energy_balance)
        temperature_reward = - temp_reward_coeff * temperature_error
        storage_reward = - storage_reward_coeff * storage_soc_error
        ev_reward = - ev_reward_coeff * ev_soc_error
        return (
            energy_balance_reward + temperature_reward + storage_reward + ev_reward,
            np.abs(energy_balance)
        )

    def step(
            actions: tuple[float, ...],
            temp_inside_t: float,
            storage_soc_t: float,
            ev_soc_per_id_t: dict[int, float],
            dt: int
    ) -> tuple[float, float, float, float, dict, bool]:
        temp_setting = actions[0]
        storage_charging_power = actions[1]
        ev_charging_power_list = actions[2:]

        delta_temp = temp_inside_t - temp_setting
        if abs(delta_temp) > temp_window:
            next_is_heating_on = delta_temp < 0
        else:
            next_is_heating_on = is_heating_on
        heating_energy = (
                heating_coefficient * next_is_heating_on * heating_devices_power * 1000 * dt
                - heat_loss_coefficient * (temp_inside_t - temp_outside) * dt
        )  # J
        delta_temp = heating_energy / heat_capacity
        next_temp_inside = temp_inside_t + delta_temp
        heating_consumption = next_is_heating_on * heating_devices_power * dt / 3600

        storage_power_reduction = min(1.0, max(epsilon, (100. - storage_soc_t) / (100. - storage_charging_switch_level)))
        delta_capacity = storage_charging_power * dt / 3600 * (
            storage_efficiency * storage_power_reduction if storage_charging_power > 0 else 1.)
        next_storage_soc = storage_soc_t + delta_capacity / storage_max_capacity * 100.0
        next_storage_soc = (1.0 - storage_energy_loss * dt / 100.0) * next_storage_soc
        next_storage_soc = min(max(next_storage_soc, epsilon), 100.0)
        real_delta_capacity = (next_storage_soc - storage_soc_t) / 100. * storage_max_capacity
        storage_consumption = real_delta_capacity / (
                storage_power_reduction * storage_efficiency if storage_charging_power > 0 else 1.)

        next_ev_soc_per_id = {}
        ev_consumption = 0.
        for ev_id, ev_charging_power in zip(ev_id_list, ev_charging_power_list):
            ev_driving_power = ev_driving_state_per_id[ev_id]["driving_power"]
            ev_soc_t = ev_soc_per_id_t[ev_id]
            ev_battery_parameters = ev_battery_parameters_per_id[ev_id]
            ev_charging_switch_level = ev_battery_parameters["charging_switch_level"]
            ev_efficiency = ev_battery_parameters["efficiency"]
            ev_max_capacity = ev_battery_parameters["max_capacity"]
            ev_energy_loss = ev_battery_parameters["energy_loss"]
            if ev_driving_power == 0.:
                ev_power_reduction = min(1.0, max(epsilon, (100. - ev_soc_t) / (100. - ev_charging_switch_level)))
                delta_capacity = ev_charging_power * dt / 3600 * ev_efficiency * ev_power_reduction
                next_ev_soc = ev_soc_t + delta_capacity / ev_max_capacity * 100.0
                next_ev_soc = (1.0 - ev_energy_loss * dt / 100.0) * next_ev_soc
                next_ev_soc = min(max(next_ev_soc, epsilon), 100.0)
                real_delta_capacity = (next_ev_soc - ev_soc_t) / 100. * ev_max_capacity
                ev_consumption = real_delta_capacity / (ev_power_reduction * ev_efficiency)
            else:
                next_ev_soc = ev_soc_t - ev_driving_power * dt / 3600 / ev_max_capacity * 100.0
                next_ev_soc = max(next_ev_soc, epsilon)
                ev_consumption = 0.
            next_ev_soc_per_id[ev_id] = next_ev_soc

        controlled_consumption = heating_consumption + storage_consumption + ev_consumption
        reward_t, energy_balance_t = get_reward(
            controlled_consumption,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc_per_id,
            dt
        )

        return (
            reward_t,
            energy_balance_t,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc_per_id,
            next_is_heating_on,
        )

    def get_data_from_besmart(
            cid: int,
            mid: int,
            moid: int,
            is_cumulative: bool = False
    ) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        since = int(besmart_parameters["since"]) - cycle_timedelta_s
        if is_cumulative:
            since -= cycle_timedelta_s
        body = [{
            "client_cid": cid,
            "sensor_mid": mid,
            "signal_type_moid": moid,
            "since": since * 1000,
            "till": int(besmart_parameters["till"]) * 1000,
            "get_last": True,
        }]
        res = requests.post(
            'https://api.besmart.energy/api/sensors/signals/data',
            headers=headers, json=body
        )
        if res.status_code == 200:
            return res.json()[0]['data']
        return res.status_code

    def get_energy_data(
            identifier: dict[str, int],
            is_cumulative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        data = get_data_from_besmart(identifier["cid"],
                                     identifier["mid"],
                                     identifier["moid"],
                                     is_cumulative)
        try:
            time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
            value = np.array(data['value'])
            origin = np.array(data['origin'])
        except Exception as e:
            raise Exception(f'{e} - besmart returned HTTP {data}')

        real_value = value[origin == 1]
        real_time = time[origin == 1]
        pred_value = value[origin == 2]
        pred_time = time[origin == 2]

        try:
            real_value, real_time = validate_data(real_time, real_value, is_cumulative)
            pred_value, pred_time = validate_data(pred_time, pred_value, is_cumulative)
        except ValueError:
            raise Exception(
                'Not enough energy data for evaluation '
                f'(cid: {identifier["cid"]}, mid: {identifier["mid"]}, moid: {identifier["moid"]})'
            )

        if is_cumulative:
            real_value = np.diff(real_value) / (np.diff(real_time.astype(int)) / 60)
            pred_value = np.diff(pred_value) / (np.diff(pred_time.astype(int)) / 60)

        return real_value, pred_value

    def get_temperature_data() -> np.ndarray:
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
            "since": (int(besmart_parameters["since"]) - cycle_timedelta_s) * 1000,
            "till": int(besmart_parameters["till"]) * 1000,
            'delta_t': cycle_timedelta_s // 60,
            'raw': False,
            'get_last': True,
        }
        res = requests.get(
            f'https://api.besmart.energy/api/weather/{sensor["lat"]}/{sensor["lon"]}/{besmart_parameters["temperature_moid"]}/data',
            headers=headers, params=params
        )
        if res.status_code == 200:
            data = res.json()['data']
        else:
            raise Exception(f'Besmart returned HTTP {res.status_code}')

        time = (np.array(data['time']) * 1e6).astype(int).astype('datetime64[ns]').astype('datetime64[m]')
        value = np.array(data['value'])
        origin = np.array(data['origin'])
        estm_value = value[origin == 3]
        estm_time = time[origin == 3]
        try:
            pred_value, _ = validate_data(estm_time, estm_value)
        except ValueError:
            raise Exception(
                f'Not enough temperature data for evaluation (lat: {sensor["lat"]}, lon: {sensor["lon"]})'
            )

        return pred_value - 272.15

    def validate_data(
            time: np.ndarray,
            value: np.ndarray,
            is_cumulative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        since = np.datetime64(int(besmart_parameters["since"]), "s")
        if is_cumulative:
            since -= np.timedelta64(cycle_timedelta_s, 's')
        expected_time = np.arange(since,
                                  np.datetime64(int(besmart_parameters["till"]), "s"),
                                  np.timedelta64(cycle_timedelta_s, 's')).astype('datetime64[m]')
        missing_time = np.array([t for t in expected_time if t not in time])
        num_missing = len(missing_time)
        if num_missing > 0:
            new_time = np.concatenate((time, missing_time))
            new_value = np.concatenate((value, np.array(len(missing_time) * [np.nan])))
            ind = np.argsort(new_time)
            new_time = new_time[ind]
            new_value = new_value[ind]
            missing_data_mask = np.isnan(new_value)
            sequences_last_indexes = np.append(np.where(missing_data_mask[1:] != missing_data_mask[:-1]),
                                               len(missing_data_mask) - 1)
            sequences_lengths = np.diff(np.append(-1, sequences_last_indexes))
            gap_lengths = sequences_lengths[missing_data_mask[sequences_last_indexes]]
            if np.any(gap_lengths > 2):
                raise ValueError
            new_value = np.interp(new_time.astype('float64'),
                                  new_time[~missing_data_mask].astype('float64'),
                                  new_value[~missing_data_mask])
        else:
            new_value = value.copy()
            new_time = time.copy()
        if len(new_time) > len(expected_time):
            new_value = np.array([v for t, v in zip(new_time, new_value) if t in expected_time])

        return new_value, expected_time


    epsilon = 1e-8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_parameters = json.loads(eval_parameters)
    s3_parameters = json.loads(s3_parameters)
    besmart_parameters = json.loads(besmart_parameters)
    home_model_parameters = json.loads(home_model_parameters)
    storage_parameters = json.loads(storage_parameters)
    ev_battery_parameters_per_id = json.loads(ev_battery_parameters_per_id) if (
            ev_battery_parameters_per_id != json.dumps(None)) else {}
    heating_parameters = json.loads(heating_parameters)
    user_preferences = json.loads(user_preferences)

    energy_reward_coeff = eval_parameters["energy_reward_coeff"]
    temp_reward_coeff = eval_parameters["temp_reward_coeff"]
    storage_reward_coeff = eval_parameters["storage_reward_coeff"]
    ev_reward_coeff = eval_parameters["ev_reward_coeff"]

    heating_coefficient = home_model_parameters["heating_coefficient"]
    heat_loss_coefficient = home_model_parameters["heat_loss_coefficient"]
    heat_capacity = home_model_parameters["heat_capacity"]
    temp_window = home_model_parameters["temp_window"]
    min_temp_setting = home_model_parameters["min_temp_setting"]
    max_temp_setting = home_model_parameters["max_temp_setting"]

    heating_devices_power = sum(heating_parameters["powers_of_heating_devices"])

    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_charging_switch_level = storage_parameters["charging_switch_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_energy_loss = storage_parameters["energy_loss"]
    storage_nominal_power = storage_parameters["nominal_power"]

    ev_driving_schedule_per_id = user_preferences["ev_driving_schedule"]
    pref_temp_schedule = user_preferences["pref_temp_schedule"]
    pref_temp_schedule_time = np.array([datetime.datetime.strptime(t, "%H:%M").time()
                                        for t in pref_temp_schedule["time"]])
    cycle_timedelta_s = user_preferences["cycle_timedelta_s"]

    ev_id_list = list(ev_battery_parameters_per_id.keys())
    ev_id_list.sort()
    for ev_driving_schedule_dict in ev_driving_schedule_per_id.values():
        ev_driving_schedule_dict["time"] = np.array([datetime.datetime.strptime(t, "%H:%M").time()
                                                     for t in ev_driving_schedule_dict["time"]])

    lower_bounds = [min_temp_setting, - storage_nominal_power] + len(ev_id_list) * [0.]
    upper_bounds = [max_temp_setting, storage_nominal_power] + [
        ev_battery_parameters_per_id[ev_id]["nominal_power"] for ev_id in ev_id_list]

    timestamps = np.arange(np.datetime64(int(besmart_parameters["since"]), "s"),
                           np.datetime64(int(besmart_parameters["till"]), "s"),
                           datetime.timedelta(seconds=cycle_timedelta_s))

    token = besmart_parameters["token"]
    pv_generation_real, pv_generation_pred = get_energy_data(besmart_parameters["pv_generation"])
    energy_consumption_real, energy_consumption_pred = get_energy_data(besmart_parameters["energy_consumption"], True)
    temp_outside_pred = get_temperature_data()

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_parameters["endpoint_url"],
            aws_access_key_id=s3_parameters["access_key_id"],
            aws_secret_access_key=s3_parameters["secret_access_key"],
        )
        stream = BytesIO()
        s3_client.download_fileobj(
            Bucket=s3_parameters["bucket_name"],
            Key=s3_parameters["model_filename"],
            Fileobj=stream
        )
        stream.seek(0)
        onnx_model = onnx.load_model_from_string(stream.getvalue())
        model = convert(onnx_model)

        stream = BytesIO()
        state_filename = '.'.join(s3_parameters["model_filename"].split('.')[:-1]) + "_state_range.json"
        s3_client.download_fileobj(
            Bucket=s3_parameters["bucket_name"],
            Key=state_filename,
            Fileobj=stream
        )
        stream.seek(0)
        state_range = json.loads(stream.read().decode("utf-8"))
        pv_generation_range = state_range["pv_generation"]
        energy_consumption_range = state_range["energy_consumption"]
        temp_outside_range = state_range["temperature"]
    except botocore.exceptions.ClientError:
        return False

    storage_soc = np.random.uniform(storage_min_charge_level, 100.)
    ev_soc_per_id = {
        ev_id: np.random.uniform(ev_battery_parameters["min_charge_level"], 100.)
        for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items()
    }
    is_heating_on = bool(np.random.randint(2))
    number_of_cycles = datetime.timedelta(days=1) // datetime.timedelta(seconds=cycle_timedelta_s)
    remainder_cycles = len(timestamps) % number_of_cycles
    if remainder_cycles != 0:
        timestamps = timestamps[:-remainder_cycles]

    reward_list = []
    energy_balance_list = []
    for i, timestamp in enumerate(timestamps):
        ts = (timestamp - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        time = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).time()
        pref_temperature = get_preferred_temperature()
        if i == 0:
            temp_inside = pref_temperature + np.random.uniform(- temp_window, temp_window)
        ev_driving_state_per_id = {
            ev_id: get_ev_driving_state(ev_driving_schedule_per_id[ev_id]) for ev_id in ev_id_list
        }

        state = get_state(index=i)
        action = select_action(
            model,
            torch.tensor(state, dtype=torch.float).unsqueeze(0),
            lower_bounds,
            upper_bounds,
        )

        pv_generation = pv_generation_real[i]
        energy_consumption = energy_consumption_real[i]
        temp_outside = temp_outside_pred[i]

        # Receive state and reward from environment.
        reward, energy_balance_abs, temp_inside, storage_soc, ev_soc_per_id, is_heating_on = step(
            action, temp_inside, storage_soc, ev_soc_per_id, cycle_timedelta_s
        )
        reward_list.append(reward)
        energy_balance_list.append(energy_balance_abs)

    reward_array = np.array(reward_list).reshape(number_of_cycles, -1)
    energy_balance_array = np.array(energy_balance_list).reshape(number_of_cycles, -1)

    is_valid = bool(np.mean(np.sum(reward_array, axis=0)) > eval_parameters.get("mean_reward_threshold", 0.0))
    if eval_parameters.get("return_metrics", False):
        return json.dumps(
            {
                "is_valid": is_valid,
                "mean_reward": float(np.mean(np.sum(reward_array, axis=0))),
                "mean_energy_balance": float(np.mean(np.sum(energy_balance_array, axis=0))),
            }
        )
    else:
        return is_valid
