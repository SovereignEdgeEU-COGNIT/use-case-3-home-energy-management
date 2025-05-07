from typing import Any

import numpy as np

def make_decision(
        timestamp: float,
        s3_parameters: dict[str, str],
        home_model_parameters: dict[str, float],
        storage_parameters: dict[str, float],
        ev_battery_parameters: dict[str, float],
        room_heating_params_list: list[dict],
        pv_generation: float,
        uncontrolled_consumption: float,
        temp_outside: float,
        cycle_timedelta_s: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    from datetime import datetime
    from io import BytesIO

    import boto3
    import numpy as np
    import onnx
    import torch
    from onnx2torch import convert

    def select_action(actor_model: torch.fx.graph_module.GraphModule,
                      tensor_state: torch.Tensor,
                      lower_bounds: list[float],
                      upper_bounds: list[float]) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.FloatTensor(tensor_state).to(device)
            tensor_action = actor_model(tensor_state)

        tensor_action = tensor_action.detach().cpu().numpy().flatten()
        tensor_action[0] = tensor_action[0] * (upper_bounds[0] - lower_bounds[0]) / 2 + (
                upper_bounds[0] + lower_bounds[0]) / 2
        tensor_action[1] = tensor_action[1] * upper_bounds[1]
        tensor_action[2] = (tensor_action[2] + 1) * upper_bounds[2] / 2
        tensor_action = np.clip(tensor_action, np.array(lower_bounds), np.array(upper_bounds))

        return tensor_action

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_temp_setting = home_model_parameters["min_temp_setting"]
    max_temp_setting = home_model_parameters["max_temp_setting"]
    storage_max_charging_power = storage_parameters["nominal_power"]
    storage_soc = storage_parameters["curr_charge_level"]
    is_ev_available = ev_battery_parameters["is_available"]
    hours_till_ev_departure = ev_battery_parameters["time_until_charged"] / 3600
    ev_max_charging_power = ev_battery_parameters["nominal_power"]
    ev_soc = ev_battery_parameters["curr_charge_level"]
    temp_inside = np.mean(np.array([room["curr_temp"] for room in room_heating_params_list]))
    pref_temp = np.mean(np.array([room["preferred_temp"] for room in room_heating_params_list]))
    temp_window = home_model_parameters["heating_delta_temperature"]
    lower_bounds = [min_temp_setting, - storage_max_charging_power, 0.]
    upper_bounds = [max_temp_setting, storage_max_charging_power, ev_max_charging_power]

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_parameters["endpoint_url"],
        # aws_access_key_id=s3_parameters["access_key_id"],
        # aws_secret_access_key=s3_parameters["secret_access_key"],
    )
    stream = BytesIO()
    s3_client.download_fileobj(Bucket=s3_parameters["bucket_name"], Key=s3_parameters["model_filename"], Fileobj=stream)
    stream.seek(0)
    onnx_model = onnx.load_model_from_string(stream.getvalue())
    model = convert(onnx_model)

    state = (
        float(is_ev_available),
        datetime.fromtimestamp(timestamp).hour / 24,
        hours_till_ev_departure / 24,
        pv_generation / 3,
        uncontrolled_consumption / 3,
        (temp_inside - min_temp_setting) / max_temp_setting,
        (pref_temp - temp_window - min_temp_setting) / max_temp_setting,
        (pref_temp + temp_window - min_temp_setting) / max_temp_setting,
        temp_outside / 30,
        storage_soc / 100,
        ev_soc / 100,
    )

    action = select_action(
        model,
        torch.tensor(state, dtype=torch.float).unsqueeze(0),
        lower_bounds,
        upper_bounds,
    )
    (temp_setting, storage_charging_power, ev_charging_power) = action

    conf_temp_per_room = {}
    for room in room_heating_params_list:
        conf_temp_per_room[room["name"]] = temp_setting

    storage_params = {"InWRte": 0.0, "OutWRte": 0.0}
    if storage_charging_power > 0:
        storage_params["InWRte"] = storage_charging_power / storage_max_charging_power * 100.
        storage_params["StorCtl_Mod"] = 1
    else:
        storage_params["OutWRte"] = - storage_charging_power / storage_max_charging_power * 100.
        storage_params["StorCtl_Mod"] = 2

    ev_params = {"InWRte": 0.0, "OutWRte": 0.0}
    if ev_charging_power > 0:
        ev_params["InWRte"] = ev_charging_power / ev_max_charging_power * 100.
        ev_params["StorCtl_Mod"] = 1
    else:
        ev_params["StorCtl_Mod"] = 0

    return (
        conf_temp_per_room,
        storage_params,
        ev_params,
    )


def training_function(
        train_parameters: dict[str, Any],
        s3_parameters: dict[str, str],
        home_model_parameters: dict[str, Any],
        storage_parameters: dict[str, float],
        ev_battery_parameters: dict[str, float],
        heating_parameters: dict[str, Any],
        cycle_timedelta_s: int,
        timestamps_hour: np.ndarray[int],
        pv_generation_train: np.ndarray[float],
        pv_generation_pred_train: np.ndarray[float],
        uncontrolled_consumption_train: np.ndarray[float],
        uncontrolled_consumption_pred_train: np.ndarray[float],
        temp_outside_train: np.ndarray[float],
        temp_outside_pred_train: np.ndarray[float],
) -> list[float]:
    import logging
    import math
    from io import BytesIO

    import boto3
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import MultivariateNormal
    from torch.utils.data import TensorDataset, DataLoader

    def get_state(index: int) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
        ev_driving_power = ev_driving_state["driving_power"]
        hours_till_ev_departure = ev_driving_state["hours_till_departure"]

        return (
            float(ev_driving_power == 0.),
            hour / 24,
            hours_till_ev_departure / 24,
            pv_generation_pred_list[index] / 3,
            uncontrolled_consumption_pred_list[index] / 3,
            (temp_inside - min_temp_setting) / max_temp_setting,
            (pref_temperature - temp_window - min_temp_setting) / max_temp_setting,
            (pref_temperature + temp_window - min_temp_setting) / max_temp_setting,
            temp_outside_pred_list[index] / 30,
            storage_soc / 100,
            ev_soc / 100,
        )

    def get_ev_driving_state() -> dict[str, float]:
        ev_schedule_ind = np.where(hour >= np.array(ev_driving_schedule["hour"]))[0][-1]
        ev_driving_power = ev_driving_schedule["driving_power"][ev_schedule_ind]

        next_driving_power_arr = np.array(ev_driving_schedule["driving_power"][ev_schedule_ind + 1:]
                                          + ev_driving_schedule["driving_power"][:ev_schedule_ind + 1])
        next_driving_hour_list = (ev_driving_schedule["hour"][ev_schedule_ind + 1:]
                                  + (np.array(ev_driving_schedule["hour"][:ev_schedule_ind + 1]) + 24.).tolist())
        next_ev_departure_hour = next_driving_hour_list[np.where(next_driving_power_arr > 0.)[0][0]]
        hours_till_ev_departure = next_ev_departure_hour - hour

        return {
            "driving_power": ev_driving_power,
            "hours_till_departure": hours_till_ev_departure,
        }

    def get_preferred_temperature() -> float:
        pref_temp_schedule_ind = np.where(hour >= np.array(pref_temp_schedule["hour"]))[0][-1]
        return pref_temp_schedule["temp"][pref_temp_schedule_ind]


    def get_reward(controlled_consumption_t: float,
                   temp_inside_t: float,
                   storage_soc_t: float,
                   ev_soc_t: float,
                   dt: int) -> float:
        energy_balance = pv_generation - uncontrolled_consumption - controlled_consumption_t
        temperature_error = max(np.abs(temp_inside_t - pref_temperature) - temp_window, 0.)
        storage_soc_error = (max(storage_soc_t - 100., 0.)
                             + max(storage_min_charge_level - storage_soc_t, 0.)
                             ) / 100. * storage_max_capacity

        ev_driving_power = ev_driving_state["driving_power"]
        hours_till_ev_departure = ev_driving_state["hours_till_departure"]
        if ev_driving_power == 0.:
            ev_soc_error = (max(ev_soc_t - 100., 0.)
                            + max(ev_min_charge_level - ev_soc_t, 0.)
                            ) / 100. * ev_max_capacity
            if hours_till_ev_departure * 3600. <= dt:
                ev_soc_departure_error = max(ev_driving_charge_level - ev_soc_t, 0.) / 100. * ev_max_capacity
                ev_soc_error += ev_soc_departure_error
        else:
            ev_soc_error = 0

        energy_balance_reward = - energy_reward_coeff * np.abs(energy_balance)
        temperature_reward = - temp_reward_coeff * temperature_error
        storage_reward = - storage_reward_coeff * storage_soc_error
        ev_reward = - ev_reward_coeff * ev_soc_error
        return energy_balance_reward + temperature_reward + storage_reward + ev_reward

    def step(actions: tuple[float, float, float],
             temp_inside_t: float,
             storage_soc_t: float,
             ev_soc_t: float,
             dt: int = 3600) -> tuple[float, float, float, float, bool]:
        temp_setting, storage_charging_power, ev_charging_power = actions

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

        ev_driving_power = ev_driving_state["driving_power"]
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

        controlled_consumption = heating_consumption + storage_consumption + ev_consumption
        reward_t = get_reward(
            controlled_consumption,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc,
            dt
        )

        return (
            reward_t,
            next_temp_inside,
            next_storage_soc,
            next_ev_soc,
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

        def act(self, tensor_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            action_mean = self.actor(tensor_state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            tensor_action = dist.sample()
            tensor_action = torch.clip(tensor_action, -1, 1)

            action_logprob = dist.log_prob(tensor_action)
            state_val = self.critic(tensor_state)
            return tensor_action.detach(), action_logprob.detach(), state_val.detach()

        def evaluate(self,
                     tensor_state: torch.Tensor,
                     tensor_action: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tensor_action[2] = (tensor_action[2] + 1) * upper_bounds[2] / 2
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

    def advantage(rewards: torch.Tensor,
                  done: list,
                  values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    epsilon = 1e-8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    temp_window = home_model_parameters["heating_delta_temperature"]
    min_temp_setting = home_model_parameters["min_temp_setting"]
    max_temp_setting = home_model_parameters["max_temp_setting"]
    ev_driving_schedule = home_model_parameters["ev_driving_schedule"]
    pref_temp_schedule = home_model_parameters["pref_temp_schedule"]

    heating_devices_power = sum(heating_parameters["powers_of_heating_devices"])

    storage_max_capacity = storage_parameters["max_capacity"]
    storage_min_charge_level = storage_parameters["min_charge_level"]
    storage_charging_switch_level = storage_parameters["charging_switch_level"]
    storage_efficiency = storage_parameters["efficiency"]
    storage_energy_loss = storage_parameters["energy_loss"]
    storage_nominal_power = storage_parameters["nominal_power"]

    ev_max_capacity = ev_battery_parameters["max_capacity"]
    ev_min_charge_level = ev_battery_parameters["min_charge_level"]
    ev_driving_charge_level = ev_battery_parameters["driving_charge_level"]
    ev_charging_switch_level = ev_battery_parameters["charging_switch_level"]
    ev_efficiency = ev_battery_parameters["efficiency"]
    ev_energy_loss = ev_battery_parameters["energy_loss"]
    ev_nominal_power = ev_battery_parameters["nominal_power"]

    lower_bounds = [min_temp_setting, - storage_nominal_power, 0.]
    upper_bounds = [max_temp_setting, storage_nominal_power, ev_nominal_power]
    state_dim = 11
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

    ep_reward_list = []
    max_train_index = len(timestamps_hour) - 25
    train_indexes = np.random.randint(max_train_index, size=(max_train_index,))
    for ep in range(number_of_episodes):
        episode_start_index = train_indexes[ep % len(train_indexes)]  # np.random.randint(max_train_index)
        episode_start_hour = timestamps_hour[episode_start_index]

        pv_generation_list = pv_generation_train[episode_start_index: episode_start_index + 25]
        pv_generation_pred_list = pv_generation_pred_train[episode_start_index: episode_start_index + 25]
        uncontrolled_consumption_list = uncontrolled_consumption_train[episode_start_index: episode_start_index + 25]
        uncontrolled_consumption_pred_list = uncontrolled_consumption_pred_train[
                                             episode_start_index: episode_start_index + 25]
        temp_outside_list = temp_outside_train[episode_start_index: episode_start_index + 25]
        temp_outside_pred_list = temp_outside_pred_train[episode_start_index: episode_start_index + 25]

        hour = episode_start_hour % 24
        pref_temperature = get_preferred_temperature()
        temp_inside = pref_temperature + np.random.uniform(- temp_window, temp_window)
        storage_soc = np.random.uniform(storage_min_charge_level, 100.)
        ev_soc = np.random.uniform(ev_min_charge_level, 100.)
        is_heating_on = bool(np.random.randint(2))

        episodic_reward = 0
        for h in range(24):
            hour = (episode_start_hour + h) % 24
            ev_driving_state = get_ev_driving_state()
            pref_temperature = get_preferred_temperature()
            state = get_state(index=h)
            action = select_action(torch.tensor(state, dtype=torch.float).unsqueeze(0))

            pv_generation = pv_generation_list[h]
            uncontrolled_consumption = uncontrolled_consumption_list[h]
            temp_outside = temp_outside_list[h]

            # Receive state and reward from environment.
            reward, temp_inside, storage_soc, ev_soc, is_heating_on = step(
                action, temp_inside, storage_soc, ev_soc, cycle_timedelta_s
            )
            buffer.rewards.append(reward)
            if h == 23:
                buffer.is_terminals.append(1)
            else:
                buffer.is_terminals.append(0)
            episodic_reward += reward

        if ep % action_std_decay_freq == 0:
            action_std = decay_action_std(action_std)
        if ep % update_epoch == 0:
            update()

        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-100:])
        logging.debug(f"Episode * {ep} * Avg Reward is ==> {avg_reward} " + f"* Std {action_std}")

    example_state = get_state(index=0)
    example_inputs = (torch.FloatTensor(torch.tensor(example_state, dtype=torch.float).unsqueeze(0).to(device)), )
    tmp_stream = BytesIO()
    torch.onnx.export(
        policy_old.actor,
        example_inputs,
        tmp_stream,  # where to save the model (can be a file or file-like object)
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
    )
    bucket = s3.Bucket(s3_parameters["bucket_name"])
    bucket.put_object(Key=s3_parameters["model_filename"], Body=tmp_stream.getvalue())

    return ep_reward_list
