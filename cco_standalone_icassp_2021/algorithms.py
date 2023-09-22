from dataclasses import dataclass
from typing import Tuple

import logging
import numpy as np
import torch
from problem_formulation import (
    CCORasterBlanketFormulation,
)
from simulated_rsrp import SimulatedRSRP


class CCOAlgorithm:
    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        **kwargs,
    ):
        self.simulated_rsrp = simulated_rsrp
        self.problem_formulation = problem_formulation

        # Get configuration range for downtilts and powers
        (
            self.downtilt_range,
            self.power_range,
        ) = self.simulated_rsrp.get_configuration_range()

        # Get the number of total sectors
        _, self.num_sectors = self.simulated_rsrp.get_configuration_shape()

    def step(self) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        float,
        Tuple[float, float],
    ]:
        """Stub for one step of the algorithm.

            Return tuple::
            1. configuration : nested tuple of optimal tilts and optimal powers
            2. reward : weigted combination of metrics
            3. metrics : tuple of dual objectives : under-coverage and over-coverage
        """
        return [None, None], None, [0.0, 0.0]


class RandomSelection(CCOAlgorithm):
    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        **kwargs,
    ):
        super().__init__(
            simulated_rsrp=simulated_rsrp, problem_formulation=problem_formulation
        )

    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        # Random select powers and downtilts

        downtilts_for_sectors = np.random.uniform(
            self.downtilt_range[0], self.downtilt_range[1], self.num_sectors
        )
        power_for_sectors = np.random.uniform(
            self.power_range[0], self.power_range[1], self.num_sectors
        )

        # power_for_sectors = [max_tx_power_dBm] * num_sectors
        configuration = (downtilts_for_sectors, power_for_sectors)
        # Get the rsrp and interferences powermap
        (
            rsrp_powermap,
            interference_powermap,
            _,
        ) = self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)

        # According to the problem formulation, calculate the reward
        reward = self.problem_formulation.get_objective_value(
            rsrp_powermap, interference_powermap
        )

        # Get the metrics
        metrics = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_powermap, interference_powermap
        )
        return configuration, reward, metrics

class DDPG(CCOAlgorithm):
    class ReplayBuffer:
        @dataclass
        class Experience:
            '''Experience class for storing the experience in the replay buffer'''
            state: np.ndarray
            action: np.ndarray
            reward: float
            next_state: np.ndarray

        def __init__(
            self,
            buffer_capacity: int = 10000,
            batch_size: int = 64,
        ):
            self.buffer_capacity = buffer_capacity
            self.batch_size = batch_size
            self.buffer_counter = 0

            self.replay_buffer = []

        def store(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
        ):
            '''Store the experience in the replay buffer'''
            experience = DDPG.ReplayBuffer.Experience(state, action, reward, next_state)

            if self.buffer_counter < self.buffer_capacity:
                self.replay_buffer.append(experience)
                self.buffer_counter += 1
            else:
                self.replay_buffer.pop(0)
                self.replay_buffer.append(experience)

        def sample(
            self
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            '''Sample a batch of experiences from the replay buffer'''
            try:
                assert self.buffer_counter >= self.batch_size
            except AssertionError:
                logging.error("Buffer counter is less than batch size")
                raise

            batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)

            states = np.array([experience.state for experience in batch])
            actions = np.array([experience.action for experience in batch])
            rewards = np.array([experience.reward for experience in batch])
            next_states = np.array([experience.next_state for experience in batch])

            return states, actions, rewards, next_states

    class Actor(torch.nn.Module):
        def __init__(
            self,
            input_dim: int = 15,
            output_dim: Tuple[int, int] = (15, 11),
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim

            # The input and hidden layers are common.
            self.common = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.ReLU(),
            )

            # The output layers
            # Torch output layer with sigmoid activation for power
            self.power_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.output_dim[0]),
                torch.nn.Sigmoid(),
            )

            # Torch output layer with log softmax activation for downtilts
            self.tilt_layer = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_dim, self.output_dim[1]),
                        torch.nn.LogSoftmax(dim=1),
                    )
                    for _ in range(self.output_dim[0])
                ]
            )

        def forward(self, x) -> torch.Tensor:
            '''Forward pass of the actor network'''

            x = self.common(x)
            power = self.power_layer(x)
            tilt_probs = [layer(x) for layer in self.tilt_layer]
            tilt = [torch.argmax(tilt_prob, dim=-1) for tilt_prob in tilt_probs]
            tilt = torch.stack(tilt, dim=1)

            action = torch.cat([tilt, power], dim=1)

            return action

    class Critic(torch.nn.Module):
        def __init__(
            self,
            input_state_dim: int = 15,
            input_action_dim: int = 30,
            output_dim: int = 1,
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.input_state_dim = input_state_dim
            self.input_action_dim = input_action_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim

            self.layer1 = torch.nn.Linear(self.input_state_dim, self.hidden_dim)
            self.layer2 = torch.nn.Linear(self.hidden_dim+self.input_action_dim, self.hidden_dim)
            self.layer3 = torch.nn.Linear(self.hidden_dim, self.output_dim)

        def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:
            '''Forward pass of the critic network'''

            x = torch.relu(self.layer1(state))
            x = torch.cat([x, action], dim=1)
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)

            return x

    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        device: torch.device,
        **kwargs
    ):

        super().__init__(
            simulated_rsrp=simulated_rsrp, problem_formulation=problem_formulation
        )

        self.device = device
        self.gamma = kwargs.get("gamma", 0.99)
        self.target_tau = kwargs.get("target_tau", 0.005)
        self.exploration_noise = kwargs.get("exploration_noise", 1)
        self.exploration_noise_decay = kwargs.get("exploration_noise_decay", (1-4e-5))
        self.lr = kwargs.get("lr", 0.001)

        # The fixed state. We assume that the state is fixed for all
        # actions in the action space
        self.state = torch.ones(self.num_sectors, dtype=torch.float32).to(self.device)
        self.state_np = self.state.cpu().numpy()

        # Number of available antenna downtilts
        self.num_tilts = len(self.simulated_rsrp.downtilts_keys)

        self.actor = DDPG.Actor(
            input_dim=self.num_sectors,
            output_dim=(self.num_sectors, self.num_tilts),
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)

        self.actor_target = DDPG.Actor(
            input_dim=self.num_sectors,
            output_dim=(self.num_sectors, self.num_tilts),
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)

        self.critic = DDPG.Critic(
            input_state_dim=self.num_sectors,
            input_action_dim=self.num_sectors*2,
            output_dim=1,
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)

        self.critic_target = DDPG.Critic(
            input_state_dim=self.num_sectors,
            input_action_dim=self.num_sectors*2,
            output_dim=1,
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = DDPG.ReplayBuffer(
            buffer_capacity=kwargs.get("buffer_capacity", 10000),
            batch_size=kwargs.get("batch_size", 64),
        )

        # Store enough experiences in the replay buffer before training
        for _ in range(self.replay_buffer.batch_size):
            action = self.get_action(self.state, is_training=True)
            configuration = self.get_configuration(action)
            rsrp_powermap, interference_powermap, _ = \
                self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)
            reward = self.problem_formulation.get_objective_value(
                rsrp_powermap, interference_powermap
            )
            self.replay_buffer.store(
                self.state_np,
                np.concatenate(action, axis=0),
                reward,
                self.state_np
            )

    def get_action(
        self,
        state: torch.Tensor,
        is_training: bool = True,
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''Get the action from the actor network'''

        # Get the actions(powers and downtilts) from the actor network
        state = state.unsqueeze(0)
        action = self.actor(state).cpu().detach().numpy()
        downtilt_for_sectors = action[0, :self.num_sectors]
        power_for_sectors = action[0, self.num_sectors:]

        # Add exploration noise to the actions
        if is_training:
            power_for_sectors += self.exploration_noise * np.random.randn(*power_for_sectors.shape)
            # For downtilts, the actions are discrete.
            # Explore with probability exploration_noise
            if np.random.rand() < self.exploration_noise:
                downtilt_for_sectors = np.random.randint(
                    int(self.downtilt_range[0]),
                    int(self.downtilt_range[1]) + 1,
                    self.num_sectors,
                )

            self.exploration_noise *= self.exploration_noise_decay

        return (downtilt_for_sectors, power_for_sectors)

    def get_configuration(
        self,
        action: Tuple[np.ndarray, np.ndarray],
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''Rescale the predicted power and make the configuration'''

        # Rescale the power to min and max power
        power_for_sectors = action[1] * (self.power_range[1] - self.power_range[0])
        power_for_sectors += self.power_range[0]

        # Clip the power to the min and max TX power
        power_for_sectors = np.clip(
            power_for_sectors,
            self.power_range[0],
            self.power_range[1],
        )

        return (action[0], power_for_sectors)

    def get_reward_and_metrics(
        self,
        configuration: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[float, Tuple[float, float]]:
        '''Get the reward and metrics for the given configuration'''

        # Get the rsrp and interferences powermap
        rsrp_powermap, interference_powermap, _ = \
            self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)

        # According to the problem formulation, calculate the reward
        reward = self.problem_formulation.get_objective_value(
            rsrp_powermap, interference_powermap
        )

        # Get the metrics
        metrics = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_powermap, interference_powermap
        )

        return reward, metrics

    def train_actor_and_critic(
        self,
    ):
        '''Train the actor and critic networks on one batch of data.'''

        # Get the batch of data from the replay buffer
        states, actions, rewards, next_states = self.replay_buffer.sample()

        # Convert the numpy arrays to torch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        # Reshape rewards to (num_batch, 1)
        rewards = rewards.unsqueeze(1)

        # Get the next actions from the target actor
        next_actions = self.actor_target(next_states)

        # Get the next Q values from the target critic
        next_Q = self.critic_target(next_states, next_actions)

        # Calculate the target Q values
        target_Q = rewards + self.gamma * next_Q

        # Get the current Q values from the critic
        current_Q = self.critic(states, actions)

        # Calculate the critic loss
        critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate the actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_networks(self):
        '''Update the target networks'''

        # Update the actor target network
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1 - self.target_tau) * target_param.data
            )

        # Update the critic target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1 - self.target_tau) * target_param.data
            )

    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        # Get the action from the target actor
        action = self.get_action(self.state, is_training=True)

        # Get the configuration from the action
        configuration = self.get_configuration(action)

        # Get the reward
        reward, metrics = self.get_reward_and_metrics(configuration)

        # Add the transition to the replay buffer
        self.replay_buffer.store(
            self.state_np,
            np.concatenate(action, axis=0),
            reward,
            self.state_np
        )

        # Train the actor and critic networks
        self.train_actor_and_critic()

        # Update the target networks
        self.update_target_networks()

        return configuration, reward, metrics

class DDQN(CCOAlgorithm):
    class ReplayBuffer:
        @dataclass
        class Experience:
            '''Experience class for storing the experience in the replay buffer'''
            state: np.ndarray
            action: np.ndarray
            reward: float
            next_state: np.ndarray

        def __init__(
            self,
            buffer_capacity: int = 10000,
            batch_size: int = 64,
        ):
            self.buffer_capacity = buffer_capacity
            self.batch_size = batch_size
            self.buffer_counter = 0

            self.replay_buffer = []

        def store(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
        ):
            '''Store the experience in the replay buffer'''
            experience = DDQN.ReplayBuffer.Experience(state, action, reward, next_state)

            if self.buffer_counter < self.buffer_capacity:
                self.replay_buffer.append(experience)
                self.buffer_counter += 1
            else:
                self.replay_buffer.pop(0)
                self.replay_buffer.append(experience)

        def sample(
            self
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            '''Sample a batch of experiences from the replay buffer'''
            try:
                assert self.buffer_counter >= self.batch_size
            except AssertionError:
                logging.error("Buffer counter is less than batch size")
                raise

            batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)

            states = np.array([experience.state for experience in batch])
            actions = np.array([experience.action for experience in batch])
            rewards = np.array([experience.reward for experience in batch])
            next_states = np.array([experience.next_state for experience in batch])

            return states, actions, rewards, next_states

    class DQN(torch.nn.Module):
        def __init__(
            self,
            input_dim: int = 15,
            output_dim: Tuple[int, int, int] = (15, 11, 10),
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim

            self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.layer2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layer3 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.hidden_dim, self.output_dim[1]*self.output_dim[2])
                    for _ in range(self.output_dim[0])
                ]
            )

        def forward(self, x) -> torch.Tensor:
            '''Forward pass of the actor network'''

            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            x = torch.relu(x)

            q_values = [layer(x) for layer in self.layer3]
            q_values = torch.stack(q_values, dim=1)

            return q_values

    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        device: torch.device,
        **kwargs
    ):

        super().__init__(
            simulated_rsrp=simulated_rsrp, problem_formulation=problem_formulation
        )

        self.device = device
        self.gamma = kwargs.get("gamma", 0.99)
        self.target_tau = kwargs.get("target_tau", 0.005)
        self.exploration_noise = kwargs.get("exploration_noise", 1)
        self.exploration_noise_decay = kwargs.get("exploration_noise_decay", (1-4e-5))
        self.lr = kwargs.get("lr", 0.001)

        # The fixed state. We assume that the state is fixed for all
        # actions in the action space
        self.state = torch.ones(self.num_sectors, dtype=torch.float32).to(self.device)
        self.state_np = self.state.cpu().numpy()

        # Number of available antenna downtilts
        self.num_tilts = len(self.simulated_rsrp.downtilts_keys)
        # Number and Discretized power values
        self.num_powers = 200
        self.power_values = np.linspace(
            self.power_range[0], self.power_range[1], self.num_powers
        )

        # Iterate over all the powers and tilts to get the action space
        self.action_space = []
        for power in self.power_values:
            for tilt in self.simulated_rsrp.downtilts_keys:
                self.action_space.append((tilt, power))

        # DQN
        self.dqn = DDQN.DQN(
            input_dim=self.num_sectors,
            output_dim=(self.num_sectors, self.num_tilts, self.num_powers),
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)

        self.dqn_target = DDQN.DQN(
            input_dim=self.num_sectors,
            output_dim=(self.num_sectors, self.num_tilts, self.num_powers),
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = DDQN.ReplayBuffer(
            buffer_capacity=kwargs.get("buffer_capacity", 10000),
            batch_size=kwargs.get("batch_size", 64),
        )

        # Store enough experiences in the replay buffer before training
        for _ in range(self.replay_buffer.batch_size):
            action = self.get_action(self.state, is_training=True)
            configuration = self.get_configuration(action)
            rsrp_powermap, interference_powermap, _ = \
                self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)
            reward = self.problem_formulation.get_objective_value(
                rsrp_powermap, interference_powermap
            )
            self.replay_buffer.store(
                self.state_np,
                action,
                reward,
                self.state_np
            )

    def get_action(
        self,
        state: torch.Tensor,
        is_training: bool = True,
        ) -> np.ndarray:
        '''Get the Q values from main DQN and convert them to actions.'''

        # Get the Q values for all the actions from the DQN
        state = state.unsqueeze(0)
        q_values = self.dqn(state).squeeze(0).cpu().detach().numpy()

        # Select the action with the maximum Q value
        action = np.argmax(q_values, axis=1)

        # Add exploration noise to the actions
        if True: #is_training:
            # Explore with probability exploration_noise
            if np.random.rand() < self.exploration_noise:
                action = np.random.randint(
                    0,
                    self.num_tilts*self.num_powers,
                    (self.num_sectors,),
                )

            self.exploration_noise *= self.exploration_noise_decay

        return action

    def get_configuration(
        self,
        action: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''Get the downtilts and power values from the action indices.'''

        # Get the downtilts and power values for the action indices from
        # the action space.
        downtilt_for_sectors, power_for_sectors = [], []
        for action_idx in action:
            downtilt, power = self.action_space[action_idx]
            downtilt_for_sectors.append(downtilt)
            power_for_sectors.append(power)

        return np.array(downtilt_for_sectors), np.array(power_for_sectors)

    def get_reward_and_metrics(
        self,
        configuration: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[float, Tuple[float, float]]:
        '''Get the reward and metrics for the given configuration'''

        # Get the rsrp and interferences powermap
        rsrp_powermap, interference_powermap, _ = \
            self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)

        # According to the problem formulation, calculate the reward
        reward = self.problem_formulation.get_objective_value(
            rsrp_powermap, interference_powermap
        )

        # Get the metrics
        metrics = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_powermap, interference_powermap
        )

        return reward, metrics

    def train_dqn(
        self,
    ):
        '''Train the DQN networks on one batch of data.'''

        # Get the batch of data from the replay buffer
        states, actions, rewards, next_states = self.replay_buffer.sample()

        # Convert the numpy arrays to torch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        # Reshape rewards to (num_batch, 1)
        rewards = rewards.unsqueeze(1)

        # Get the Q values from the target DQN and get the actions
        next_Q = self.dqn_target(next_states)

        # Calculate the target Q values
        target_Q = rewards + self.gamma * torch.max(next_Q, dim=2)[0]

        # Get the current Q values from the DQN
        current_Q = self.dqn(states)

        # Get the current Q values for the actions taken
        current_Q = current_Q.gather(2, actions.unsqueeze(2)).squeeze(2)

        # Calculate the DQN loss
        dqn_loss = torch.nn.functional.mse_loss(current_Q, target_Q)

        # Update the DQN network
        self.dqn_optimizer.zero_grad()
        dqn_loss.backward()
        self.dqn_optimizer.step()

    def update_target_networks(self):
        '''Update the target networks'''

        # Update the target DQN
        for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1 - self.target_tau) * target_param.data
            )

    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        # Get the action from the target actor
        action = self.get_action(self.state, is_training=True)

        # Get the configuration from the action
        configuration = self.get_configuration(action)

        # Get the reward
        reward, metrics = self.get_reward_and_metrics(configuration)

        # Add the transition to the replay buffer
        self.replay_buffer.store(
            self.state_np,
            action,
            reward,
            self.state_np
        )

        # Train the DQN networks
        self.train_dqn()

        # Update the target networks
        self.update_target_networks()

        return configuration, reward, metrics
