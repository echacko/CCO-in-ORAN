from dataclasses import dataclass
from typing import Any, List, Tuple

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
            output_dim: int = 30,
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim

            self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.layer2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layer3 = torch.nn.Linear(self.hidden_dim, self.output_dim)

        def forward(self, x) -> torch.Tensor:
            '''Forward pass of the actor network'''

            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.tanh(self.layer3(x))

            return x

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
        self.exploration_noise_decay = kwargs.get("exploration_noise_decay", 0.996)
        self.max_iterations = kwargs.get("max_iterations", 30000)
        self.lr = kwargs.get("lr", 0.001)

        # The fixed state. We assume that the state is fixed for all
        # actions in the action space
        self.state = torch.ones(self.num_sectors, dtype=torch.float32).to(self.device)
        self.state_np = self.state.cpu().numpy()

        self.actor = DDPG.Actor(
            input_dim=self.num_sectors,
            output_dim=self.num_sectors*2,
            hidden_dim=kwargs.get("hidden_dim", 64),
        ).to(self.device)

        self.actor_target = DDPG.Actor(
            input_dim=self.num_sectors,
            output_dim=self.num_sectors*2,
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

        # Store experience in the replay buffer
        for _ in range(self.replay_buffer.batch_size):
            configuration = self.get_action(self.state, is_training=True)
            rsrp_powermap, interference_powermap, _ = \
                self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)
            reward = self.problem_formulation.get_objective_value(
                rsrp_powermap, interference_powermap
            )
            self.replay_buffer.store(
                self.state_np,
                np.concatenate(configuration, axis=0),
                reward,
                self.state_np
            )

    def get_action(
        self,
        state: torch.Tensor,
        is_training: bool = True
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''Get the action from the actor'''
        action = self.actor(state).detach().cpu().numpy()

        if is_training:
            action += self.exploration_noise * np.random.randn(self.num_sectors*2)
            action = np.clip(action, -1, 1)
            self.exploration_noise *= self.exploration_noise_decay

        # The first half of the action is the downtilts for each sector
        # Discritize the downtilt to the nearest integer
        downtilt_for_sector = np.round(action[:self.num_sectors])
        downtilt_for_sectors = np.clip(
            downtilt_for_sector,
            self.downtilt_range[0],
            self.downtilt_range[1],
        )

        # The second half of the action is the downtilt for each sector
        # Clip the power to the min and max TX power
        power_for_sectors = np.clip(
            action[self.num_sectors:],
            self.power_range[0],
            self.power_range[1],
        )

        return (downtilt_for_sectors, power_for_sectors)

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
        # Get the action(configuration) from the target actor
        configuration = self.get_action(self.state, is_training=True)

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

        # Add the transition to the replay buffer
        self.replay_buffer.store(
            self.state_np,
            np.concatenate(configuration, axis=0),
            reward,
            self.state_np
        )

        # Train the actor and critic networks
        self.train_actor_and_critic()

        # Update the target networks
        self.update_target_networks()

        return configuration, reward, metrics
