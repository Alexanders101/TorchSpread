from threading import Thread

import gym
import numpy as np
import torch
import ctypes

from threading import Thread

from typing import Tuple, List

from torch import nn
from torch.multiprocessing import JoinableQueue, Process, Value
from torch.nn import functional as F

from torch_spread import NetworkClient, NetworkManager, SpreadModule
from torch_spread.BufferQueue import BufferRing
from torch_spread.BufferTools import Buffer, raw_buffer_and_size

from argparse import ArgumentParser, Namespace
from scipy import signal

process_type = Process
# process_type = Thread


class DuelingNetwork(SpreadModule):
    """ A simple feed forward neural network for training a q-value on cartpole. """

    def __init__(self, worker: bool, state_shape: Tuple[int], num_actions: int):
        super(DuelingNetwork, self).__init__(worker)
        self.input_shape = int(np.prod(state_shape))

        self.encoder = nn.Sequential(
            nn.Linear(self.input_shape, 16),
            nn.PReLU(16),
            nn.Linear(16, 32),
            nn.PReLU(32),
        )

        self.value_output = nn.Linear(32, 1)
        self.advantage_output = nn.Linear(32, num_actions)

    def forward(self, input_buffer):
        x = self.encoder(input_buffer.view(-1, self.input_shape))

        value = self.value_output(x)
        advantage = self.advantage_output(x)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def q_values(self, states, actions):
        return self.forward(states).gather(1, actions.unsqueeze(1)).squeeze()


class Episode:
    def __init__(self, n_step: int = 1, discount: float = 0.99):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.action_probabilities: List[float] = []
        self.length: int = 0

        self.n_step = n_step
        self.discount = discount

        if n_step > 1:
            self.discount_filter = np.arange(n_step, dtype=np.float32)
            self.discount_filter = discount ** self.discount_filter

    def add(self, state: np.ndarray, action: int, reward: float, action_probability: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_probabilities.append(action_probability)
        self.length += 1

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.action_probabilities.clear()
        self.length = 0

    @property
    def total_reward(self):
        return sum(self.rewards)

    @property
    def observations(self):
        states = torch.as_tensor(self.states, dtype=torch.float32)
        actions = torch.as_tensor(self.actions, dtype=torch.long)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32)
        action_probabilities = torch.as_tensor(self.action_probabilities, dtype=torch.float32)

        # Priorities are calculated when adding to replay buffer
        priorities = torch.zeros(self.length, dtype=torch.float32)

        # Full Monte Carlo discounts
        if self.n_step < 1:
            terminals = torch.ones(self.length, dtype=torch.uint8)
            results = torch.zeros_like(states)
            discount_rewards = signal.lfilter([1], [1, -self.discount], x=rewards[::-1].numpy())
            discount_rewards = torch.from_numpy(discount_rewards[::-1])

        # TD(n) discounts
        else:
            # Compute terminals as a binary mask for states that hit the terminal state during n-step
            terminals = torch.zeros(self.length, dtype=torch.uint8)
            terminals[-self.n_step:] = 1

            # Compute the next-states as the n-offset of the states with zero padding
            results = torch.zeros_like(states)
            if self.length > self.n_step:
                results[:self.length - self.n_step] = states[self.n_step:]

            # Compute the n-step discount returns
            discount_rewards = rewards
            if self.n_step > 1:
                discount_rewards = signal.correlate(rewards.numpy(), self.discount_filter[:self.length], 'full')
                discount_rewards = torch.from_numpy(discount_rewards[-self.length:])

        return {
            "states": states,
            "actions": actions,
            "results": results,
            "rewards": rewards,
            "terminals": terminals,
            "priorities": priorities,
            "discount_rewards": discount_rewards,
            "action_probabilities": action_probabilities
        }


class PrioritizedReplayBuffer(BufferRing):
    def __init__(self, state_shape: Tuple[int], max_size: int, alpha: float = 0.6, beta: float = 0.4):
        """ A ring buffer for storing a prioritized replay buffer. Used for Deep Q Learning.

        Parameters
        ----------
        state_shape: tuple
            The numpy shape of a single state
        max_size: int
            Maximum number of unique samples to hold in this buffer.
        alpha: float
            Prioritized Experience Replay alpha parameter
        beta: float
            Prioritized Experience Replay beta parameter
        """
        buffer_shapes = {
            "states": state_shape,
            "results": state_shape,
            "actions": tuple(),
            "rewards": tuple(),
            "terminals": tuple(),
            "priorities": tuple(),
            "discount_rewards": tuple(),
            "action_probabilities": tuple()
        }

        buffer_types = {
            "states": torch.float32,
            "results": torch.float32,
            "actions": torch.long,
            "rewards": torch.float32,
            "terminals": torch.uint8,
            "priorities": torch.float32,
            "discount_rewards": torch.float32,
            "action_probabilities": torch.float32
        }

        super(PrioritizedReplayBuffer, self).__init__(buffer_shapes, buffer_types, max_size)

        self.alpha = alpha
        self.beta = beta

        self.max_priority = Value(ctypes.c_float, lock=False)
        self.max_priority.value = 1.0

    @property
    def priorities(self):
        current_size = self.size
        return self.buffer[:current_size]('priorities').numpy()

    def update_priority(self, idx: np.ndarray, delta: torch.Tensor):
        self.buffer('priorities')[idx] = torch.abs(delta.detach().cpu())

    def update_max_priority(self):
        self.max_priority.value = float(np.max(self.priorities, initial=1))

    def put(self, buffer, size: int = None):
        buffer, size = raw_buffer_and_size(buffer, size)

        # Compute the priority for an incoming sample
        buffer['priorities'][:] = self.max_priority.value

        # Put it into the buffer
        super(PrioritizedReplayBuffer, self).put(buffer, size)

    def add_episode(self, episode: Episode):
        self.put(episode.observations, episode.length)

    def sample(self, num_samples: int = 32):
        current_size = self.size

        assert num_samples <= current_size, f"Buffer is not large enough to provide {num_samples} samples"

        # Calculate probabilities
        P = self.buffer[:current_size]('priorities').numpy() ** self.alpha
        P /= P.sum()

        # Calculate IS weights
        w = (current_size * P) ** -self.beta

        # Normalize weights
        max_w = np.max(w)
        if max_w > 1e-5:
            w /= max_w

        # Sample based on probabilities
        idx = np.random.choice(current_size, size=num_samples, replace=True, p=P)
        idx = np.sort(idx)

        # Get the sampled batch
        batch = self.buffer[idx]
        weights = torch.from_numpy(w[idx])

        return idx, batch, weights


class EpsilonGreedyClient(NetworkClient):
    def __init__(self, config: dict, batch_size: int, epsilon: float, epsilon_update=None):
        """ An extension to the regular NetworkClient that provides Q-learning policy functions.

        Parameters
        ----------
        config: dict
            Client configuration from the network manager.
        batch_size: int
            Maximum number of states you're planning on predicting at once.
        epsilon: float
            Probability of performing a random action
        epsilon_update: float -> float, optional
            A function for updating the epsilon value every time you sample
        """
        super().__init__(config, batch_size)

        self.epsilon = epsilon
        self.epsilon_update = epsilon_update

    def update_epsilon(self):
        """ Perform a single call to the epsilon update function. """
        if self.epsilon_update is not None:
            self.epsilon = self.epsilon_update(self.epsilon)

    def sample_actions(self, states):
        """ Sample many actions at once (for vectorized environment). """
        q_values = self.predict(states)
        num_states, num_actions = q_values.shape
        greedy_actions = q_values.max(dim=1)[1]
        greedy_actions = greedy_actions.numpy()

        random_actions = np.random.randint(low=0, high=num_actions, size=num_states, dtype=greedy_actions.dtype)

        epsilons = np.random.rand(num_states)
        epsilons = (epsilons < self.epsilon).astype(np.int64)

        return np.choose(epsilons, (greedy_actions, random_actions))

    def sample_action(self, state, num_actions):
        """ Sample a single action. """
        q_values = self.predict(torch.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)))
        action = greedy_action = q_values.max(dim=1).indices[0].item()

        if np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=num_actions)

        if action == greedy_action:
            return action, (1 - self.epsilon) + self.epsilon / num_actions
        else:
            return action, self.epsilon / num_actions


class EpisodeCollector(process_type):
    def __init__(self,
                 parameters: Namespace,
                 client_config: dict,
                 replay_buffer: PrioritizedReplayBuffer,
                 request_queue: JoinableQueue):
        super(EpisodeCollector, self).__init__()

        self.epsilon_initial: float = parameters.epsilon_initial
        self.epsilon_final: float = parameters.epsilon_final
        self.epsilon_episodes: int = parameters.epsilon_decay_episodes // parameters.num_collectors
        self.environment_name: str = parameters.environment_name
        self.discount: float = parameters.discount
        self.n_step: int = parameters.n_step

        self.client_config = client_config
        self.request_queue = request_queue
        self.replay_buffer = replay_buffer

        self.average_reward = Value(ctypes.c_float, lock=False)
        self.average_length = Value(ctypes.c_float, lock=False)
        self.total_episodes = Value(ctypes.c_int64, lock=False)

    @staticmethod
    def create(parameters: Namespace, client_config: dict, replay_buffer: PrioritizedReplayBuffer):
        request_queue: JoinableQueue = JoinableQueue(parameters.num_collectors)

        collectors: List[EpisodeCollector] = []
        for _ in range(parameters.num_collectors):
            collector = EpisodeCollector(parameters, client_config, replay_buffer, request_queue)
            collector.start()
            collectors.append(collector)

        return request_queue, collectors

    @staticmethod
    def kill_collectors(request_queue: JoinableQueue, collectors: List["EpisodeCollector"]):
        for _ in range(len(collectors)):
            request_queue.put(-1)

        for collector in collectors:
            collector.join(timeout=2)

    @staticmethod
    def collect(request_queue: JoinableQueue, collectors: List["EpisodeCollector"], total_states: int):
        num_workers = len(collectors)
        states_per_worker = total_states // num_workers

        for _ in range(num_workers):
            request_queue.put(states_per_worker)

        request_queue.join()

        average_reward = sum(collector.average_reward.value for collector in collectors) / num_workers
        total_episodes = sum(collector.total_episodes.value for collector in collectors)
        return total_episodes, average_reward

    def run(self):
        torch.manual_seed(self.pid)

        epsilon_factor = (self.epsilon_final / self.epsilon_initial) ** (1 / self.epsilon_episodes)

        def epsilon_update(epsilon):
            return max(epsilon * epsilon_factor, self.epsilon_final)

        env = gym.make(self.environment_name)
        num_actions = env.action_space.n

        state = env.reset()
        episode = Episode(self.n_step, self.discount)

        with EpsilonGreedyClient(self.client_config, 1, self.epsilon_initial, epsilon_update) as client:
            while True:
                num_states = self.request_queue.get()
                self.average_reward.value = 0
                self.average_length.value = 0
                num_episodes = 0

                # Kill Switch
                if num_states < 0:
                    return

                for sample in range(num_states):
                    # Sample an action from our network
                    action, probability = client.sample_action(state, num_actions)

                    # Perform action and get the results from the environment
                    next_state, reward, terminal, _ = env.step(action)

                    # Add all of the observations to the current episode
                    episode.add(state, action, reward, probability)

                    # Set variables for next step
                    state = next_state

                    # If the environment has terminated, then we add our observations to the replay buffer and reset.
                    if terminal:
                        self.replay_buffer.add_episode(episode)

                        num_episodes += 1
                        self.average_reward.value += (episode.total_reward - self.average_reward.value) / num_episodes
                        self.average_length.value += (episode.length - self.average_length.value) / num_episodes
                        self.total_episodes.value += 1

                        episode.clear()
                        state = env.reset()
                        client.update_epsilon()

                self.request_queue.task_done()


def main():
    params = Namespace()
    params.environment_name = "CartPole-v0"
    params.num_collectors = 32

    params.replay_buffer_size = 200_000
    params.total_episodes = 30_000
    params.warmup_episodes = 1_000

    params.states_per_epoch = 20_000
    params.train_iterations = 40
    params.train_batch_size = 1024
    params.target_network_update_epochs = 1

    params.epsilon_initial = 1.0
    params.epsilon_final = 0.001
    params.epsilon_decay_episodes = 10_000

    params.discount = 0.99
    params.n_step = 20

    env = gym.make(params.environment_name)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    manager = NetworkManager(input_shape=state_shape,
                             input_type=torch.float32,
                             output_shape=num_actions,
                             output_type=torch.float32,
                             batch_size=params.num_collectors,
                             network_class=DuelingNetwork,
                             network_args=[state_shape, num_actions],
                             placement={'cuda:0': 1})
    with manager:
        optimizer = torch.optim.Adam(manager.training_parameters, lr=0.005)

        # Create the dynamically changing target network
        device = manager.training_placement
        target_network = DuelingNetwork(False, state_shape, num_actions).to(device)

        # Prioritized replay buffer for collecting states
        replay_buffer = PrioritizedReplayBuffer(manager.input_shape, params.replay_buffer_size)

        # Create the worker processes for collecting observations
        request_queue, collectors = EpisodeCollector.create(params, manager.client_config, replay_buffer)

        # Loop variables
        num_episodes = 0
        num_states = 0
        iteration = 0

        while num_episodes < params.total_episodes:
            num_episodes, average_reward = EpisodeCollector.collect(request_queue, collectors, params.states_per_epoch)
            num_states += params.states_per_epoch
            iteration += 1

            print(f"{iteration}")
            print("=" * 60)
            print(f"Average Reward: {average_reward}")
            print(f"Number of Episodes: {num_episodes}")
            print(f"Number of States: {num_states}")
            print()

            if num_episodes < params.warmup_episodes:
                continue

            with manager.training_network as policy_network:
                for train_iteration in range(params.train_iterations):
                    sample_index, sample, weights = replay_buffer.sample(params.train_batch_size)

                    sample = sample.to(device)
                    weights = weights.to(device)

                    # Generate n-step double dqn target
                    with torch.no_grad():
                        next_states = sample("results")
                        discount = params.discount ** params.n_step
                        policy_actions = policy_network(next_states).max(dim=1).indices
                        q_max = target_network.q_values(next_states, policy_actions)
                        targets = sample("discount_rewards") + q_max * discount * (1 - sample("terminals"))

                    # Generate current estimates for the q-value
                    states, actions = sample("states"), sample("actions")
                    all_q_values = policy_network(states)
                    q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze()

                    # Compute bellman loss term
                    delta = targets - q_values
                    loss = delta * delta * weights
                    loss = loss.sum()

                    # Perform gradient descent step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    replay_buffer.update_priority(sample_index, delta)

            if iteration % params.target_network_update_epochs == 0:
                target_network.load_state_dict(manager.state_dict)

            # Update the incoming priority once before sampling
            replay_buffer.update_max_priority()

        EpisodeCollector.kill_collectors(request_queue, collectors)


if __name__ == '__main__':
    main()
