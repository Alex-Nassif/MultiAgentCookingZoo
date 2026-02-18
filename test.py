import sys
import os

# Get the absolute path to three directories above the current file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(BASE_DIR)

sys.path.append(BASE_DIR)


from cooking_zoo.environment.cooking_env import parallel_env
import numpy as np
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import Box, Dict


class CookingEnv:
    def __init__(self, **kwargs):
        self.n_agents = kwargs.get("num_agents", 2)
        self.episode_limit = kwargs.get("episode_limit", 400)
        self.communication_bits = 2  # Number of bits in communication vector

        self.env = parallel_env(
            level=kwargs.get("level", "coop_test"),
            meta_file=kwargs.get("meta_file", "example"),
            num_agents=self.n_agents,
            max_steps=self.episode_limit,
            recipes=kwargs.get("recipes", ["TomatoLettuceSalad", "CarrotBanana"]),
            agent_visualization=kwargs.get("agent_visualization", ["human", "robot"]),
            obs_spaces=kwargs.get("obs_spaces", ["feature_vector"] * self.n_agents),
            end_condition_all_dishes=kwargs.get("end_condition_all_dishes", True),
            action_scheme=kwargs.get("action_scheme", "scheme3"),
            render=kwargs.get("render", True),
            reward_scheme=kwargs.get("reward_scheme", {
                "recipe_reward": 20,
                "max_time_penalty": 0,
                "recipe_penalty": -40,
                "recipe_node_reward": 10,
            }),
        )

        obs = self.env.reset()  # Reset the environment to get initial observations
        if isinstance(obs, tuple):
            obs = obs[0]
        self.last_obs = obs
        self.agents = self.env.agents
        self.last_obs = obs
        self.last_comms = {
            agent: np.zeros(self.communication_bits * (self.n_agents - 1), dtype=np.float32)
            for agent in self.agents
        }

        print("NUM OF AGENTS:", self.n_agents) #returns 2
        print("self.communication_bits:", self.communication_bits) #returns 2
        print("self.obs_shape:", self.env.observation_spaces[self.agents[0]].shape[0]) #returns 278
        print("self.obs_spaces:", self.env.observation_spaces)
        self.observation_spaces = {agent: Box(0, 1, shape=(278 + self.communication_bits * (self.n_agents - 1),)) for agent in self.agents}
        self._obs_shape = self.env.observation_spaces[self.agents[0]].shape[0] + self.communication_bits * (self.n_agents - 1)
        self._n_actions = self.env.action_spaces[self.agents[0]].n
        #self.action_spaces = {agent: self.env.action_spaces[agent] for agent in self.agents}
        self.action_spaces = {agent: MultiDiscrete([self._n_actions*2]) for agent in self.agents}
        print(self.action_spaces, "action_spaces")  # Debugging output
        self.t = 0

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def reset(self):
        self.t = 0
        obs = self.env.reset()  # Reset the environment to get initial observations
        if isinstance(obs, tuple):
            obs = obs[0]
        self.last_obs = obs
        self.last_comms = {
            agent: np.zeros(self.communication_bits * (self.n_agents - 1), dtype=np.float32)
            for agent in self.agents
        }
        return [self._build_obs_player0(), self._build_obs_player1()]

    def step(self, actions_with_comm):
        #print("Actions with commsOG:", actions_with_comm)  # Debugging output
        #print("Actions with commsOG:", actions_with_comm.tolist())  # Debugging output


#        if isinstance(actions_with_comm, np.ndarray) or hasattr(actions_with_comm, "shape"):
#            # If it's a tensor/array, convert to list of tuples with None comms
#            actions_with_comm = [(actions_with_comm.cpu().detach().numpy(), None) for a in actions_with_comm.tolist()]
#        elif not isinstance(actions_with_comm, (list, tuple)):
#            actions_with_comm = [(actions_with_comm.cpu().detach().numpy(), None)]
#        actions, comms = zip(*actions_with_comm)  # actions, communication bits

        comms = [0, 0]

        actions = actions_with_comm.tolist()
        
        # Convert action list to dict
        action_dict = {agent: actions[i] % self._n_actions for i, agent in enumerate(self.agents)}
        #print("Actions with comms:", actions_with_comm)  # Debugging output
        #print("Actions:", action_dict)  # Debugging output
        # Store communications
        self.last_comms = {}
        for i, agent in enumerate(self.agents):
            self.last_comms[agent] = np.zeros(2, dtype=np.float32)  # Ensure comms is a numpy array
            self.last_comms[agent][int(actions[i]//self._n_actions)] = 1  # Normalize comms to [0, 1]
        #print(self.agents)
#        new_comms = {}
#        for i, sender in enumerate(self.agents):
#            print(i, sender, actions[i])
#            msg = np.array(comms[i], dtype=np.float32)
#            if msg.ndim == 0:
#                msg = np.array([msg], dtype=np.float32)
#            for j, receiver in enumerate(self.agents):
#                if i != j:
#                    if receiver not in new_comms:
#                        new_comms[receiver] = []
#                    new_comms[receiver].extend(msg)
#        for agent in self.agents:
#            self.last_comms[agent] = np.array(new_comms[agent], dtype=np.float32)
        for agent, action in action_dict.items():
            if not self.env.action_spaces[agent].contains(action):
                print(f"Invalid action {action} for {agent}")
        # Step the environment
        obs, rewards, terminated, truncated, infos = self.env.step(action_dict)
        dones = {agent: terminated[agent] or truncated[agent] for agent in self.agents}
        #print(obs, "OBS")  # Debugging output
        if isinstance(obs, tuple):
            obs = obs[0]
        self.last_obs = obs
        self.t += 1
        terminated = all(dones.values()) or self.t >= self.episode_limit
        self.env.render()
        print ("Rewards", sum(rewards.values()))
        print("Rewards,", rewards)
        print("Actions:", action_dict)
        return (
            [self._build_obs_player0(), self._build_obs_player1()],  # obs
            sum(rewards.values()),                             # rewards
            [terminated] * self.n_agents,                       # terminated
            [False] * self.n_agents,                            # truncated
            {}#infos                             # infos
        )

    def _build_obs_player0(self):
        base_obs = self.last_obs["player_0"]
        comm = self.last_comms["player_1"]
        #print("Comms:", comm)
        #print(np.concatenate([base_obs, comm], axis=0).shape)  # Debugging shape
        return np.concatenate([base_obs, comm], axis=0)
    
    def _build_obs_player1(self):
        base_obs = self.last_obs["player_1"]
        comm = self.last_comms["player_0"]
        #print("Comms:", comm)
        #print(np.concatenate([base_obs, comm], axis=0).shape)  # Debugging shape
        return np.concatenate([base_obs, comm], axis=0)

    def get_obs(self):
        return [self._build_obs_player0(), self._build_obs_player1()]

    def get_obs_agent(self, agent_id):
        return self._build_obs(self.agents[agent_id])

    def get_obs_size(self):
        return self._obs_shape

    def get_state(self):
        return np.concatenate([self._build_obs_player0(), self._build_obs_player1()], axis=0)

    def get_state_size(self):
        return self.get_state().shape[0]

    def get_avail_actions(self):
        return [list(range(self._n_actions*2)) for _ in range(len(self.agents))]

    def get_avail_agent_actions(self, agent_id):
        return list(range(self._n_actions*2))

    def get_total_actions(self):
        return self._n_actions

    def get_env_info(self):
        return {
            "n_actions": self._n_actions*2,
            "n_agents": self.n_agents,
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "episode_limit": self.episode_limit,
        }

    def close(self):
        self.env.close()

if __name__ == "__main__":
    env = CookingEnv(num_agents=2, episode_limit=400)
    obs = env.reset()
    for i in range(10000):
        actions = {"player_0": np.int64(1), "player_1": np.int64(2)}  # Example actions
        env.env.render()
        env.env.step(actions)
    env.close()
