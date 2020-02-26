import numpy as np 
import copy

class Env(object):
    def reset(self, **kwags):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

class CarRental(Env):
    observation_space = [[i, j] for i in range(21) for j in range(21)]
    action_space = list(range(-5, 6))

    def reset(self):
        self.observation =[10, 10]
    
    def step(self, action):
        reward = 0
        assert action in self.action_space
        if action > 0:
            real_action = min(action, self.observation[0])
        elif action < 0:
            real_action = max(action, -self.observation[1])
        self.observation[0] -= real_action
        self.observation[1] += real_action

        place1_render = min(np.random.poisson(3), self.observation[0])
        self.observation[0] -= place1_render
        reward += 10 * place1_render

        place1_return = np.random.poisson(3)
        self.observation[0] += min(place1_return, 20 - self.observation[0])

        place2_render = min(np.random.poisson(4), self.observation[1])
        self.observation[1] -= place2_render
        reward += 10 * place2_render

        place2_return = np.random.poisson(2)
        self.observation[1] += min(place2_return, 20 - self.observation[1])

        return self.observation, reward

    @staticmethod
    def p(observation, action, next_observation, reward):
        if action > 0:
            real_action = min(action, observation[0])
        elif action < 0:
            real_action = max(action, -observation[1])
        else:
            real_action = action

        if (reward + real                                                                                                                                                                                                                                                                                                                  _action * 2) % 10 != 0:
            return 0
        else:
            total_render = int((reward + real_action * 2) / 10)

        observation_ = copy.deepcopy(observation)
        observation_[0] -= real_action
        observation_[1] += real_action

        poisson_prob = lambda lambda1, n: lambda1**n / np.math.factorial(n) * np.exp(-lambda1)
        prob = 0
    
        gap1 = observation_[0] - observation[0]
        gap2 = observation_[1] - observation_[1]

        for place1_render in range(0, total_render+1):
            place2_render = total_render - place1_render
            place1_return = gap1 + place1_render
            place2_return = gap2 + place2_render
            prob += poisson_prob(3, place1_render) * poisson_prob(4, place2_render) * poisson_prob(3, place1_return) * poisson_prob(2, place2_return)

        return prob

def policy_evaluate(env, policy, gamma, delta_threshold):
    values = np.zeros_like(policy)
    while True:
        delta = 0
        for ob in env.observation_space:
            v = values[ob[0], ob[1]]
            for ob_ in env.observation_space:
                for re in range(-10, 401, 2):
                    v += env.p(ob, policy[ob[0], ob[1]], ob_, re) * (re + gamma * values[ob_[0], ob_[1]])
                    delta = max(delta, abs(v - values[ob[0], ob[1]]))
            values[ob[0], ob[1]] = v
            print(ob[0], ob[1], v)
        if delta < delta_threshold:
            break
    return values

def policy_improve(env, policy, values, gamma):
    policy_stable = True
    for ob in env.observation_space:
        for ac in env.action_space:
            if ac == policy[ob[0], ob[1]]:
                continue
            v = 0
            for ob_ in env.observation_space:
                for re in range(-10, 401, 2):
                    v += env.p(ob, policy[ob[0], ob[1]], ob_, re) * (re + gamma * values[ob[0], ob[1]])
            if v > values[ob[0], ob[1]]:
                policy[ob[0], ob[1]] = ac
                values[ob[0], ob[1]] = v
                policy_stable = False
    return policy_stable

def policy_iterate(env, policy, gamma=0.9, delta_threshold=1):
    while True:
        values = policy_evaluate(env, policy, gamma, delta_threshold)
        policy_stable = policy_improve(env, policy, values, gamma)
        break
        if policy_stable:
            break 

if __name__ == "__main__":
    env = CarRental()
    initial_policy = np.zeros((21, 21))
    policy_iterate(env, initial_policy)