import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

stationary = True


class Bandit():
    def __init__(self, arm_count):
        """
        Multi-armed bandit with rewards 1 or 0.

        At initialization, multiple arms are created. The probability of each arm
        returning reward 1 if pulled is sampled from Bernouilli(p), where p randomly
        chosen from Uniform(0,1) at initialization
        """
        self.arm_count = arm_count
        self.generate_thetas()
        self.timestep = 0
        global stationary
        self.stationary = stationary

    def generate_thetas(self):
        self.thetas = np.random.uniform(0, 1, self.arm_count)

    def get_reward_regret(self, arm):
        """ Returns random reward for arm action. Assumes actions are 0-indexed
        Args:
          arm is an int
        """
        self.timestep += 1
        if (self.stationary == False) and (self.timestep % 100 == 0):
            self.generate_thetas()
        # Simulate bernouilli sampling
        sim = np.random.uniform(0, 1, self.arm_count)
        rewards = (sim < self.thetas).astype(int)
        reward = rewards[arm]
        regret = self.thetas.max() - self.thetas[arm]

        return reward, regret


class BetaAlgo():
    """
    The algos try to learn which Bandit arm is the best to maximize reward.

    It does this by modelling the distribution of the Bandit arms with a Beta,
    assuming the true probability of success of an arm is Bernouilli distributed.
    """

    def __init__(self, bandit):
        """
        Args:
          bandit: the bandit class the algo is trying to model
        """
        self.bandit = bandit
        self.arm_count = bandit.arm_count
        self.alpha = np.ones(self.arm_count)
        self.beta = np.ones(self.arm_count)
        self.mu = np.ones(self.arm_count)
        self.sigma = np.ones(self.arm_count)

    def get_reward_regret(self, arm):
        reward, regret = self.bandit.get_reward_regret(arm)
        self._update_params(arm, reward)
        return reward, regret

    def _update_params(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        self.mu = self.alpha / (self.alpha + self.beta)
        self.sigma = 1 / (self.alpha + self.beta)


class BernThompson2(BetaAlgo):
    def __init__(self, bandit):
        super().__init__(bandit)

    @staticmethod
    def name():
        return 'thompson with normal distribution'

    def get_action(self):
        """ Bernouilli parameters are sampled from the normal"""
        theta = np.random.normal(self.mu, self.sigma)
        return theta.argmax()


class BernGreedy(BetaAlgo):
    def __init__(self, bandit):
        super().__init__(bandit)

    @staticmethod
    def name():
        return 'beta-greedy'

    def get_action(self):
        """ Bernouilli parameters are the expected values of the beta"""
        theta = self.alpha / (self.alpha + self.beta)
        return theta.argmax()


class BernThompson(BetaAlgo):
    def __init__(self, bandit):
        super().__init__(bandit)

    @staticmethod
    def name():
        return 'thompson'

    def get_action(self):
        """ Bernouilli parameters are sampled from the beta"""
        theta = np.random.beta(self.alpha, self.beta)
        return theta.argmax()


epsilon = 0.1


class EpsilonGreedy():
    """
    Epsilon Greedy with incremental update.
    Based on Sutton and Barto pseudo-code, page. 24
    """

    def __init__(self, bandit):
        global epsilon
        self.epsilon = epsilon
        self.bandit = bandit
        self.arm_count = bandit.arm_count
        self.Q = np.zeros(self.arm_count)  # q-value of actions
        self.N = np.zeros(self.arm_count)  # action count

    @staticmethod
    def name():
        return 'epsilon-greedy'

    def get_action(self):
        if np.random.uniform(0, 1) > self.epsilon:
            action = self.Q.argmax()
        else:
            action = np.random.randint(0, self.arm_count)
        return action

    def get_reward_regret(self, arm):
        reward, regret = self.bandit.get_reward_regret(arm)
        self._update_params(arm, reward)
        return reward, regret

    def _update_params(self, arm, reward):
        self.N[arm] += 1  # increment action count
        self.Q[arm] += 1 / self.N[arm] * (reward - self.Q[arm])  # inc. update rule


ucb_c = 2


class UCB():
    """
    Epsilon Greedy with incremental update.
    Based on Sutton and Barto pseudo-code, page. 24
    """

    def __init__(self, bandit):
        global ucb_c
        self.ucb_c = ucb_c
        self.bandit = bandit
        self.arm_count = bandit.arm_count
        self.Q = np.zeros(self.arm_count)  # q-value of actions
        self.N = np.zeros(self.arm_count) + 0.0001  # action count
        self.timestep = 1

    @staticmethod
    def name():
        return 'ucb'

    def get_action(self):
        ln_timestep = np.log(np.full(self.arm_count, self.timestep))
        confidence = self.ucb_c * np.sqrt(ln_timestep / self.N)
        action = np.argmax(self.Q + confidence)
        self.timestep += 1
        return action

    def get_reward_regret(self, arm):
        reward, regret = self.bandit.get_reward_regret(arm)
        self._update_params(arm, reward)
        return reward, regret

    def _update_params(self, arm, reward):
        self.N[arm] += 1  # increment action count
        self.Q[arm] += 1 / self.N[arm] * (reward - self.Q[arm])  # inc. update rule


class RandomSampling:
    def __init__(self, bandit):
        self.bandit = bandit
        self.arm_count = bandit.arm_count

    @staticmethod
    def name():
        return 'random sampling'

    def get_action(self):
        action = np.random.randint(0, self.arm_count)
        return action

    def get_reward_regret(self, arm):
        reward, regret = self.bandit.get_reward_regret(arm)
        return reward, regret


def plot_data(y):
    """ y is a 1D vector """
    x = np.arange(y.size)
    _ = plt.plot(x, y, 'o')


def multi_plot_data(data, names, title1='', title2=''):
    """ data, names are lists of vectors """
    x = np.arange(data[0].size)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, y in enumerate(data):
        plt.plot(x, y, 'o', markersize=2, label=names[i])
    plt.legend(loc='upper right', prop={'size': 16}, numpoints=10, title=title1)
    ax.set_title(title2)
    plt.show()


def simulate(simulations, timesteps, arm_count, Algorithm):
    """ Simulates the algorithm over 'simulations' epochs """
    sum_regrets = np.zeros(timesteps)
    for e in range(simulations):
        bandit = Bandit(arm_count)
        algo = Algorithm(bandit)
        regrets = np.zeros(timesteps)
        for i in range(timesteps):
            action = algo.get_action()
            reward, regret = algo.get_reward_regret(action)
            regrets[i] = regret
        sum_regrets += regrets
    mean_regrets = sum_regrets / simulations
    return mean_regrets


def experiment(arm_count=10, timesteps=1000, simulations=1000):
    """
    Standard setup across all experiments
    Args:
      timesteps: (int) how many steps for the algo to learn the bandit
      simulations: (int) number of epochs
    """
    algos = [EpsilonGreedy]
    regrets = []
    names = []
    for algo in algos:
        regrets.append(simulate(simulations, timesteps, arm_count, algo))
        names.append(algo.name())
    print(regrets[0].size)
    # multi_plot_data(regrets, names)


def experiment1(arm_count=10, epsilons=None, timesteps=10, simulations=10):
    if epsilons is None:
        epsilons = [0]
    algo = EpsilonGreedy
    global epsilon
    regrets = []
    names = []
    for e in epsilons:
        epsilon = e
        regrets.append(simulate(simulations, timesteps, arm_count, algo))
        names.append(e)
    # multi_plot_data(regrets, names, title2='e-greedy with different epsilon')
    print(regrets, names)


def experiment2(arm_count=10, ucb_cs=None, timesteps=1000, simulations=1000):
    if ucb_cs is None:
        ucb_cs = [0, 0.1, 2]
    algo = UCB
    global ucb_c
    regrets = []
    names = []
    for c in ucb_cs:
        ucb_c = c
        regrets.append(simulate(simulations, timesteps, arm_count, algo))
        names.append(c)
    multi_plot_data(regrets, names, title2='UCB with different c')


def experiment3(arm_counts=None, algos=None, timesteps=3000, simulations=1000):
    global ucb_c
    if algos is None:
        algos = [EpsilonGreedy, UCB, BernThompson, RandomSampling]
    if arm_counts is None:
        arm_counts = [10, 50, 200]
    for algo in algos:
        regrets = []
        names = []
        if algo == UCB:
            for c in [2, 0.01]:
                ucb_c = c
                for arm_count in arm_counts:
                    regrets.append(simulate(simulations, timesteps, arm_count, algo))
                    names.append(arm_count)
                multi_plot_data(regrets, names, 'Number of arms', 'c=' + str(ucb_c))
                regrets.clear()
                names.clear()
        else:
            for arm_count in arm_counts:
                regrets.append(simulate(simulations, timesteps, arm_count, algo))
                names.append(arm_count)
            multi_plot_data(regrets, names, 'Number of arms')


def stationarity(arm_count=10, algo=None, timesteps=1000, simulations=1000, title=''):
    global stationary
    stationers = [True, False]
    regrets = []
    names = []
    for s in stationers:
        stationary = s
        regrets.append(simulate(simulations, timesteps, arm_count, algo))
        if stationary:
            names.append('stationary')
        else:
            names.append('non-stationary')
    multi_plot_data(regrets, names, title2=algo.name() + ' ' + title)
    regrets.clear()
    names.clear()


def experiment4():
    global ucb_c, epsilon
    algos = [EpsilonGreedy, UCB, BernThompson, RandomSampling]
    for algo in algos:
        if algo == UCB:
            for c in [2, 0.01]:
                ucb_c = c
                stationarity(algo=algo, title='c=' + str(c))
        elif algo == EpsilonGreedy:
            for e in [0.1, 0.5]:
                epsilon = e
                stationarity(algo=algo, title='e=' + str(e))
        else:
            stationarity(algo=algo)


def experiment5(arm_count=10, timesteps=3000, simulations=1000):
    algos = [BernThompson, BernThompson2]
    regrets = []
    names = []
    for algo in algos:
        regrets.append(simulate(simulations, timesteps, arm_count, algo))
        names.append(algo.name())
    multi_plot_data(regrets, names)


if __name__ == "__main__":
    experiments = {
        1:
            # different epsilon
            experiment1,
        2:
            # different c
            experiment2,
        3:
            # action space
            experiment3,
        4:
            # stationary
            experiment4,
        5:
            # TS with Normal distribution
            experiment5,
        6:
            experiment
    }
    e = input("Enter experiment number:")
    experiment = experiments.get(int(e))
    experiment()
