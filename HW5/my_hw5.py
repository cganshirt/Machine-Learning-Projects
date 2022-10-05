import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from stoppedcar import StoppedCar
from fourierbasis import FourierBasis
import time

# TODO read all functions and documentation to understand what each component is supposed to do.
# It is recommended to start with the main function since it is the highest level and dive into each component.

class LinearSoftmax(object):
    def __init__(self, basis:FourierBasis, n_actions:int):
        """
        This creates a linear softmax policy using the specified basis function
        Parameters
        ----------
        basis: FourierBasis
            The basis function to use with the policy
        n_actions : int
            The number of possible actions
        """

        self.basis = basis
        self.n_actions = n_actions
        self.n_inputs =  basis.getNumFeatures()

        self.basis = basis

        # These are the policy weights. They are a 2D numpy array.
        # To get the vector of weights for the a^th action you can do self.theta[a]
        self.theta = np.zeros((self.n_actions, self.n_inputs))

        self.num_params = int(self.theta.size)


    def get_action(self, state:np.ndarray)->int:
        """
        This function samples an action for the provided state features.

        Parameters
        ----------
        state : np.ndarray
            The state features (no basis function applied yet)

        Returns
        -------
        a : int
            The sampled action
        """
        x = self.basis.encode(state)  # Computes the basis function representation of the state features
        p = self.get_action_probabilities(x)  # computes the probabilities of each action
        a = int(np.random.choice(range(p.shape[0]), p=p, size=1))  # samples the action from p

        return a

    def get_params(self)->np.ndarray:
        """
        This function returns the weights of the policy. This is just a helper
        function.
        Returns
        -------
        theta : np.ndarray
            The weights of the policy
        """
        return self.theta

    def add_to_params(self, x: np.ndarray):
        """
        This function adds the input array to the weights. You can use this
        function to update the policy weights.

        Parameters
        ----------
        x : np.ndarray
            An array that is used to change the policy weights

        Returns
        -------
        None

        """
        assert self.theta.shape == x.shape, "x and theta have different shapes"
        self.theta += x

    def get_action_probabilities(self, x: np.ndarray)->np.ndarray:
        """
        Compute the probabilities for each action for the features x. This
        function should compute the outputs values for each action unit then
        perform a softmax over all outputs. The return value should be a 1D
        numpy array containing the probabilities for each action.

        Parameters
        ----------
        x : np.ndarray
            The state features after the basis function is applied.

        Returns
        -------
        p : np.ndarray
            The probabilities for each action
        """
        theta = self.theta
        p = np.zeros(self.n_actions)  # TODO replace this line and implement function

        return p

    def gradient_logprob(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        This function computes the partial derivative of ln pi(s,a) with
        respect to the policy weights. The functions returns two quantities:
        a numpy array of the partial derivatives and the log probability of
        the action specified, e.g., ln pi(s,a). Note we also mean the natural
        log here and not log with base 2 or 10.

        Parameters
        ----------
        state : np.ndarray
            The state features (no basis function is applied)
        action : int
            The action that was chosen

        Returns
        -------
        dtheta : np.ndarray
            A 2D numpy array containing the partial derivatives (needs to be the same shape as the weights)
        logp : float
            The log probability of the action for the state features
        """
        dtheta = np.zeros_like(self.theta)  # initialize the vector of partial derivatives
        logp = 0.0  # TODO replace this line with the actual log probability
        # TODO compute derivative. Remember to get the features from the basis function. State is the actual state features.

        return dtheta, logp

    def gradient_prob(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        This function computes the partial derivative of the pi(s,a) with
        respect to the policy parameters. The function should return a 2D numpy
        array wit the same shape as the weights of the policy.

        Parameters
        ----------
        state : np.ndarray
            The state features (basis function has not been applied)
        action : int
            The action that was chosen

        Returns
        -------
        dtheta : np.ndarray
            2D numpy array containing the partial derivatives for all weights (needs to be the same shape as the weights)
        """
        dtheta = np.zeros_like(self.theta)  # TODO replace this line and implement function

        return dtheta

def compute_sum_of_discounted_rewards(rewards: List[float], gamma: float)->List[float]:
    """
    The function returns a list where the t^th element of the list is the discounted sum of rewards starting at time t.
    Parameters
    ----------
    rewards : List[float]
        The list of rewards observed from one episode
    gamma : float
        The reward discount factor

    Returns
    -------
    Gs : List[float]
        List of the discounted sum of rewards starting at each point in time
    """
    Gs = []
    # TODO implement function

    return Gs

def sample_episode(env: StoppedCar, policy: LinearSoftmax):
    """
    This function runs one episode sampling form the actions. It then returns
    the sum of rewards, and lists of all states, actions, and rewards

    Parameters
    ----------
    env : StoppedCar
        The environment to interact with
    policy : LinearSoftmax
        The policy used to sample actions.

    Returns
    -------
    G : float
        Sum of rewards (no discounting)
    states: List[np.ndarray]
        List of every state encountered before the end of the episode
    actions: List[int]
        List of every action encountered before the end of the episode
    rewards: List[float]
        List of every reward observed before the end of the episode
    """
    G = 0.0  # variable to store the sum of rewards with no discounting (gamma=1.0)
    # lists to store the states, actions, and rewards from the episode
    states = []
    actions = []
    rewards = []

    s = env.reset()  # sample the initial state from the environment
    done = False  # a flag to see if the episode is over

    while not done:
        a = policy.get_action(s)  # sample action from policy
        states.append(s)
        actions.append(a)
        s,reward,done = env.step(a)  # get the next state, reward, and see if episode is over
        rewards.append(reward)
        G += reward  # add the reward to the sum of rewards


    return G, states, actions, rewards

def update_policy(policy:LinearSoftmax, states, actions, rewards, alpha, gamma):
    """
    This function performance the policy update for the Simple RL algorithm
    covered in class. The update is performed using the passed in observations
    from one episode.

    Parameters
    ----------
    policy : LinearSoftmax
        The policy being optimized
    states : List[np.ndarray]
        The list of states (in order) from one episode
    actions : List[int]
        The list of actions (in order) from one episode
    rewards : List[float]
        The list of rewards (in order) from one episode
    alpha : float
        The step size used to update the policy weights
    gamma : float
        The reward discount factor.

    Returns
    -------
    None

    """
    # TODO implement this function


def run_simple_rl_algorithm(env: StoppedCar, policy: LinearSoftmax,
                            alpha: float, gamma: float, num_episodes: int):
    """
    This function runs the Simple RL algorithm for one lifetime and returns
    a list of the sum of rewards (with no discounting) from all episodes.
    Parameters
    ----------
    env : StoppedCar
        The environment to interact with
    policy : LinearSoftmax
        The policy to optimize
    alpha : float
        The step size used to update the policy
    gamma : float
        The rewards discount factor
    num_episodes : int
        The number of episodes in one lifetime

    Returns
    -------
    sums_of_rewards : list[float]
        The list of sum of rewards from each episode in the lifetime
    """
    sums_of_rewards = []
    # TODO read this function and understand what is getting called
    for _ in range(num_episodes):
        G, states, actions, rewards = sample_episode(env, policy)  # run one episode
        sums_of_rewards.append(G)  # save the sum of rewards (no discounting)
        update_policy(policy, states, actions, rewards, alpha, gamma)  # update the policy using the data from the episode

    return sums_of_rewards


def learning_curve(all_sums_of_rewards):
    """
    This function just makes the learning curve plot
    Parameters
    ----------
    all_sums_of_rewards : list
        a list of the sums_of_rewards from each lifetime

    Returns
    -------
    nothing
    """
    fig, axs = plt.subplots()
    for sums_of_rewards in all_sums_of_rewards:
        x = range(len(sums_of_rewards))
        axs.scatter(x, sums_of_rewards, color="dodgerblue", alpha=0.01, s=0.5)
    mn = np.mean(all_sums_of_rewards, axis=0)
    std = np.std(all_sums_of_rewards, axis=0)
    axs.plot(mn, linewidth=2, color="crimson")
    axs.fill_between(range(len(mn)), mn+std, mn-std, alpha=0.5, color="crimson")
    axs.hlines(7.5, 0, len(mn), color="black", ls="dashed")
    axs.set_xlabel("Episode")
    axs.set_ylabel("Sum of Rewards")
    axs.set_title("Performance of Policy")
    plt.savefig("learning_curve.png")
    plt.show()

def main():

    env = StoppedCar()  # create environment

    order = 1  # order for fourier basis TODO tune this parameter
    basis = FourierBasis(env.obs_rangs, order)  # NOTE:  the fourier basis uses coupled terms not just the independent terms as in the previous homework


    alpha = 0.0  # TODO optimize the steps-size
    gamma = 0.0  # TODO optimize the discount factor

    all_sums_of_rewards = []
    number_of_lifetimes = 100  # number of lifetimes (trials) to run the algorithm. Set the number of lifetimes to 1 to debug and tune your step size faster, but you must run 100 lifetimes to report your performance
    number_of_episodes = 400  # number of episodes per lifetime. This can be set small for debuging and tuning the step size, but must be 400 to report your results

    start = time.time()  # start a timer to see how long it takes to run the algorithm. On my old laptop I can run 100 lifetimes of 400 episodes in about 400 seconds. However, different hyperparmeters will have different run times.
    # run the algorithm for each lifetime
    for i in range(number_of_lifetimes):
        policy = LinearSoftmax(basis, env.num_actions)  # initialize the policy for each lifetime
        sums_of_rewards = run_simple_rl_algorithm(env, policy, alpha, gamma, number_of_episodes)  # run the algorithm
        all_sums_of_rewards.append(sums_of_rewards)  # log the performance from this lifetime

    end = time.time()  # take not of the stopping time
    print("Time to run all lifetimes: {0:.1f}(s)".format(end-start))

    learning_curve(all_sums_of_rewards)  # plot the performance of the agent for all lifetimes.


if __name__ == "__main__":
    main()