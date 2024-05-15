from itertools import permutations

import numpy as np


class Agent:
    id: int
    costs: dict  # costs[alternative] = cost
    preference: int
    ordinal_preferences: list

    def __init__(self, id, costs=None, ordinal_preferences=None, alpha=None):
        """
        :param id:
        :param costs:
        :param ordinal_preferences:
        :param alpha:
        """
        self.id = id
        self.costs=dict()
        if costs is not None:
            self.costs = costs
            self.ordinal_preferences = sorted(costs, key=costs.get)
            ordinal_preferences = self.ordinal_preferences
            self.preference = self.ordinal_preferences[0]
        if alpha is not None:
            self.ordinal_preferences = ordinal_preferences
            self.preference = ordinal_preferences[0]
            for i in range(len(ordinal_preferences)):
                self.costs[ordinal_preferences[i]] = 0
            self.costs[ordinal_preferences[1]] = 2 * np.random.uniform(0.0, 2.0)
            self.costs = self.possible_costs(alpha)

    def possible_costs(self,alpha):
        """
        randomly generates costs for the agent such that the original preference is being preserved (the lowest cost)
        and the ratio between the lowest cost and the second-lowest cost is alpha at most
        """
        fictitious_costs = {}
        alternatives = self.ordinal_preferences.copy()
        second_alternative = alternatives[1]
        fictitious_costs[second_alternative] = self.costs[second_alternative]
        alternatives.remove(second_alternative)
        alpha_preserving_cost = self.costs[second_alternative] * alpha
        fictitious_costs[self.preference] = np.random.uniform(0,alpha_preserving_cost)
        alternatives.remove(self.preference)
        # randomly assign costs to the remaining alternatives while preserving the order
        costs = sorted([np.random.uniform(fictitious_costs[second_alternative],2) for _ in range(len(alternatives))])
        for i in range(len(alternatives)):
            fictitious_costs[alternatives[i]] = costs[i]
        return fictitious_costs


class Election:
    candidates: list
    alternatives: list
    alpha: float

    def __init__(self, candidates):
        self.candidates = candidates
        self.alternatives = candidates # alternatives are the candidates themselves

    def get_best_alpha(self, agents):
        """
        :return: the best alpha for the election
        """
        alpha = 0
        for agent in agents:
            costs = agent.costs
            costs = sorted(costs.items(), key=lambda x: x[1])
            alpha = max(alpha, costs[0][1] / costs[1][1])
        return alpha
def random_election(num_agents, num_candidates):
    agents = []
    candidates = list(range(num_candidates))
    election = Election(candidates)
    for i in range(num_agents):
        costs = {}
        for alternative in election.alternatives:
            costs[alternative] = np.random.rand()
        agents.append(Agent(i, costs))
    # find the best alpha for the election
    alpha = 0
    for agent in agents:
        costs = agent.costs
        costs = sorted(costs.items(), key=lambda x: x[1])
        alpha = max(alpha, costs[0][1] / costs[1][1])
    election.alpha = alpha
    return election, agents
def random_election_with_alpha(num_agents, num_candidates, alpha):
    agents = []
    candidates = list(range(num_candidates))
    election = Election(candidates)
    election.alpha = alpha
    for i in range(num_agents):
        preference_order= list(np.random.permutation(election.alternatives))
        agents.append(Agent(i, ordinal_preferences=preference_order, alpha=alpha))
    return election, agents
