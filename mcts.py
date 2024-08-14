# mcts.py

import numpy as np
from collections import defaultdict

class MCTS:
    def __init__(self, model, action_evolution, max_depth=5, num_rollouts=32):
        self.model = model
        self.action_evolution = action_evolution
        self.max_depth = max_depth
        self.num_rollouts = num_rollouts
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        
    def search(self, initial_state):
        for _ in range(self.num_rollouts):
            self._simulate(initial_state)
        return max(self.action_evolution.get_relevant_actions(self.model.embed(initial_state)),
                   key=lambda a: self.Q[(initial_state, a)])

    def _simulate(self, state, depth=0):
        if depth == self.max_depth:
            return self._evaluate(state)
        
        action = self._select_action(state)
        next_state = self.action_evolution.apply_action(action, state)
        reward = self._simulate(next_state, depth + 1)
        
        self.Q[(state, action)] += reward
        self.N[(state, action)] += 1
        return reward

    def _select_action(self, state):
        actions = self.action_evolution.get_relevant_actions(self.model.embed(state))
        return max(actions, key=lambda a: self._ucb_score(state, a))

    def _ucb_score(self, state, action):
        q = self.Q[(state, action)]
        n = self.N[(state, action)]
        N = sum(self.N[(state, a)] for a in self.action_evolution.get_relevant_actions(self.model.embed(state)))
        return q / (n + 1e-8) + np.sqrt(2 * np.log(N) / (n + 1e-8))

    def _evaluate(self, state):
        words = state.split()
        unique_words = set(words)
        clarity_score = len(unique_words) / len(words)
        complexity_score = sum(len(word) for word in words) / len(words)
        correctness_score = 1.0 if "Solved directly" in state or "Applied relevant formula" in state else 0.5
        return (clarity_score + complexity_score + correctness_score) / 3
