# rstar.py

import torch
from adaptive_action_evolution import AdaptiveActionEvolution
from mcts import MCTS
from discriminator import Discriminator

class RStar:
    def __init__(self, model, initial_action_space, discriminator):
        self.model = model
        self.action_evolution = AdaptiveActionEvolution(initial_action_space, model)
        self.mcts = MCTS(model, self.action_evolution)
        self.discriminator = discriminator
    
    def solve(self, problem):
        embedding = self.model.embed(problem)
        candidate_solutions = []
        
        for _ in range(5):  # Generate multiple candidate solutions
            solution = self._generate_solution(problem)
            candidate_solutions.append(solution)
            self.action_evolution.update_performance(solution, embedding, self._evaluate_solution(solution))
        
        self.action_evolution.evolve_actions()
        
        best_solution = max(candidate_solutions, key=self._evaluate_solution)
        return best_solution
    
    def _generate_solution(self, problem):
        current_state = problem
        solution = []
        
        for _ in range(self.mcts.max_depth):
            action = self.mcts.search(current_state)
            current_state = self.action_evolution.apply_action(action, current_state)
            solution.append(action)
        
        return " -> ".join(solution)
    
    def _evaluate_solution(self, solution):
        with torch.no_grad():
            return self.discriminator(torch.tensor(self.model.embed(solution), dtype=torch.float32)).item()
