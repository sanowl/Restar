# adaptive_action_evolution.py

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class AdaptiveActionEvolution:
    def __init__(self, initial_action_space, model):
        self.action_space = initial_action_space
        self.model = model
        self.action_performance = defaultdict(list)
        self.problem_embeddings = []
        
    def update_performance(self, action, problem_embedding, performance):
        self.action_performance[action].append((problem_embedding, performance))
        self.problem_embeddings.append(problem_embedding)
        
    def evolve_actions(self):
        if len(self.problem_embeddings) < 5:
            return

        kmeans = KMeans(n_clusters=min(5, len(self.problem_embeddings)))
        clusters = kmeans.fit_predict(self.problem_embeddings)
        
        cluster_performance = defaultdict(lambda: defaultdict(float))
        for action, performances in self.action_performance.items():
            for embedding, perf in performances:
                cluster = kmeans.predict([embedding])[0]
                cluster_performance[cluster][action] += perf
        
        new_actions = []
        for cluster in cluster_performance:
            top_actions = sorted(cluster_performance[cluster].items(), key=lambda x: x[1], reverse=True)[:2]
            if len(top_actions) > 1:
                new_action = self.combine_actions(top_actions[0][0], top_actions[1][0])
                new_actions.append(new_action)
        
        self.action_space = list(set(self.action_space + new_actions))
        
    def combine_actions(self, action1, action2):
        return f"Hybrid({action1}, {action2})"
    
    def get_relevant_actions(self, problem_embedding):
        if len(self.problem_embeddings) < 5:
            return self.action_space[:5]

        kmeans = KMeans(n_clusters=min(5, len(self.problem_embeddings)))
        cluster = kmeans.fit_predict([problem_embedding])[0]
        cluster_actions = sorted(self.action_performance.items(), 
                                 key=lambda x: np.mean([p for e, p in x[1] if kmeans.predict([e])[0] == cluster]),
                                 reverse=True)
        return [action for action, _ in cluster_actions[:5]]

    def apply_action(self, action, state):
        if action.startswith("Hybrid"):
            sub_actions = action[7:-1].split(", ")
            for sub_action in sub_actions:
                state = self.apply_action(sub_action, state)
            return state
        elif action == "decompose":
            return f"{state} [Decomposed into subproblems]"
        elif action == "solve_directly":
            return f"{state} [Solved directly]"
        elif action == "apply_formula":
            return f"{state} [Applied relevant formula]"
        elif action == "estimate":
            return f"{state} [Estimated result]"
        elif action == "simplify":
            return f"{state} [Simplified problem]"
        elif action == "rephrase":
            return f"{state} [Rephrased problem for clarity]"
        elif action == "verify_step":
            return f"{state} [Verified this reasoning step]"
        else:
            return f"{state} [Applied {action}]"
