# main.py

from sentence_embedder import SentenceEmbedder
from adaptive_action_evolution import AdaptiveActionEvolution
from mcts import MCTS
from discriminator import Discriminator
from rstar import RStar

def main():
    model = SentenceEmbedder()
    initial_actions = ["decompose", "solve_directly", "apply_formula", "estimate", "simplify", "rephrase", "verify_step"]
    discriminator = Discriminator(input_dim=384)  # 384 is the embedding dimension for the specified model
    rstar = RStar(model, initial_actions, discriminator)

    problem = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"
    solution = rstar.solve(problem)
    print(f"Problem: {problem}")
    print(f"Solution: {solution}")

if __name__ == "__main__":
    main()
