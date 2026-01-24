"""
Distributed Monte Carlo Tree Search (MCTS) Reasoner
====================================================
A "build your own o1" implementation using Modal for parallel reasoning exploration.

This system explores thousands of reasoning paths in parallel, using MCTS to intelligently
navigate the space of possible thought sequences and find optimal solutions to complex problems.
"""

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import modal

# Modal setup
app = modal.App("mcts-reasoner")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai>=1.12.0", "numpy>=1.24.0"
)

# Configuration
OPENAI_API_KEY = modal.Secret.from_name("openai-api-key")
MODEL_NAME = "gpt-4o-mini"  # Fast and cheap for exploration
MAX_DEPTH = 8  # Maximum reasoning steps
EXPLORATION_CONSTANT = 1.414  # UCB1 exploration parameter (sqrt(2))
NUM_SIMULATIONS = 100  # Total MCTS iterations
PARALLEL_WORKERS = 20  # Concurrent explorations


@dataclass
class Node:
    """
    Represents a node in the MCTS reasoning tree.
    Each node is a partial reasoning path with statistics for UCB1 selection.
    """

    state: str  # Current reasoning step/thought
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # Total reward accumulated
    depth: int = 0
    is_terminal: bool = False
    is_solution: bool = False

    def ucb1_score(self, exploration_weight: float = EXPLORATION_CONSTANT) -> float:
        """
        Upper Confidence Bound formula for balancing exploration vs exploitation.
        Returns infinity for unvisited nodes to prioritize exploration.
        """
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self) -> Optional["Node"]:
        """Select child with highest UCB1 score."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1_score())

    def update(self, reward: float):
        """Backpropagate reward up the tree."""
        self.visits += 1
        self.value += reward

    def get_path(self) -> List[str]:
        """Get full reasoning path from root to this node."""
        path = []
        current = self
        while current is not None:
            if current.state:  # Skip empty root
                path.append(current.state)
            current = current.parent
        return list(reversed(path))


@app.cls(
    image=image,
    secrets=[OPENAI_API_KEY],
    timeout=1800,
)
class MCTSMaster:
    """
    Stateful master node managing the global MCTS tree.
    Handles selection, expansion, and backpropagation across parallel workers.
    """

    def __init__(self, problem: str):
        self.problem = problem
        self.root = Node(state="", depth=0)
        self.lock = asyncio.Lock()
        self.best_solution = None
        self.best_reward = -float("inf")

    @modal.method()
    async def select_leaf(self) -> Dict[str, Any]:
        """
        Selection phase: traverse tree using UCB1 until reaching a leaf.
        Returns the leaf node's state and path for expansion.
        """
        async with self.lock:
            current = self.root

            # Traverse to leaf using UCB1
            while current.children and not current.is_terminal:
                current = current.best_child()
                if current is None:
                    break

            # Return serializable node data
            return {
                "state": current.state,
                "depth": current.depth,
                "path": current.get_path(),
                "node_id": id(current),  # For tracking during backprop
            }

    @modal.method()
    async def expand_and_evaluate(
        self, node_data: Dict[str, Any], new_steps: List[str], rewards: List[float]
    ):
        """
        Expansion phase: add new children to the tree and backpropagate rewards.
        This is called by workers after LLM generation.
        """
        async with self.lock:
            # Find parent node (simplified - in production use proper ID mapping)
            current = self.root
            for step in node_data["path"]:
                found = False
                for child in current.children:
                    if child.state == step:
                        current = child
                        found = True
                        break
                if not found:
                    break

            # Add new children
            for step, reward in zip(new_steps, rewards):
                child = Node(
                    state=step,
                    parent=current,
                    depth=current.depth + 1,
                    is_terminal=(current.depth + 1 >= MAX_DEPTH),
                )
                current.children.append(child)

                # Backpropagate
                self._backpropagate(child, reward)

                # Track best solution
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_solution = child.get_path()

    def _backpropagate(self, node: Node, reward: float):
        """Propagate reward up to root, updating all ancestors."""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    @modal.method()
    async def get_best_solution(self) -> Dict[str, Any]:
        """Return the best reasoning path found."""
        async with self.lock:
            if self.best_solution is None:
                # Fallback: most visited path from root
                path = []
                current = self.root
                while current.children:
                    current = max(current.children, key=lambda c: c.visits)
                    path.append(current.state)
                return {"path": path, "reward": 0.0}

            return {
                "path": self.best_solution,
                "reward": self.best_reward,
                "total_simulations": self.root.visits,
            }

    @modal.method()
    async def get_tree_stats(self) -> Dict[str, int]:
        """Get tree exploration statistics."""
        async with self.lock:

            def count_nodes(node: Node) -> int:
                return 1 + sum(count_nodes(c) for c in node.children)

            return {
                "total_nodes": count_nodes(self.root),
                "root_visits": self.root.visits,
                "num_children": len(self.root.children),
            }


@app.function(
    image=image,
    secrets=[OPENAI_API_KEY],
    timeout=300,
)
async def mcts_worker(
    problem: str, node_data: Dict[str, Any], worker_id: int
) -> Dict[str, Any]:
    """
    Worker function: expands a leaf node by generating next reasoning steps with LLM.
    Returns new steps and their evaluated rewards.
    """
    import openai

    client = openai.AsyncOpenAI()
    current_path = node_data["path"]
    depth = node_data["depth"]

    # Build prompt with current reasoning chain
    prompt = f"""You are solving this problem using step-by-step reasoning:

PROBLEM: {problem}

REASONING SO FAR:
{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(current_path)) if current_path else "(starting)"}

Generate the NEXT logical reasoning step. Be concise and specific.
If you've reached a solution, state it clearly with "SOLUTION: [answer]"

Next step:"""

    try:
        # Generate next reasoning step
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8,  # Higher temp for diversity
            n=3,  # Generate multiple candidate steps
        )

        steps = []
        rewards = []

        for choice in response.choices:
            step = choice.message.content.strip()

            # Check if this is a solution
            is_solution = "SOLUTION:" in step.upper() or depth >= MAX_DEPTH - 1

            # Evaluate step quality (simulate rollout)
            reward = await evaluate_step(
                problem, current_path + [step], is_solution, client
            )

            steps.append(step)
            rewards.append(reward)

        return {"steps": steps, "rewards": rewards, "worker_id": worker_id}

    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        return {"steps": [], "rewards": [], "worker_id": worker_id}


async def evaluate_step(
    problem: str, path: List[str], is_solution: bool, client
) -> float:
    """
    Evaluate the quality of a reasoning path.
    Uses LLM to score logical consistency and correctness.
    """
    # Fast heuristic evaluation
    path_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(path))

    if is_solution:
        # Full verification for claimed solutions
        verify_prompt = f"""Problem: {problem}

Proposed solution path:
{path_text}

Is this solution correct? Answer with a score from 0.0 (completely wrong) to 1.0 (perfect solution).
Score:"""

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": verify_prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            score_text = response.choices[0].message.content.strip()
            # Extract numeric score
            match = re.search(r"(\d+\.?\d*)", score_text)
            if match:
                return float(match.group(1))
            return 0.5

        except Exception:
            return 0.5
    else:
        # Heuristic: longer valid paths get higher base reward
        # Penalize repetition
        unique_ratio = len(set(path)) / len(path) if path else 1.0
        depth_bonus = len(path) * 0.1
        return min(unique_ratio * depth_bonus, 0.8)  # Cap non-solution rewards


@app.function(
    image=image,
    secrets=[OPENAI_API_KEY],
    timeout=1800,
)
async def run_mcts(
    problem: str, num_simulations: int = NUM_SIMULATIONS
) -> Dict[str, Any]:
    """
    Main MCTS orchestration: run parallel simulations to find best reasoning path.
    """
    print(f"🧠 Starting MCTS with {num_simulations} simulations...")
    print(f"Problem: {problem}\n")

    # Initialize master tree
    master = MCTSMaster(problem)

    # Run simulations in batches for parallelism
    batch_size = PARALLEL_WORKERS
    num_batches = (num_simulations + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        print(f"📊 Batch {batch_idx + 1}/{num_batches}")

        # Select leaves to expand
        tasks = []
        for worker_id in range(
            min(batch_size, num_simulations - batch_idx * batch_size)
        ):
            tasks.append(master.select_leaf.remote())

        leaf_nodes = await asyncio.gather(*tasks)

        # Expand leaves in parallel
        expansion_tasks = []
        for worker_id, node_data in enumerate(leaf_nodes):
            expansion_tasks.append(
                mcts_worker.remote.aio(problem, node_data, worker_id)
            )

        results = await asyncio.gather(*expansion_tasks)

        # Update tree with results
        for node_data, result in zip(leaf_nodes, results):
            if result["steps"]:
                await master.expand_and_evaluate.remote(
                    node_data, result["steps"], result["rewards"]
                )

        # Show progress
        stats = await master.get_tree_stats.remote()
        print(f"  Nodes: {stats['total_nodes']}, Root visits: {stats['root_visits']}")

    # Get final solution
    solution = await master.get_best_solution.remote()

    print("\n✅ MCTS Complete!")
    print(f"Best reward: {solution['reward']:.3f}")
    print(f"Total simulations: {solution['total_simulations']}")

    return solution


@app.local_entrypoint()
def main(problem: str = None):
    """
    CLI entry point with example problems.
    """
    # Example problems
    EXAMPLE_PROBLEMS = {
        "math": "If x + 5 = 12 and y = 2x, what is y²?",
        "logic": "Three people (A, B, C) are in a room. A says 'B is lying'. B says 'C is lying'. C says 'A and B are both lying'. Who is telling the truth?",
        "code": "Write a Python function to find the longest palindromic substring in O(n²) time.",
    }

    if problem is None:
        print("🎯 Example Problems:")
        for key, prob in EXAMPLE_PROBLEMS.items():
            print(f"  {key}: {prob}")
        problem = EXAMPLE_PROBLEMS["math"]
        print("\n🔥 Running default problem: math\n")

    # Run MCTS
    result = run_mcts.remote(problem)

    # Display results
    print("\n" + "=" * 60)
    print("🎯 FINAL REASONING PATH:")
    print("=" * 60)
    for i, step in enumerate(result["path"], 1):
        print(f"{i}. {step}")
    print("=" * 60)
    print(f"\n💎 Confidence Score: {result['reward']:.2f}")


# Usage:
# modal run distributed_mcts_reasoner.py
# modal run distributed_mcts_reasoner.py --problem "Your custom problem here"
