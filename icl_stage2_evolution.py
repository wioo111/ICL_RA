from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import exp, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]
D, C, G = "D", "C", "G"
STRATEGIES = np.array([D, C, G])


@dataclass
class Agent:
    ability: float
    strategy: str


@dataclass
class Config:
    # Stage 2: 5x5 = 25 agents
    L: int = 5

    # Paper-model parameters
    c: float = 1.0
    d: float = 1.5
    beta: float = 1.5
    M: float = 2.0
    T: int = 2
    kappa: float = 0.1

    # Ability distribution A ~ N(m, v).
    # The paper calls v the variance, so numpy uses sqrt(v) as std.
    m: float = 1.0
    v: float = 0.1

    # Simulation settings
    steps: int = 200
    seed: int = 42


class ICLWorld:
    """Stage-2 executable I-C-L evolutionary game.

    Implemented:
    - L x L square lattice
    - periodic boundaries
    - 4-neighbor Von Neumann neighborhood
    - local group = focal + 4 neighbors
    - D/C/G strategies
    - rank-based allocation
    - average payoff across 5 overlapping local games
    - asynchronous Fermi imitation

    Stage-2 convention: 1 Monte Carlo step (MCS) = N elementary updates.
    We will validate this low-level convention later against the paper figures.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.L = cfg.L
        self.N = cfg.L * cfg.L
        self.rng = np.random.default_rng(cfg.seed)

        abilities = self.rng.normal(
            loc=cfg.m,
            scale=sqrt(cfg.v),
            size=(cfg.L, cfg.L),
        )
        strategy_grid = self.rng.choice(
            STRATEGIES,
            size=(cfg.L, cfg.L),
            p=[1 / 3, 1 / 3, 1 / 3],
        )

        self.agents: Dict[Coord, Agent] = {}
        for r in range(cfg.L):
            for col in range(cfg.L):
                self.agents[(r, col)] = Agent(
                    ability=float(abilities[r, col]),
                    strategy=str(strategy_grid[r, col]),
                )

    # ---------- network ----------
    def neighbors(self, pos: Coord) -> List[Coord]:
        r, col = pos
        L = self.L
        return [
            ((r - 1) % L, col),
            ((r + 1) % L, col),
            (r, (col - 1) % L),
            (r, (col + 1) % L),
        ]

    def group(self, center: Coord) -> List[Coord]:
        return [center] + self.neighbors(center)

    def groups_containing(self, pos: Coord) -> List[Coord]:
        return [pos] + self.neighbors(pos)

    # ---------- strategy mechanics ----------
    def cost(self, strategy: str) -> float:
        if strategy == D:
            return self.cfg.d
        if strategy == C:
            return self.cfg.c
        if strategy == G:
            return 0.0
        raise ValueError(f"Unknown strategy: {strategy}")

    def utility(self, pos: Coord) -> float:
        agent = self.agents[pos]
        if agent.strategy == D:
            return self.cfg.beta * agent.ability
        if agent.strategy == C:
            return agent.ability
        if agent.strategy == G:
            return 0.0
        raise ValueError(f"Unknown strategy: {agent.strategy}")

    # ---------- rank-based allocation ----------
    def rank_in_group(self, pos: Coord, center: Coord) -> int:
        members = self.group(center)
        ordered = sorted(
            members,
            key=lambda p: (-self.utility(p), p[0], p[1]),
        )
        return ordered.index(pos) + 1

    def payoff_in_group(self, pos: Coord, center: Coord) -> float:
        agent = self.agents[pos]
        if agent.strategy == G:
            return 0.0

        reward = self.cfg.M if self.rank_in_group(pos, center) <= self.cfg.T else 0.0
        return reward - self.cost(agent.strategy)

    def total_payoff(self, pos: Coord) -> float:
        values = [
            self.payoff_in_group(pos, center)
            for center in self.groups_containing(pos)
        ]
        return float(np.mean(values))

    # ---------- Fermi update ----------
    def imitation_probability(self, pi_i: float, pi_j: float) -> float:
        z = (pi_i - pi_j) / self.cfg.kappa
        if z > 700:
            return 0.0
        if z < -700:
            return 1.0
        return 1.0 / (1.0 + exp(z))

    def elementary_update(self) -> None:
        idx = int(self.rng.integers(self.N))
        i = (idx // self.L, idx % self.L)
        nbrs = self.neighbors(i)
        j = nbrs[int(self.rng.integers(4))]

        pi_i = self.total_payoff(i)
        pi_j = self.total_payoff(j)
        p_copy = self.imitation_probability(pi_i, pi_j)

        if float(self.rng.random()) < p_copy:
            self.agents[i].strategy = self.agents[j].strategy

    def monte_carlo_step(self) -> None:
        for _ in range(self.N):
            self.elementary_update()

    # ---------- observables ----------
    def strategy_counts(self) -> Dict[str, int]:
        counts = {D: 0, C: 0, G: 0}
        for agent in self.agents.values():
            counts[agent.strategy] += 1
        return counts

    def summary_row(self, step: int) -> Dict[str, float]:
        counts = self.strategy_counts()
        mean_payoff = float(np.mean([self.total_payoff(pos) for pos in self.agents]))
        return {
            "step": step,
            "n_D": counts[D],
            "n_C": counts[C],
            "n_G": counts[G],
            "f_D": counts[D] / self.N,
            "f_C": counts[C] / self.N,
            "f_G": counts[G] / self.N,
            "mean_payoff": mean_payoff,
        }


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "n_D", "n_C", "n_G", "f_D", "f_C", "f_G", "mean_payoff"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in rows]
    f_D = [r["f_D"] for r in rows]
    f_C = [r["f_C"] for r in rows]
    f_G = [r["f_G"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, f_D, label="D: involution")
    ax.plot(steps, f_C, label="C: cooperation")
    ax.plot(steps, f_G, label="G: lying flat")
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel("Strategy proportion")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Stage 2: I-C-L strategy evolution on a 5x5 lattice")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_final_state(world: ICLWorld, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "ability", "strategy"])
        for r in range(world.L):
            for col in range(world.L):
                agent = world.agents[(r, col)]
                writer.writerow([r, col, agent.ability, agent.strategy])


def run(cfg: Config, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    world = ICLWorld(cfg)

    expected_corner_neighbors = {(cfg.L - 1, 0), (1, 0), (0, cfg.L - 1), (0, 1)}
    assert set(world.neighbors((0, 0))) == expected_corner_neighbors

    rows: List[Dict[str, float]] = [world.summary_row(step=0)]
    for step in range(1, cfg.steps + 1):
        world.monte_carlo_step()
        rows.append(world.summary_row(step=step))

    trajectory_csv = output_dir / "stage2_strategy_evolution.csv"
    figure_png = output_dir / "stage2_strategy_evolution.png"
    final_state_csv = output_dir / "stage2_final_state.csv"
    config_json = output_dir / "stage2_config.json"

    save_csv(rows, trajectory_csv)
    save_plot(rows, figure_png)
    save_final_state(world, final_state_csv)
    config_json.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    first, last = rows[0], rows[-1]
    print("=== Stage 2 finished ===")
    print(f"N={world.N}, L={cfg.L}, steps={cfg.steps}, seed={cfg.seed}")
    print(
        f"initial: D={first['n_D']} ({first['f_D']:.3f}), "
        f"C={first['n_C']} ({first['f_C']:.3f}), "
        f"G={first['n_G']} ({first['f_G']:.3f})"
    )
    print(
        f"final:   D={last['n_D']} ({last['f_D']:.3f}), "
        f"C={last['n_C']} ({last['f_C']:.3f}), "
        f"G={last['n_G']} ({last['f_G']:.3f})"
    )
    print(f"CSV:    {trajectory_csv}")
    print(f"PNG:    {figure_png}")
    print(f"STATE:  {final_state_csv}")
    print(f"CONFIG: {config_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: 25-agent I-C-L evolutionary simulation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage2"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(L=5, steps=args.steps, seed=args.seed)
    run(cfg, args.output_dir)
