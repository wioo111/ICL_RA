
from __future__ import annotations
#让后续类型注解延迟求值，避免一些版本兼容问题
from dataclasses import dataclass
from math import exp
from random import Random
#随机数生成器
from typing import Dict, List, Tuple
#定义类型注解

Coord = Tuple[int, int]
D, C, G = "D", "C", "G"

@dataclass
#定义一个数据类，用于表示智能体
class Agent:
    ability: float
    strategy: str

class ICLWorld:
    def __init__(self, L=5, c=1.0, d=1.5, beta=1.5, M=2.0, T=2, kappa=0.1, seed=42):
        self.L, self.c, self.d = L, c, d
        self.beta, self.M, self.T, self.kappa = beta, M, T, kappa
        self.rng = Random(seed)
        self.agents: Dict[Coord, Agent] = {
            (r, col): Agent(1.0, C)
            for r in range(L)
            for col in range(L)
        }

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

    def cost(self, strategy: str) -> float:
        if strategy == D:
            return self.d
        if strategy == C:
            return self.c
        if strategy == G:
            return 0.0
        raise ValueError(strategy)

    def utility(self, pos: Coord) -> float:
        a = self.agents[pos]
        if a.strategy == D:
            return self.beta * a.ability
        if a.strategy == C:
            return a.ability
        if a.strategy == G:
            return 0.0
        raise ValueError(a.strategy)

    def rank_in_group(self, pos: Coord, center: Coord) -> int:
        members = self.group(center)
        ordered = sorted(members, key=lambda p: (-self.utility(p), p[0], p[1]))
        return ordered.index(pos) + 1

    def payoff_in_group(self, pos: Coord, center: Coord) -> float:
        a = self.agents[pos]
        if a.strategy == G:
            return 0.0
        reward = self.M if self.rank_in_group(pos, center) <= self.T else 0.0
        return reward - self.cost(a.strategy)

    def total_payoff(self, pos: Coord) -> float:
        vals = [self.payoff_in_group(pos, center) for center in self.groups_containing(pos)]
        return sum(vals) / len(vals)

    def imitation_probability(self, pi_i: float, pi_j: float) -> float:
        z = (pi_i - pi_j) / self.kappa
        if z > 700:
            return 0.0
        if z < -700:
            return 1.0
        return 1.0 / (1.0 + exp(z))

def build_demo_world():
    w = ICLWorld()
    w.agents.update({
        (2, 2): Agent(1.00, C),
        (1, 2): Agent(0.90, D),
        (2, 1): Agent(1.20, C),
        (2, 3): Agent(0.80, G),
        (3, 2): Agent(1.10, D),
    })
    return w

if __name__ == "__main__":
    w = build_demo_world()

    assert set(w.neighbors((0, 0))) == {(4,0), (1,0), (0,4), (0,1)}
    assert len(w.group((0,0))) == 5

    print("=== periodic boundary ===")
    print("corner (0,0) neighbors:", w.neighbors((0,0)))

    center = (2,2)
    print("\n=== local five-person group ===")
    print("pos      A     S    u      rank  group_payoff")
    for pos in w.group(center):
        a = w.agents[pos]
        print(f"{str(pos):8} {a.ability:4.2f}  {a.strategy}  {w.utility(pos):5.2f}    {w.rank_in_group(pos, center)}      {w.payoff_in_group(pos, center):5.2f}")

    print("\n=== center agent participates in five overlapping games ===")
    for gc in w.groups_containing(center):
        print(gc, "=>", round(w.payoff_in_group(center, gc), 3))
    print("average payoff of center:", round(w.total_payoff(center), 3))

    print("\n=== Fermi sanity checks ===")
    for pi_i, pi_j in [(-1.0, 0.5), (0.0, 0.0), (0.5, -1.0)]:
        print(pi_i, pi_j, "=>", round(w.imitation_probability(pi_i, pi_j), 6))
