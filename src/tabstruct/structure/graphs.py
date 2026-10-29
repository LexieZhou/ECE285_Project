"""Graph representation shared by all variants, in generation-order (position) space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Graph:
    """`order[j]` is the original column generated at position j; `parents[j]` lists the
    positions (< j) that position j conditions on."""

    order: list[int]
    parents: list[list[int]]
    name: str = ""

    def __post_init__(self) -> None:
        self.validate()

    @property
    def d(self) -> int:
        return len(self.order)

    def in_degrees(self) -> list[int]:
        return [len(p) for p in self.parents]

    def validate(self) -> None:
        if sorted(self.order) != list(range(self.d)):
            raise ValueError(f"order must be a permutation of 0..{self.d - 1}: {self.order}")
        if len(self.parents) != self.d:
            raise ValueError(f"expected {self.d} parent lists, got {len(self.parents)}")
        for j, pa in enumerate(self.parents):
            if len(set(pa)) != len(pa) or any(not 0 <= p < j for p in pa):
                raise ValueError(f"parents of position {j} must be distinct positions < {j}: {pa}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "order": self.order, "parents": self.parents}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        return cls(order=data["order"], parents=data["parents"], name=data.get("name", ""))

    @classmethod
    def full_prefix(cls, order: list[int]) -> Graph:
        return cls(order=order, parents=[list(range(j)) for j in range(len(order))], name="full_prefix")

    @classmethod
    def marginal(cls, order: list[int]) -> Graph:
        return cls(order=order, parents=[[] for _ in order], name="marginal")
