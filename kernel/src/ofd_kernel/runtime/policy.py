from __future__ import annotations

from dataclasses import dataclass
from ..contracts import Variant


@dataclass(frozen=True)
class LayerPolicy:
    variant: Variant

    @property
    def rotate_k(self) -> bool:
        return self.variant.rotate_k

    @property
    def rotate_v(self) -> bool:
        return self.variant.rotate_v


class StaticPolicy:
    """Model-level static per-layer policy.

    This is intentionally tiny: calibration is offline; the decode hot path
    only performs a table lookup.
    """

    def __init__(self, variants: list[str | Variant] | tuple[str | Variant, ...]):
        self._layers = tuple(
            LayerPolicy(v if isinstance(v, Variant) else Variant(v))
            for v in variants
        )

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, layer_idx: int) -> LayerPolicy:
        return self._layers[layer_idx]

    def variants(self) -> tuple[Variant, ...]:
        return tuple(x.variant for x in self._layers)

    @classmethod
    def dual_cutoff(
        cls,
        n_layers: int,
        k_cut: int,
        v_cut: int,
    ) -> "StaticPolicy":
        if not (0 <= k_cut <= n_layers and 0 <= v_cut <= n_layers):
            raise ValueError("cutoffs must be in [0, n_layers]")

        out: list[Variant] = []
        for layer in range(n_layers):
            rk = layer < k_cut
            rv = layer < v_cut
            if rk and rv:
                out.append(Variant.KV)
            elif rk:
                out.append(Variant.K)
            elif rv:
                out.append(Variant.V)
            else:
                out.append(Variant.P)
        return cls(out)
