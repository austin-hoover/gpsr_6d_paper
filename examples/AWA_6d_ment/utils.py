from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from bmadx.bmad_torch.track_torch import Beam
from bmadx.bmad_torch.track_torch import TorchLattice


def coords_to_edges(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[1] - coords[0]
    edges = torch.zeros(len(coords) + 1)
    edges[:-1] = coords - 0.5 * delta
    edges[-1] = edges[-2] + delta
    return edges


class WrappedNumpyTransform:
    """NumPy -> PyTorch -> Numpy."""
    def __init__(self, function: Callable) -> np.ndarray:
        self.function = function

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        x_out = self.function(torch.from_numpy(x), *args, **kwargs)
        x_out = x_out.detach().numpy()
        return x_out
        

class TorchLatticeTransform(torch.nn.Module):
    def __init__(self, lattice: TorchLattice, beam_kws: dict = None) -> None:
        super().__init__()
        self.lattice = lattice
        self.beam_kws = beam_kws
        if self.beam_kws is None:
            self.beam_kws = dict()

        self.base_beam = Beam(torch.zeros((1, 6)), **self.beam_kws)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        beam = Beam(X, self.base_beam.p0c, self.base_beam.s, self.base_beam.mc2)
        lattice = deepcopy(self.lattice)
        beam_out = lattice(beam)
        U = torch.vstack([
            beam_out.x,
            beam_out.px,
            beam_out.y,
            beam_out.py,
            beam_out.z,
            beam_out.pz,
        ])
        U = U.T
        return U
        

class LatticeFactory:
    def __init__(
        self,
        lattice0: TorchLattice,
        lattice1: TorchLattice,
        scan_ids: list[int],
        beam_kws: dict = None,
    ) -> None:
        self.lattice0 = lattice0
        self.lattice1 = lattice1
        self.scan_ids = scan_ids
        self.beam_kws = beam_kws

    def make_lattice(self, params: torch.Tensor, dipole_on: bool) -> TorchLattice:
        ids = self.scan_ids

        lattice = None
        if dipole_on:
            lattice = self.lattice1.copy()
            lattice.elements[ids[0]].K1.data = params[0]
            lattice.elements[ids[1]].VOLTAGE.data = params[1]

            G = params[2]
            l_bend = 0.3018
            theta = torch.arcsin(l_bend * G)  # AWA parameters
            l_arc = theta / G
            lattice.elements[ids[2]].G.data = G
            lattice.elements[ids[2]].L.data = l_arc
            lattice.elements[ids[2]].E2.data = theta
            lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        else:
            lattice = self.lattice0.copy()
            lattice.elements[ids[0]].K1.data = params[0]
            lattice.elements[ids[1]].VOLTAGE.data = params[1]

            G = params[2]
            l_bend = 0.3018
            theta = torch.arcsin(l_bend * G)  # AWA parameters
            l_arc = theta / G
            lattice.elements[ids[2]].G.data = G
            lattice.elements[ids[2]].L.data = l_arc
            lattice.elements[ids[2]].E2.data = theta
            lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        return lattice

    def make_transform(self, params: torch.Tensor, dipole_on: bool) -> Callable:
        lattice = self.make_lattice(params, dipole_on)
        transform = TorchLatticeTransform(lattice, beam_kws=self.beam_kws)
        return transform

    def make_transform_numpy(self, **kws) -> Callable:
        transform = self.make_transform(**kws)
        transform = WrappedNumpyTransform(transform)
        return transform