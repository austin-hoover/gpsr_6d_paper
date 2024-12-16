import numpy as np
import torch

from bmadx.bmad_torch.track_torch import Beam
from bmadx.bmad_torch.track_torch import TorchLattice


class TorchLatticeTransform:
    """Transforms NumPy arrays using TorchLattice."""
    def __init__(self, lattice: TorchLattice, beam_kws: dict = None) -> None:
        self.lattice = lattice
        self.beam_kws = beam_kws
        if self.beam_kws is None:
            self.beam_kws = dict()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        beam = Beam(torch.from_numpy(X), **self.beam_kws)
        beam = self.lattice(beam)
        X = torch.vstack([beam.x, beam.px, beam.y, beam.py, beam.z, beam.pz]).T
        X = X.detach().numpy()
        return X


class LatticeFactory:
    """Creates lattices for 3x2 scans."""
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

    def make_transform(self, params: torch.Tensor, dipole_on: bool) -> TorchLatticeTransform:
        lattice = self.make_lattice(params, dipole_on)
        transform = TorchLatticeTransform(lattice, beam_kws=self.beam_kws)
        return transform