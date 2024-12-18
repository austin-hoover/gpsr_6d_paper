import argparse
import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import downscale_local_mean

from bmadx.bmad_torch.track_torch import Beam
from bmadx.bmad_torch.track_torch import TorchLattice

from phase_space_reconstruction.utils import split_2screen_dset
from phase_space_reconstruction.virtual.beamlines import quad_tdc_bend

import mentflow as mf
from utils import LatticeFactory
from utils import TorchLatticeTransform
from utils import coords_to_edges    


# Setup
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--downscale", type=int, default=1)
parser.add_argument("--nsamp", type=int, default=80_000)
parser.add_argument("--iterations", type=int, default=3000)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--diag-bandwidth", type=float, default=0.5)
parser.add_argument("--nn-width", type=int, default=50)
parser.add_argument("--nn-depth", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()


timestamp = time.strftime("%y%m%d%H%M%S")
output_dir = os.path.join("outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)


device = args.device


def send(x: torch.Tensor) -> torch.Tensor:
    x = x.type(torch.float32).to(device)
    return x


def grab(x: torch.Tensor) -> np.array:
    return x.detach().cpu().numpy()



# Helper functions
# --------------------------------------------------------------------------------------

class NNTransform(torch.nn.Module):
    def __init__(
        self,
        ndim: int = 6,
        depth: int = 2,
        width: int = 32,
        dropout: float = 0.0,
        scale=0.01,
    ) -> None:
        super().__init__()

        activation = torch.nn.Tanh()

        layer_sequence = []
        
        layer_sequence.append(torch.nn.Linear(ndim, width))
        layer_sequence.append(activation)
        for _ in range(depth):
            layer_sequence.append(torch.nn.Linear(width, width))
            layer_sequence.append(torch.nn.Dropout(dropout))
            layer_sequence.append(activation)
        layer_sequence.append(torch.nn.Linear(width, ndim))

        self.stack = torch.nn.Sequential(*layer_sequence)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        X = self.stack(Z)
        X = X * self.scale
        return X


class Distribution(torch.nn.Module):
    def __init__(self, ndim: int, transform: torch.nn.Module, device=None) -> None:
        super().__init__()
        self.ndim = ndim
        self.transform = transform
        self.device = device
        
    def sample(self, size: int) -> torch.Tensor:
        Z = torch.randn((size, self.ndim))
        Z = Z.float().to(self.device)
        X = self.transform(Z)
        return X
        

class ReconstructionModel:
    def __init__(
        self,
        distribution: Distribution,
        transforms: list[Callable],
        diagnostics: list[Callable],
        projections: list[torch.Tensor],
    ) -> None:
        self.distribution = distribution
        self.transforms = transforms
        self.diagnostics = diagnostics
        self.projections = projections
        self.n_meas = len(projections)

    def sample(self, size: int) -> torch.Tensor:
        return self.distribution.sample(size)
    
    def simulate_data(self, X: torch.Tensor) -> list[torch.Tensor]:
        projections = []
        for transform, diagnostic in zip(self.transforms, self.diagnostics):
            projection = diagnostic(transform(X))
            projections.append(projection)
        return projections

    def loss_mean(self, batch_size) -> torch.Tensor:
        X = self.sample(batch_size)
        loss = 0.0 
        loss += torch.mean(torch.abs(torch.mean(X, axis=0)))
        
        loss += torch.abs(torch.std(X[:, 0]) - 0.0002)
        loss += torch.abs(torch.std(X[:, 1]) - 0.0002)
        loss += torch.abs(torch.std(X[:, 2]) - 0.0002)
        loss += torch.abs(torch.std(X[:, 3]) - 0.0002)
        loss += torch.abs(torch.std(X[:, 4]) - 0.0001)
        loss += torch.abs(torch.std(X[:, 5]) - 0.0001)
        return loss
            
    def loss(self, batch_size: int) -> torch.Tensor:
        X = self.sample(batch_size)

        loss = 0.0
        for transform, diagnostic, projection_meas in zip(self.transforms, self.diagnostics, self.projections):
            U = transform(X)
            projection_pred = diagnostic(U)
            
            ymeas = projection_pred / torch.sum(projection_pred)
            ypred = projection_meas / torch.sum(projection_meas)
            
            loss = loss + torch.mean(torch.abs(ymeas - ypred))
        return loss


def plot_proj(projections_pred: list[np.ndarray], projections_meas: list[np.ndarray]):
    n = len(projections_pred)
    ncols = min(10, n)
    nrows = int(np.ceil(n / ncols))
    nrows = nrows * 2
            
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(ncols * 1.0, nrows * 1.0),
        gridspec_kw=dict(hspace=0, wspace=0),
    )
    
    index = 0
    for i in range(0, nrows, 2):
        for j in range(ncols):
            axs[i, j].pcolormesh(projections_pred[index].T)
            axs[i + 1, j].pcolormesh(projections_meas[index].T)
            index += 1
            
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for loc in ax.spines:
            ax.spines[loc].set_visible(False)
            
    return fig, axs


# Load data
# --------------------------------------------------------------------------------------

data = torch.load("clipped_dset.pt")
train_data, test_data = split_2screen_dset(data)

params = train_data.params
params = send(params)
n_images = params.shape[0] * params.shape[1] * params.shape[2]
params_list = params.reshape(n_images, params.shape[-1])

projections = train_data.images.clone()
projections = projections.sum(axis=3)  # average over multi-shot images
projections = projections.reshape(n_images, projections.shape[-2], projections.shape[-1])
projections = [send(projection) for projection in projections]

downscale = args.downscale
for i in range(len(projections)):
    if downscale > 1:
        projections[i] = downscale_local_mean(grab(projections[i]), (downscale, downscale))
        projections[i] = torch.from_numpy(projections[i])

for i in range(len(projections)):
    projections[i] = send(projections[i])
        

# Create transforms
# --------------------------------------------------------------------------------------

p0c = 43.3e06
lattice0 = quad_tdc_bend(p0c=p0c, dipole_on=False)
lattice1 = quad_tdc_bend(p0c=p0c, dipole_on=True)

for lattice in [lattice0, lattice1]:
    for i, element in enumerate(lattice.elements):
        element.type(torch.float32).to(device)

lattice_factory = LatticeFactory(
    lattice0.copy(),
    lattice1.copy(),
    scan_ids=[0, 2, 4],
    beam_kws=dict(p0c=torch.tensor(p0c)),
)

transforms = []
for index, params in enumerate(params_list):
    dipole_on = params[2] > 1.00e-13
    transform = lattice_factory.make_transform(params=params, dipole_on=dipole_on)
    transform.to(device)
    transforms.append(transform)
    

# Create diagnostics
# --------------------------------------------------------------------------------------

npix = projections[0].shape[0]
screen0_size = 30.22e-03 * npix / 700.0
screen1_size = 26.96e-03 * npix / 700.0

diagnostics = []
for index, params in enumerate(params_list):
    dipole_on = params[2] > 1.00e-13
    
    size = None
    if dipole_on:
        size = screen1_size
    else:
        size = screen0_size
        
    bin_coords = torch.linspace(-size / 2.0, size / 2.0, npix)
    bin_edges = coords_to_edges(bin_coords)
    bin_edges = send(bin_edges)

    diagnostic = mf.diagnostics.Histogram2D(
        axis=(0, 2), 
        edges=(bin_edges, bin_edges),
        bandwidth=(args.diag_bandwidth, args.diag_bandwidth),
    )
    diagnostic.to(device)
    diagnostics.append(diagnostic)


# Reconstruct distribution
# --------------------------------------------------------------------------------------

ndim = 6
batch_size = args.nsamp

transformer = NNTransform(
    ndim=ndim, 
    depth=args.nn_depth, 
    width=args.nn_width, 
    scale=0.01
)
transformer.to(device)

distribution = Distribution(ndim=ndim, transform=transformer)

rec_model = ReconstructionModel(distribution, transforms, diagnostics, projections)

optimizer = torch.optim.Adam(distribution.parameters(), lr=args.lr)

distribution.train()
for iteration in range(args.iterations):
    if iteration < 100:
        loss = rec_model.loss_mean(batch_size)
    else:
        loss = rec_model.loss(batch_size)
        
    print(iteration, loss)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if iteration % 10 == 0:
        distribution.eval()

        X_pred = rec_model.sample(args.nsamp)
        projections_pred = rec_model.simulate_data(X_pred)
        projections_meas = rec_model.projections

        projections_pred = [grab(p) for p in projections_pred]
        projections_meas = [grab(p) for p in projections_meas]
        
        fig, axs = plot_proj(projections_pred, projections_meas)
        filename = f"fig_proj_{iteration:04.0f}.png"
        filename = os.path.join(output_dir, filename)
        plt.savefig(filename, dpi=250)
        plt.close()
        
        distribution.train()






        