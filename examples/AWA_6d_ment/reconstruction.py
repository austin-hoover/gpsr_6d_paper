import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from phase_space_reconstruction.virtual.beamlines import quad_tdc_bend
from phase_space_reconstruction.virtual.scans import run_3d_scan_2screens
from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.utils import split_2screen_dset
from phase_space_reconstruction.train import train_3d_scan_2screens

from bmadx.distgen_utils import create_beam
from bmadx.plot import plot_projections
from bmadx.constants import PI

from analysis_scripts import plot_3d_scan_data_2screens, plot_3d_scan_data_2screens_contour, create_clipped_dset


# Setup
timestamp = time.strftime("%y%m%d%H%M%S")
output_dir = os.path.join("outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)


# Load data
dset = torch.load('clipped_dset.pt')
train_dset, test_dset = split_2screen_dset(dset)


# Define diagnostics lattice parameters
p0c = 43.3e6
lattice0 = quad_tdc_bend(p0c=p0c, dipole_on=False)
lattice1 = quad_tdc_bend(p0c=p0c, dipole_on=True)

# Scan over quad strength, tdc on/off and dipole on/off
scan_ids = [0, 2, 4] 

# create 2 diagnostic screens: 
def create_screen(size, pixels):
    bins = torch.linspace(-size/2, size/2, pixels)
    bandwidth = (bins[1]-bins[0]) / 2
    return ImageDiagnostic(bins, bins, bandwidth)

width = dset.images.shape[-1]
screen0 = create_screen(30.22*1e-3*width/700, width)
screen1 = create_screen(26.96*1e-3*width/700, width)


# Device selection:
DEVICE = torch.device(device)
print(f"Using device: {DEVICE}")

params = train_dset.params.type(torch.float32).to(DEVICE)
imgs = train_dset.images.type(torch.float32).to(DEVICE)
n_imgs_per_param = imgs.shape[-3]

train_dset_device = ImageDataset3D(params, imgs)
train_dataloader = DataLoader(
    train_dset_device, batch_size=batch_size, shuffle=True
)


for lattice in [lattice0, lattice1]:
    for i, element in enumerate(lattice.elements):
        element.type(torch.float32).to(DEVICE)
        

# Create phase space reconstruction model
base_dist = torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6))
nn_transformer = NNTransform(2, 20, output_scale=1e-2)
nn_transformer.to(DEVICE)

nn_beam = InitialBeam(
    nn_transformer,
    base_dist,
    n_particles,
    p0c=torch.tensor(p0c),
    device=DEVICE,
)
model = PhaseSpaceReconstructionModel3D_2screens(
    lattice0.copy(), lattice1.copy(), screen0, screen1, nn_beam
)
model = model.to(DEVICE)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_fn = MAELoss()

for i in range(n_epochs + 1):
    for elem in train_dataloader:
        params_i, target_images = elem[0], elem[1]
        params_i.type(torch.float32).to(DEVICE)
        target_images.type(torch.float32).to(DEVICE)
        optimizer.zero_grad()
        output = model(params_i, n_imgs_per_param, ids)
        loss = loss_fn(output, target_images)
        loss.backward()
        optimizer.step()

    print(i, loss)

    # dump current particle distribution to file
    if i % 100 == 0:
        model.eval()
        with torch.no_grad():
            predicted_beam = model.beam.forward().detach_clone()
            fig, ax = plot_projections(pred_beam, bins=100)
            filename = f"fig_{iteration:04.0f}.png"
            filename = os.path.join(output_dir, filename)
            plt.savefig(filename, dpi=250)
            plt.close("all")
        model.train()      


