import torch

from bmadx.bmad_torch.track_torch import Beam
from bmadx.bmad_torch.track_torch import TorchLattice

from phase_space_reconstruction.diagnostics import ImageDiagnostic
from phase_space_reconstruction.modeling import NNTransform
from phase_space_reconstruction.modeling import InitialBeam
from phase_space_reconstruction.modeling import PhaseSpaceReconstructionModel
from phase_space_reconstruction.modeling import PhaseSpaceReconstructionModel3D_2screens
from phase_space_reconstruction.utils import split_2screen_dset
from phase_space_reconstruction.virtual.beamlines import quad_tdc_bend


# Load data
data = torch.load("clipped_dset.pt")
train_data, test_data = split_2screen_dset(data)


# Create diagnostic beamlines
p0c = 43.3e06
lattice0 = quad_tdc_bend(p0c=p0c, dipole_on=False)
lattice1 = quad_tdc_bend(p0c=p0c, dipole_on=True)

# Scan over quad strength, tdc on/off and dipole on/off
scan_ids = [0, 2, 4]


# Create diagnostic screens
def make_screen(size: float, pixels: int) -> ImageDiagnostic:
    coords = torch.linspace(-size / 2.0, size / 2.0, pixels)
    bandwidth = (coords[1] - coords[0]) / 2.0
    return ImageDiagnostic(coords, coords, bandwidth)


npix = 300
screen0_size = 30.22e-03 * npix / 700.0
screen1_size = 26.96e-03 * npix / 700.0
screen0 = make_screen(screen0_size, npix)
screen1 = make_screen(screen1_size, npix)



# Track using PhaseSpaceReconstructionModel3D_2screens
# --------------------------------------------------------------------------------------

n_particles = 10_000
nn_transformer = NNTransform(2, 20, output_scale=1.00e-02)
beam = InitialBeam(
    nn_transformer,
    torch.distributions.MultivariateNormal(torch.zeros(6), torch.eye(6)),
    n_particles,
    p0c=torch.tensor(p0c),
)

gpsr_model = PhaseSpaceReconstructionModel3D_2screens(
    lattice0.copy(), 
    lattice1.copy(), 
    screen0, 
    screen1,
    beam,
)

params = train_data.params
params_list = params.reshape(
    params.shape[0] * params.shape[1] * params.shape[2],
    params.shape[3]
)
    

n_imgs_per_param = 1

beam_in = gpsr_model.beam()
particles_in = beam_in.data.clone()
images_pred = gpsr_model.track_and_observe_beam(
    beam_in, params, n_imgs_per_param, scan_ids,
)
images_pred = images_pred.sum(axis=3)
images_pred = images_pred.reshape(
    images_pred.shape[0] * images_pred.shape[1] * images_pred.shape[2],
    images_pred.shape[3],
    images_pred.shape[4],
)
print(images_pred.shape)

for image in images_pred:
    print(image.sum())



# Track using wrapped transform
# --------------------------------------------------------------------------------------

import mentflow as mf
from utils import LatticeFactory
from utils import TorchLatticeTransform
from utils import coords_to_edges


def make_mentflow_diagnostic(screen: ImageDiagnostic) -> mf.diagnostics.Histogram2D:
    bin_coords = [
        screen.bins_x,
        screen.bins_y,
    ]
    bin_edges = [coords_to_edges(c) for c in bin_coords]
    bandwidth = 0.5
    diagnostic = mf.diagnostics.Histogram2D(
        axis=(0, 2), 
        edges=bin_edges,
        bandwidth=(bandwidth, bandwidth)
    )
    return diagnostic


X_in = particles_in.clone()

lattice_factory = LatticeFactory(
    lattice0.copy(),
    lattice1.copy(),
    scan_ids=scan_ids,
    beam_kws=dict(p0c=torch.tensor(p0c)),
)

images_pred_new = torch.zeros(images_pred.shape)
for index, params in enumerate(params_list):
    dipole_on = params[2] > 1.00e-13
    transform = lattice_factory.make_transform(params=params, dipole_on=dipole_on)

    screen = (screen1 if dipole_on else screen0)
    diagnostic = make_mentflow_diagnostic(screen)

    image = diagnostic(transform(X_in.clone()))
    image = image / torch.sum(image)
    images_pred_new[index] = image

    print(image.sum())


for image, image_new in zip(images_pred, images_pred_new):
    print(
        "max : {:0.2e} {:0.2e}".format(
            float(torch.max(image)), 
            float(torch.max(image_new))
        )
    )
    print("diff:", torch.mean(torch.abs(image - image_new)))
    print()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
    axs[0].pcolormesh(image.detach().numpy().T)
    axs[1].pcolormesh(image_new.detach().numpy().T)
    plt.show()