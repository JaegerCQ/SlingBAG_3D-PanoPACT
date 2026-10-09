"""Random clouds and an approximate physical-field TV + curvature prior.

No reconstruction, reference image, or measured signal enters this module.
The acoustic model and final Gaussian renderer remain the analytic models.
Here a mass-conserving cloud-in-cell (CIC) deposition and separable Gaussian
filter provide a differentiable, lower-resolution *approximation* for a prior.
Coordinates and widths use metres; the returned field has peak-pressure units.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def random_initialize(num_balls, *, sigma=2.5e-4, pressure=1., seed=1013,
                      bounds=(-.02, .02, -.02, .02, -.02, .02)):
    """Return independent uniform xyz, constant positive P/sigma, and metadata.

    No threshold, support mask, reference field, or data-dependent sampling is
    used. A caller may subsequently fit one global amplitude to training RF.
    """
    if isinstance(num_balls, bool) or not isinstance(num_balls, (int, np.integer)) or num_balls < 1:
        raise ValueError("num_balls must be a positive integer")
    if not all(math.isfinite(float(v)) and v > 0 for v in (sigma, pressure)):
        raise ValueError("Initial sigma and pressure must be finite and positive")
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("bounds must contain six finite min/max values")
    lower, upper = bounds[::2], bounds[1::2]
    if np.any(lower >= upper):
        raise ValueError("Every maximum bound must exceed its minimum")
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(lower, upper, size=(num_balls, 3)).astype(np.float32)
    p = np.full(num_balls, pressure, dtype=np.float32)
    widths = np.full(num_balls, sigma, dtype=np.float32)
    metadata = {
        "initialization": "IID uniform positions over the full box; constant positive peak pressure and width",
        "seed": int(seed), "num_balls": int(num_balls), "bounds_xyz_m": bounds.tolist(),
        "sigma_m": float(sigma), "peak_pressure": float(pressure),
        "reference_or_signal_used": False,
        "integrated_mass_per_source": float((2 * np.pi) ** 1.5 * pressure * sigma ** 3),
    }
    return xyz, p, widths, metadata


class GaussianGridField(nn.Module):
    """Differentiable physical pressure field for a common *fixed* source sigma.

    ``dims`` are XYZ counts; ``origin`` is the first voxel centre, in metres.
    ``forward(xyz, pressure)`` differentiates with respect to both inputs.
    ``cache = prepare_deposition(xyz)`` detaches the eight corner weights;
    ``forward(xyz, pressure, cache=cache)`` then differentiates pressure only.
    Drop this cache when positions change, including when enabling position
    optimization. Widths are fixed by this constructor, not silently ignored.

    A halo accommodates all eight deposition corners and the finite Gaussian
    filter. Only the final pressure field is cropped to the requested volume;
    physical mass outside that volume is not moved to its boundary.

    CIC adds phase-dependent variance f*(1-f)*spacing^2 per axis. The optional
    compensation subtracts its random-phase mean spacing^2/6 from the filter
    variance. This corrects an average second moment, not the exact pointwise
    Gaussian or every source's width. Final exports should use the analytic
    voxelizer. To support several fixed sigma bins, create one instance per
    width, deposit each bin's sources, and sum their physical fields; do not
    pass nonuniform widths through a single instance.
    """

    def __init__(self, sigma=2.5e-4, dims=(160, 160, 160), spacing=2.5e-4,
                 origin=(-.019875, -.019875, -.019875), cutoff=4., compensate_cic=True):
        super().__init__()
        dims = tuple(dims)
        if len(dims) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 3 for v in dims):
            raise ValueError("dims must contain three integer counts >= 3")
        origin = tuple(float(v) for v in origin)
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError("origin must contain three finite voxel-centre coordinates")
        if not all(math.isfinite(float(v)) and v > 0 for v in (sigma, spacing, cutoff)):
            raise ValueError("sigma, spacing and cutoff must be finite and positive")
        self.sigma = float(sigma)
        self.dims = tuple(int(v) for v in dims)
        self.spacing = float(spacing)
        self.origin = origin
        self.cutoff = float(cutoff)
        self.compensate_cic = bool(compensate_cic)
        variance = self.sigma ** 2 - (self.spacing ** 2 / 6 if self.compensate_cic else 0.)
        if variance <= 0:
            raise ValueError("CIC compensation requires sigma > spacing/sqrt(6)")
        self.filter_sigma = math.sqrt(variance)
        self.kernel_radius = max(1, math.ceil(self.cutoff * self.filter_sigma / self.spacing))
        self.halo = self.kernel_radius + 1
        self.padded_dims = tuple(v + 2 * self.halo for v in self.dims)
        self.padded_origin = tuple(v - self.halo * self.spacing for v in self.origin)
        offsets = torch.arange(-self.kernel_radius, self.kernel_radius + 1, dtype=torch.float64)
        kernel = torch.exp(-.5 * (offsets * self.spacing / self.filter_sigma).square())
        kernel /= kernel.sum()
        self.register_buffer("kernel", kernel.float(), persistent=True)
        self.mass_to_density = (2 * math.pi) ** 1.5 * (self.sigma / self.spacing) ** 3
        self._signature = (self.dims, self.spacing, self.origin, self.halo)
        self.metadata = {
            "model": "Common fixed-sigma CIC mass deposition plus separable normalized Gaussian smoothing",
            "sigma_m": self.sigma, "filter_sigma_m": self.filter_sigma,
            "dims_xyz": list(self.dims), "axis_order": "ZYX", "spacing_m": self.spacing,
            "voxel_origin_xyz_m": list(self.origin), "halo_voxels": self.halo,
            "padded_dims_xyz": list(self.padded_dims), "kernel_radius_voxels": self.kernel_radius,
            "requested_axis_cutoff_sigma": self.cutoff,
            "kernel_support": "Each axis uses integer offsets through ceil(cutoff*filter_sigma/spacing), inclusive",
            "cic_variance_compensation": self.compensate_cic,
            "compensated_variance_m2": self.spacing ** 2 / 6 if self.compensate_cic else 0.,
            "mass_rule": "q=(2*pi)^1.5*P*sigma^3; deposit q/spacing^3; normalize each 1D Gaussian kernel to sum 1",
            "field_units": "Same pressure units as source peak P",
            "approximation": "CIC compensation matches the random-phase mean second moment only, not exact Gaussians",
            "boundary": "Deposit and smooth on halo, then crop ROI; Gaussian pressure outside ROI is excluded",
            "sigma_policy": "Common fixed sigma per instance; distinct fixed-width bins require separate fields summed together",
        }

    def _deposition(self, xyz, detach):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype not in (torch.float32, torch.float64):
            raise ValueError("xyz must be floating [N,3] in metres")
        positions = xyz.detach() if detach else xyz
        origin = positions.new_tensor(self.padded_origin)
        grid = (positions - origin) / self.spacing
        lower_float = torch.floor(grid).detach()
        lower = lower_float.to(torch.long)
        fraction = grid - lower_float
        sizes = lower.new_tensor(self.padded_dims)
        # Fail instead of dropping or clipping out-of-grid corner mass. This
        # one scalar check synchronizes CUDA; the fixed-geometry cache pays it
        # once. Caller movement constraints should retain the physical box.
        if not bool(torch.isfinite(positions).all() & (lower >= 0).all() & (lower + 1 < sizes).all()):
            raise ValueError("Source positions must be finite and inside the padded deposition grid")
        indices, weights = [], []
        nx, ny, _ = self.padded_dims
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            ix, iy, iz = lower[:, 0] + dx, lower[:, 1] + dy, lower[:, 2] + dz
            indices.append((iz * ny + iy) * nx + ix)
            weight = (fraction[:, 0] if dx else 1 - fraction[:, 0])
            weight = weight * (fraction[:, 1] if dy else 1 - fraction[:, 1])
            weight = weight * (fraction[:, 2] if dz else 1 - fraction[:, 2])
            weights.append(weight)
        return {"indices": torch.stack(indices), "weights": torch.stack(weights),
                "num_sources": len(xyz), "grid_signature": self._signature,
                "positions_detached": bool(detach)}

    def prepare_deposition(self, xyz):
        """Cache eight indices/weights per source for an unchanged position set.

        The caller owns cache invalidation. This method always detaches xyz;
        using the returned cache intentionally disables position gradients.
        """
        return self._deposition(xyz, detach=True)

    def forward(self, xyz, pressure, *, cache=None):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or pressure.shape != (len(xyz),):
            raise ValueError("Expected xyz[N,3] and pressure[N]")
        if xyz.dtype not in (torch.float32, torch.float64) or pressure.dtype != xyz.dtype or pressure.device != xyz.device:
            raise ValueError("xyz and pressure must share float32/float64 dtype and device")
        deposition = self._deposition(xyz, detach=False) if cache is None else cache
        if deposition["num_sources"] != len(xyz) or deposition["grid_signature"] != self._signature:
            raise ValueError("Deposition cache does not match this source count/grid")
        indices, weights = deposition["indices"], deposition["weights"]
        if indices.device != xyz.device or weights.device != xyz.device or weights.dtype != xyz.dtype:
            raise ValueError("Deposition cache must share the sources' device and dtype")
        # Combining mass and voxel-volume factors avoids tiny SI intermediates.
        values = weights * (pressure * self.mass_to_density)[None, :]
        density = pressure.new_zeros(math.prod(self.padded_dims)).scatter_add(
            0, indices.reshape(-1), values.reshape(-1))
        nx, ny, nz = self.padded_dims
        field = density.reshape(1, 1, nz, ny, nx)
        kernel = self.kernel.to(device=xyz.device, dtype=xyz.dtype)
        r = self.kernel_radius
        field = F.conv3d(field, kernel.reshape(1, 1, 1, 1, -1), padding=(0, 0, r))
        field = F.conv3d(field, kernel.reshape(1, 1, 1, -1, 1), padding=(0, r, 0))
        field = F.conv3d(field, kernel.reshape(1, 1, -1, 1, 1), padding=(r, 0, 0))
        h = self.halo
        nx, ny, nz = self.dims
        return field[0, 0, h:h + nz, h:h + ny, h:h + nx]


def spatial_penalty(field, spacing=2.5e-4, reference_spacing=1e-4,
                    amplitude_scale=100., curvature_weight=2., epsilon=1e-8):
    """Return ``loss``, ``tv`` and ``curvature`` for a pressure field in ZYX.

    u=field/amplitude_scale, first differences use reference_spacing/spacing,
    and second differences its square. ``tv`` is the mean isotropic gradient
    norm; ``curvature`` is the mean symmetric-Hessian Frobenius norm (including
    twice each mixed derivative's square). This is TV + curvature, not TGV.
    No artificial image-edge-to-zero derivatives are added. Smooth norms equal
    zero on an exactly constant field and have finite gradients there.
    """
    if field.ndim != 3 or min(field.shape) < 3 or not field.is_floating_point():
        raise ValueError("field must be a floating ZYX volume with every dimension >= 3")
    if not all(math.isfinite(float(v)) and v > 0 for v in (spacing, reference_spacing, amplitude_scale, epsilon)):
        raise ValueError("Spacings, amplitude_scale and epsilon must be finite and positive")
    if not math.isfinite(float(curvature_weight)) or curvature_weight < 0:
        raise ValueError("curvature_weight must be finite and nonnegative")
    u = field / amplitude_scale
    ratio = reference_spacing / spacing
    common = u[1:, 1:, 1:]
    dz = (common - u[:-1, 1:, 1:]) * ratio
    dy = (common - u[1:, :-1, 1:]) * ratio
    dx = (common - u[1:, 1:, :-1]) * ratio
    tv = (torch.sqrt(dx.square() + dy.square() + dz.square() + epsilon ** 2) - epsilon).mean()
    center = u[1:-1, 1:-1, 1:-1]
    hxx = (u[1:-1, 1:-1, 2:] - 2 * center + u[1:-1, 1:-1, :-2]) * ratio ** 2
    hyy = (u[1:-1, 2:, 1:-1] - 2 * center + u[1:-1, :-2, 1:-1]) * ratio ** 2
    hzz = (u[2:, 1:-1, 1:-1] - 2 * center + u[:-2, 1:-1, 1:-1]) * ratio ** 2
    hxy = (u[1:-1, 2:, 2:] - u[1:-1, 2:, :-2] - u[1:-1, :-2, 2:] + u[1:-1, :-2, :-2]) * (.25 * ratio ** 2)
    hxz = (u[2:, 1:-1, 2:] - u[2:, 1:-1, :-2] - u[:-2, 1:-1, 2:] + u[:-2, 1:-1, :-2]) * (.25 * ratio ** 2)
    hyz = (u[2:, 2:, 1:-1] - u[2:, :-2, 1:-1] - u[:-2, 2:, 1:-1] + u[:-2, :-2, 1:-1]) * (.25 * ratio ** 2)
    hessian2 = hxx.square() + hyy.square() + hzz.square() + 2 * (hxy.square() + hxz.square() + hyz.square())
    curvature = (torch.sqrt(hessian2 + epsilon ** 2) - epsilon).mean()
    return {"loss": tv + curvature_weight * curvature, "tv": tv, "curvature": curvature}


def _cpu_checks():
    """Reproducible small CPU fixtures; never initializes or uses CUDA."""
    torch.set_num_threads(1)
    a = random_initialize(11)
    b = random_initialize(11)
    for first, second in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(first, second)
    # Dimensionless fixture with the same sigma/spacing ratio as production.
    prior = GaussianGridField(sigma=1., dims=(7, 8, 9), spacing=1., origin=(-3., -3.5, -4.)).double()
    xyz = torch.tensor([[.13, -.27, .31], [-.48, .34, -.21]], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([2., .7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(prior, (xyz, pressure), eps=1e-6, atol=1e-5, rtol=1e-4)
    cache = prior.prepare_deposition(xyz)
    torch.testing.assert_close(cache['weights'].sum(0), torch.ones(2, dtype=torch.float64))
    dynamic = prior(xyz, pressure)
    cached = prior(xyz, pressure, cache=cache)
    torch.testing.assert_close(dynamic, cached, rtol=0., atol=0.)
    gd = torch.autograd.grad(dynamic.square().sum(), pressure, retain_graph=True)[0]
    gc = torch.autograd.grad(cached.square().sum(), pressure)[0]
    torch.testing.assert_close(gd, gc, rtol=0., atol=0.)
    # Verify pressure-to-integrated-mass units far enough from cropped edges.
    spacious = GaussianGridField(sigma=1., dims=(17, 17, 17), spacing=1., origin=(-8.,) * 3).double()
    field = spacious(xyz, pressure)
    expected = pressure.sum() * (2 * math.pi) ** 1.5
    torch.testing.assert_close(field.sum(), expected, rtol=2e-7, atol=1e-10)
    # Sources at physical ROI corners retain all CIC mass on the halo, while
    # only their physically inside portion remains after cropping.
    boundary = torch.tensor([[-3.5, -4., -4.5], [3.5, 4., 4.5]], dtype=torch.float64)
    edge_cache = prior.prepare_deposition(boundary)
    torch.testing.assert_close(edge_cache['weights'].sum(0), torch.ones(2, dtype=torch.float64))
    assert bool(torch.isfinite(prior(boundary, pressure)).all())
    constant = torch.ones((5, 5, 5), dtype=torch.float64, requires_grad=True)
    zero = spatial_penalty(constant)['loss']
    assert zero == 0 and torch.count_nonzero(torch.autograd.grad(zero, constant)[0]) == 0
    rng = torch.Generator().manual_seed(17)
    small = torch.rand((3, 4, 3), generator=rng, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u: spatial_penalty(u)['loss'], (small,), eps=1e-6, atol=1e-7, rtol=1e-4)
    # A linear ramp has the same dimensionless physical gradient after a
    # change of grid spacing; its Hessian vanishes on either grid.
    for spacing in (.1, .25):
        coords = torch.arange(5, dtype=torch.float64) * spacing
        ramp = coords[None, None, :].expand(5, 5, 5)
        reg = spatial_penalty(ramp, spacing=spacing, reference_spacing=.1, amplitude_scale=1.)
        torch.testing.assert_close(reg['tv'], torch.tensor(.1 - 1e-8, dtype=torch.float64), atol=1e-14, rtol=1e-12)
        assert reg['curvature'] < 1e-14
    print("CPU PASS: reproducible random cloud, xyz/P finite differences, cached/dynamic equality, "
          "mass units and boundary deposition, TV+curvature gradients, and spacing scaling.")


if __name__ == '__main__':
    _cpu_checks()
