"""Small CPU checks: run with python -m unittest discover -s tests -v."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from slingbag.fit_random_response import check_normal_equations
from slingbag.random_prior import GaussianGridField, spatial_penalty
from slingbag.rapid_forward import gaussian_forward


class NumericalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_continuous_forward_gradients(self):
        sensors = torch.tensor([[7.1, 1.2, 3.4], [-6.3, 2.1, 4.4], [2.7, -7.7, 4.6]],
                               dtype=torch.float64) * 1e-3
        # Millimetre parameter units keep finite-difference tolerances readable.
        xyz_mm = torch.tensor([[.12, .23, -.14], [-.22, .11, .08],
                               [.30, -.25, .12], [-.10, -.05, .22]],
                              dtype=torch.float64, requires_grad=True)
        pressure = torch.tensor([.8, 1.1, .7, 1.4], dtype=torch.float64, requires_grad=True)
        sigma_mm = torch.tensor([.18, .23, .29, .34], dtype=torch.float64, requires_grad=True)

        def forward(x, p, s):
            return gaussian_forward(sensors, x * 1e-3, p, s * 1e-3,
                                    num_times=512, cutoff=0., backend='torch')

        self.assertTrue(torch.autograd.gradcheck(forward, (xyz_mm, pressure, sigma_mm),
                        eps=1e-6, atol=1e-7, rtol=1e-4, fast_mode=True))

    def test_empty_source_gradients(self):
        sensors = torch.tensor([[.005, 0., 0.]], dtype=torch.float64)
        xyz = torch.empty((0, 3), dtype=torch.float64, requires_grad=True)
        pressure = torch.empty(0, dtype=torch.float64, requires_grad=True)
        sigma = torch.empty(0, dtype=torch.float64, requires_grad=True)
        signal = gaussian_forward(sensors, xyz, pressure, sigma, num_times=16, backend='torch')
        self.assertEqual(signal.shape, (1, 16))
        self.assertEqual(int(torch.count_nonzero(signal)), 0)
        gradients = torch.autograd.grad(signal.sum(), (xyz, pressure, sigma))
        self.assertTrue(all(value.numel() == 0 for value in gradients))

    def test_random_grid_position_gradients_and_cache(self):
        prior = GaussianGridField(sigma=2.5e-4, dims=(7, 8, 9), spacing=2.5e-4,
                                  origin=(-.00075, -.000875, -.001)).double()
        xyz_mm = torch.tensor([[.13, -.27, .31], [-.48, .34, -.21]],
                              dtype=torch.float64, requires_grad=True)
        pressure = torch.tensor([2., .7], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x, p: prior(x * 1e-3, p),
                        (xyz_mm, pressure), eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True))
        xyz = xyz_mm * 1e-3
        dynamic = prior(xyz, pressure)
        cache = prior.prepare_deposition(xyz)
        cached = prior(xyz, pressure, cache=cache)
        torch.testing.assert_close(cached, dynamic, rtol=0., atol=0.)
        grad_dynamic = torch.autograd.grad(dynamic.square().sum(), pressure, retain_graph=True)[0]
        grad_cached = torch.autograd.grad(cached.square().sum(), pressure)[0]
        torch.testing.assert_close(grad_cached, grad_dynamic, rtol=0., atol=0.)
        loss = spatial_penalty(dynamic, spacing=2.5e-4)['loss']
        gradient = torch.autograd.grad(loss, xyz_mm)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_fir_against_dense_design(self):
        report = check_normal_equations()
        self.assertEqual(report['status'], 'passed')
        self.assertGreaterEqual(len(report['cases']), 3)


if __name__ == '__main__':
    unittest.main()
