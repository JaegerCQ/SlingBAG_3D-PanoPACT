# Attribution and scope

This project packages a local research variant derived from the workflow of
[SlingBAG by JaegerCQ](https://github.com/JaegerCQ/SlingBAG). The upstream MIT
license and copyright notice are retained verbatim in [LICENSE](LICENSE).

The continuous analytic Gaussian photoacoustic waveform follows the formulation
in the local PA-SfM-NC `forward_propagation.py`. The implementation here evaluates
the waveform directly with an FP32 CUDA extension and includes a PyTorch reference.
The local development lineage is SlingBAG → SlingBAG_rapid → the random-only
pose000 experiments → this standalone package. It is an algorithm variant and
does not claim to reproduce the unmodified upstream training algorithm.

The supplied experimental frame originated from `processed_signal_group03_pose000.txt`
and `sensor_location_group03_pose000.txt`. Source checksums and the packaging
transformations are in [provenance/portable_manifest.json](provenance/portable_manifest.json).
No separate experimental-data license was supplied with this frame; the retained
MIT code license should not be read as a new license grant for the dataset.

The historical comparison figure is included for documentation. The PA/DAS/hybrid
reference volumes, original experimental archives, and machine-specific binaries
are not runtime dependencies of this package.
