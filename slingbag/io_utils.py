"""Bulk PLY I/O and output confinement for the isolated rapid project."""
import numpy as np
import torch

from .paths import PROJECT_ROOT, ROOT, data_path, resolve_input, resolve_output


def load_ply(path):
    """Read an ASCII point cloud with named x/y/z, pressure and radius fields."""
    path = resolve_input(path)
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        count, properties, in_vertices, ascii_format = None, [], False, False
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "format":
                ascii_format = fields[1] == "ascii"
            elif fields[0] == "element":
                in_vertices = fields[1] == "vertex"
                if in_vertices:
                    count = int(fields[2])
            elif fields[0] == "property" and in_vertices:
                if fields[1] == "list":
                    raise ValueError("List-valued vertex properties are unsupported")
                properties.append(fields[-1])
            elif fields[0] == "end_header":
                break
        else:
            raise ValueError(f"Missing PLY end_header: {path}")
        if not ascii_format or count is None or count < 0:
            raise ValueError("Expected an ASCII PLY with a valid vertex count")
        if count:
            values = np.loadtxt(handle, dtype=np.float32, max_rows=count, ndmin=2)
        else:
            values = np.empty((0, len(properties)), dtype=np.float32)
    if values.shape != (count, len(properties)):
        raise ValueError(f"PLY declared {count} vertices/{len(properties)} fields; read {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite PLY data: {path}")
    try:
        xyz = values[:, [properties.index(k) for k in ("x", "y", "z")]]
        p_name = next(k for k in ("pressure_0", "intensity", "pressure", "p0") if k in properties)
        r_name = next(k for k in ("radius", "sigma") if k in properties)
    except (ValueError, StopIteration) as error:
        raise ValueError("PLY needs x/y/z, pressure_0 (or intensity), and radius (or sigma)") from error
    return tuple(np.ascontiguousarray(value) for value in
                 (xyz, values[:, properties.index(p_name)], values[:, properties.index(r_name)]))


def save_ply(path, xyz, pressure, radius):
    """Write five float32 fields with one bulk GPU-to-CPU transfer."""
    path = resolve_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(xyz, torch.Tensor):
        with torch.no_grad():
            values = torch.cat((xyz.detach().reshape(-1, 3),
                                pressure.detach().reshape(-1, 1),
                                radius.detach().reshape(-1, 1)), dim=1).cpu().numpy()
    else:
        values = np.column_stack((np.asarray(xyz).reshape(-1, 3),
                                  np.asarray(pressure).reshape(-1),
                                  np.asarray(radius).reshape(-1)))
    if not np.isfinite(values).all():
        raise ValueError("Refusing to save non-finite point-cloud parameters")
    header = ("ply\nformat ascii 1.0\nelement vertex " + str(len(values)) +
              "\nproperty float x\nproperty float y\nproperty float z" +
              "\nproperty float pressure_0\nproperty float radius\nend_header")
    np.savetxt(path, values, fmt="%.9g", header=header, comments="")
    return path
