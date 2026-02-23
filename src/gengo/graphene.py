"""
Pristine graphene generation with periodic boundary conditions
and multi-layer (AB stacking) support.

Coordinate system (same as GOPY):
    - X axis: along armchair direction of the hexagonal lattice
    - Y axis: along zigzag direction
    - Z axis: out-of-plane (stacking direction for multi-layer)

Crystallography:
    Graphene has a hexagonal lattice (space group P6_3/mmc for flat graphite,
    P6_3mc for buckled graphene). The primitive hexagonal unit cell has
    a = 2.461 A, c = 6.708 A (experimental, Z = 4).

    This module uses an **orthogonal (rectangular) supercell** representation
    with 4 atoms per cell, which is equivalent to the hexagonal lattice but
    tiles in Cartesian x/y without shear. The cell vectors are:

        a_orth = 2.456 A  (along X / armchair)
        b_orth = 4.254 A  (along Y / zigzag)

    These derive from the C-C bond length of 1.418 A (GOPY value):
        a_orth = CC_BOND * sqrt(3)       = 1.418 * 1.7321 = 2.456 A
        b_orth = 3 * CC_BOND             = 3 * 1.418      = 4.254 A

    The experimental C-C bond is 1.421 A (a_hex = 2.461 A). The difference
    is ~0.2% and does not affect the GO placement algorithm.

For periodic systems the sheet dimensions are snapped to exact multiples
of the orthogonal repeat units so the lattice tiles perfectly.

AB stacking for multi-layer (Bernal / graphite):
    - Layer A at z = 0
    - Layer B at z = d, shifted by (1.228, 0.709) so that the B-sublattice
      atoms sit above the centers of A-layer hexagons
    - Layer A' at z = 2d (same x,y as layer A), etc. (ABAB...)
"""

import numpy as np
from ase import Atoms

from .core import (
    CC_BOND,
    GRAPHITE_INTERLAYER,
    ARMCHAIR_REPEAT,
    ZIGZAG_REPEAT,
    init_metadata_arrays,
    ATOM_TYPE_KEY,
    RESIDUE_NAME_KEY,
    RESIDUE_NUM_KEY,
    LAYER_KEY,
)

# Derived constants
DX = 1.228   # = CC_BOND * cos(30deg) = 1.418 * sqrt(3)/2
DY_SHORT = 0.709   # = CC_BOND / 2 = 1.418 / 2
DY_LONG = CC_BOND  # = 1.418

# Full Y repeat = DY_SHORT + DY_LONG + DY_SHORT + DY_LONG = but actually
# the repeat in Y is 2 * ZIGZAG_REPEAT = 2 * 2.127 = 4.254 A
# where ZIGZAG_REPEAT = DY_SHORT + DY_LONG = 0.709 + 1.418 = 2.127
Y_REPEAT = 2 * ZIGZAG_REPEAT  # 4.254 A
X_REPEAT = ARMCHAIR_REPEAT     # 2.456 A

# AB stacking offset for layer B relative to layer A
AB_OFFSET_X = DX       # 1.228 A
AB_OFFSET_Y = DY_SHORT  # 0.709 A


def _generate_single_layer_coords(nx_cells: int, ny_cells: int) -> np.ndarray:
    """Generate coordinates for a single graphene layer.

    The orthogonal supercell contains 4 atoms with dimensions:
        a = X_REPEAT = 2.456 A  (along X / armchair)
        b = Y_REPEAT = 4.254 A  (along Y / zigzag)

    The 4 basis atoms within the cell (at z=0) are:
        atom 0: (0,     0    )
        atom 1: (0,     1.418)   = (0,  CC_BOND)
        atom 2: (1.228, 2.127)   = (DX, CC_BOND + DY_SHORT)
        atom 3: (1.228, 3.545)   = (DX, 2*CC_BOND + DY_SHORT)

    Cell vectors: a = (2*DX, 0) = (2.456, 0), b = (0, 2*(CC_BOND + DY_SHORT)) = (0, 4.254)

    Parameters
    ----------
    nx_cells : int
        Number of unit cell repeats in X direction.
    ny_cells : int
        Number of unit cell repeats in Y direction.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Coordinates of all atoms in the layer at z=0.
    """
    # 4 basis atoms per unit cell
    basis = np.array([
        [0.0,  0.0,  0.0],
        [0.0,  CC_BOND, 0.0],
        [DX,   CC_BOND + DY_SHORT, 0.0],
        [DX,   2 * CC_BOND + DY_SHORT, 0.0],
    ])

    coords = []
    for ix in range(nx_cells):
        for iy in range(ny_cells):
            offset = np.array([ix * X_REPEAT, iy * Y_REPEAT, 0.0])
            for b in basis:
                coords.append(b + offset)

    return np.array(coords)


def _generate_nonperiodic_coords(x_dim: float, y_dim: float) -> np.ndarray:
    """Generate graphene coordinates for a non-periodic sheet.

    Uses the same algorithm as GOPY: fill hexagons row by row,
    getting as close to the requested dimensions as possible.
    Atoms may slightly exceed the requested dimensions (same as GOPY behavior).

    Parameters
    ----------
    x_dim : float
        Desired X dimension in Angstrom.
    y_dim : float
        Desired Y dimension in Angstrom.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
    """
    # Use enough cells to cover the requested area, then trim
    nx = max(1, int(np.ceil(x_dim / X_REPEAT)) + 1)
    ny = max(1, int(np.ceil(y_dim / Y_REPEAT)) + 1)

    coords = _generate_single_layer_coords(nx, ny)

    # Trim to requested dimensions (with small tolerance)
    tol = 0.01
    mask = (coords[:, 0] <= x_dim + tol) & (coords[:, 1] <= y_dim + tol)
    coords = coords[mask]

    return coords


def generate_pristine_graphene(x_dim: float, y_dim: float,
                                periodic: bool = False,
                                nlayer: int = 1,
                                distance: float = GRAPHITE_INTERLAYER) -> Atoms:
    """Generate a pristine graphene structure.

    Parameters
    ----------
    x_dim : float
        Desired X dimension in Angstrom.
    y_dim : float
        Desired Y dimension in Angstrom.
    periodic : bool
        If True, create a periodic structure with exact lattice vectors.
        Dimensions are snapped to the nearest multiples of the repeat units.
    nlayer : int
        Number of graphene layers (default 1).
    distance : float
        Interlayer C-C distance in Angstrom (default 3.35, graphite value).
        Values below GRAPHITE_INTERLAYER are not allowed.

    Returns
    -------
    Atoms
        ASE Atoms object with gengo metadata.

    Raises
    ------
    ValueError
        If distance < GRAPHITE_INTERLAYER or dimensions are non-positive.
    """
    if x_dim <= 0 or y_dim <= 0:
        raise ValueError(f"Dimensions must be positive, got x={x_dim}, y={y_dim}")

    if nlayer < 1:
        raise ValueError(f"Number of layers must be >= 1, got {nlayer}")

    if distance < GRAPHITE_INTERLAYER:
        raise ValueError(
            f"Interlayer distance {distance:.3f} A is less than graphite "
            f"minimum {GRAPHITE_INTERLAYER:.3f} A"
        )

    if periodic:
        # Snap dimensions to exact multiples of repeat units
        nx_cells = max(1, round(x_dim / X_REPEAT))
        ny_cells = max(1, round(y_dim / Y_REPEAT))

        actual_x = nx_cells * X_REPEAT
        actual_y = ny_cells * Y_REPEAT

        coords = _generate_single_layer_coords(nx_cells, ny_cells)
        n_per_layer = len(coords)

        print(f"Periodic graphene: {nx_cells}x{ny_cells} unit cells")
        print(f"  Requested: {x_dim:.3f} x {y_dim:.3f} A")
        print(f"  Actual:    {actual_x:.3f} x {actual_y:.3f} A")
        print(f"  Atoms per layer: {n_per_layer}")
    else:
        coords = _generate_nonperiodic_coords(x_dim, y_dim)
        n_per_layer = len(coords)

        if n_per_layer == 0:
            raise ValueError(
                f"No atoms generated for dimensions {x_dim} x {y_dim} A. "
                f"Minimum dimensions: ~{X_REPEAT:.3f} x ~{Y_REPEAT:.3f} A"
            )

        actual_x = coords[:, 0].max() - coords[:, 0].min()
        actual_y = coords[:, 1].max() - coords[:, 1].min()

        print(f"Non-periodic graphene sheet")
        print(f"  Requested: {x_dim:.3f} x {y_dim:.3f} A")
        print(f"  Actual:    {actual_x:.3f} x {actual_y:.3f} A")
        print(f"  Atoms per layer: {n_per_layer}")

    # Build multi-layer structure
    all_coords = []
    all_layers = []

    for layer_idx in range(nlayer):
        z_offset = layer_idx * distance
        layer_coords = coords.copy()
        layer_coords[:, 2] = z_offset

        # AB stacking: odd layers get offset
        if layer_idx % 2 == 1:
            layer_coords[:, 0] += AB_OFFSET_X
            layer_coords[:, 1] += AB_OFFSET_Y

            if periodic:
                # Wrap shifted coordinates back into the cell
                layer_coords[:, 0] = layer_coords[:, 0] % (nx_cells * X_REPEAT)
                layer_coords[:, 1] = layer_coords[:, 1] % (ny_cells * Y_REPEAT)

        all_coords.append(layer_coords)
        all_layers.extend([layer_idx] * n_per_layer)

    all_coords = np.vstack(all_coords)
    n_total = len(all_coords)

    print(f"  Total layers: {nlayer}")
    if nlayer > 1:
        print(f"  Interlayer distance: {distance:.3f} A")
        print(f"  Stacking: ABAB (Bernal)")
    print(f"  Total atoms: {n_total}")

    # Create ASE Atoms object
    symbols = ["C"] * n_total
    atoms = Atoms(symbols=symbols, positions=all_coords)

    # Set cell and PBC
    if periodic:
        lx = nx_cells * X_REPEAT
        ly = ny_cells * Y_REPEAT
        if nlayer > 1:
            # Slab: z dimension = span of layers + vacuum
            lz = (nlayer - 1) * distance + 20.0  # 20 A vacuum
        else:
            lz = 20.0  # vacuum for single layer
        atoms.set_cell([lx, ly, lz])
        atoms.set_pbc([True, True, False])
    else:
        # Non-periodic: set a large box for visualization purposes
        margin = 5.0
        xmin, ymin = all_coords[:, 0].min(), all_coords[:, 1].min()
        xmax, ymax = all_coords[:, 0].max(), all_coords[:, 1].max()
        if nlayer > 1:
            zmax = (nlayer - 1) * distance
        else:
            zmax = 0.0
        atoms.set_cell([
            xmax - xmin + 2 * margin,
            ymax - ymin + 2 * margin,
            zmax + 2 * margin if zmax > 0 else 10.0,
        ])
        atoms.set_pbc(False)

    # Initialize metadata
    atom_types = np.array(["CX"] * n_total, dtype="U4")
    residue_names = np.array(["GGG"] * n_total, dtype="U3")
    residue_numbers = np.arange(1, n_total + 1, dtype=int)
    layers_arr = np.array(all_layers, dtype=int)

    init_metadata_arrays(atoms, atom_types, residue_names, residue_numbers, layers_arr)

    return atoms
