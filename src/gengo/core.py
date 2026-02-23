"""
Core bond definitions and periodic-aware bond checking.

This module preserves the exact scientific bond definitions and validation
logic from GOPY (Muraru et al., SoftwareX 2020), refactored to work with
ASE Atoms objects and support periodic boundary conditions.

Bond checking uses the same distance-based approach: a bond between two atoms
is valid if their distance is within 0.975-1.025 of the typical bond length
for that atom-type/residue-name pair.

Performance:
    Neighbor searches use scipy.spatial.cKDTree for O(log n) lookups instead
    of the O(n) brute-force distance scan in the original implementation.
    For periodic systems, minimum image convention is handled by wrapping
    coordinates into the cell before building the tree, then querying with
    periodic box dimensions.
"""

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# C-C bond length in graphene (Angstrom).
# GOPY uses 1.418 A; the experimental crystallographic value is 1.421 A
# (graphite, a = 2.461 A, space group P6_3/mmc). Difference is ~0.2%.
CC_BOND = 1.418
GRAPHITE_INTERLAYER = 3.35  # Graphite interlayer C-C distance (Angstrom)

# Repeat units of the orthogonal supercell used for periodic graphene.
# These derive from the C-C bond length, not independent parameters.
# Along X (armchair direction): a_orth = CC_BOND * sqrt(3) = 2 * 1.228
ARMCHAIR_REPEAT = 2.456  # Angstrom
# Along Y (zigzag direction): half of b_orth = CC_BOND + CC_BOND/2 = 2.127
ZIGZAG_REPEAT = 2.127  # Angstrom

# ---------------------------------------------------------------------------
# Bond definitions
# ---------------------------------------------------------------------------
# Each entry: (atom_name_1, residue_name_1, atom_name_2, residue_name_2) -> length
# These are the exact values from GOPY's Typical_Bond definitions.
# The identity string in GOPY was constructed as atom_name1+atom_name2+res1+res2.
# We use tuples for clarity but preserve the same bond lengths.

# Bond definitions for GO generation (bond_list_1 in GOPY)
GO_BONDS = {
    # Graphene C-C bonds
    ("CX", "GGG", "CX", "GGG"): 1.418,
    ("CX", "GGG", "CY", "C1A"): 1.418,
    ("CX", "GGG", "CY", "H1A"): 1.418,
    ("CX", "GGG", "CY", "E1A"): 1.418,
    # Carboxyl group bonds
    ("CX", "GGG", "C4", "C1A"): 1.520,
    ("CY", "C1A", "C4", "C1A"): 1.520,
    ("C4", "C1A", "OJ", "C1A"): 1.210,
    ("C4", "C1A", "OK", "C1A"): 1.320,
    ("OK", "C1A", "HK", "C1A"): 0.967,
    # Hydroxyl group bonds
    ("CX", "GGG", "OL", "H1A"): 1.490,
    ("CY", "H1A", "OL", "H1A"): 1.490,
    ("OL", "H1A", "HK", "H1A"): 0.967,
    # Epoxy group bonds
    ("CX", "GGG", "OE", "E1A"): 1.460,
    ("CY", "E1A", "OE", "E1A"): 1.460,
}

# Generic bond definitions (bond_list_3 in GOPY) - used for general checking
GENERIC_BONDS = {
    ("N", "H"): 1.010,
    ("N", "O"): 1.060,
    ("N", "C"): 1.475,
    ("N", "N"): 1.450,
    ("O", "H"): 0.970,
    ("O", "C"): 1.160,
    ("O", "O"): 1.490,
    ("C", "H"): 1.090,
    ("C", "C"): 1.540,
    ("H", "H"): 0.740,
}

# Tolerance for bond checking (same as GOPY: 0.975 to 1.025 of typical length)
BOND_TOL_LOW = 0.975
BOND_TOL_HIGH = 1.025

# Maximum bond length in GO_BONDS (used as default neighbor search cutoff)
_MAX_BOND_LEN = max(GO_BONDS.values())  # 1.520
NEIGHBOR_CUTOFF = _MAX_BOND_LEN * BOND_TOL_HIGH + 0.1  # ~1.66, generous

# ---------------------------------------------------------------------------
# Per-atom metadata keys stored in atoms.arrays
# ---------------------------------------------------------------------------
ATOM_TYPE_KEY = "atom_types"       # e.g., "CX", "CY", "C4", "OJ", "OK", "OE", "OL", "HK"
RESIDUE_NAME_KEY = "residue_names"  # e.g., "GGG", "C1A", "E1A", "H1A"
RESIDUE_NUM_KEY = "residue_numbers"  # integer residue number
LAYER_KEY = "layers"                # integer layer index (0-based)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_atom_type(atoms: Atoms, idx: int) -> str:
    """Get the atom type label (CX, CY, C4, etc.) for atom at index idx."""
    return str(atoms.arrays[ATOM_TYPE_KEY][idx])


def get_residue_name(atoms: Atoms, idx: int) -> str:
    """Get the residue name (GGG, C1A, E1A, H1A) for atom at index idx."""
    return str(atoms.arrays[RESIDUE_NAME_KEY][idx])


def get_residue_number(atoms: Atoms, idx: int) -> int:
    """Get the residue number for atom at index idx."""
    return int(atoms.arrays[RESIDUE_NUM_KEY][idx])


def set_metadata(atoms: Atoms, idx: int, atom_type: str, residue_name: str,
                 residue_number: int, layer: int = 0):
    """Set all metadata fields for a single atom."""
    atoms.arrays[ATOM_TYPE_KEY][idx] = atom_type
    atoms.arrays[RESIDUE_NAME_KEY][idx] = residue_name
    atoms.arrays[RESIDUE_NUM_KEY][idx] = residue_number
    atoms.arrays[LAYER_KEY][idx] = layer


def init_metadata_arrays(atoms: Atoms, atom_types=None, residue_names=None,
                         residue_numbers=None, layers=None):
    """Initialize metadata arrays on an Atoms object.

    If arrays are not provided, defaults are used:
    - atom_types: "CX" for all C atoms
    - residue_names: "GGG" for all atoms
    - residue_numbers: sequential from 1
    - layers: 0 for all atoms
    """
    n = len(atoms)
    if atom_types is None:
        atom_types = np.array(["CX"] * n, dtype="U4")
    atoms.arrays[ATOM_TYPE_KEY] = np.array(atom_types, dtype="U4")

    if residue_names is None:
        residue_names = np.array(["GGG"] * n, dtype="U3")
    atoms.arrays[RESIDUE_NAME_KEY] = np.array(residue_names, dtype="U3")

    if residue_numbers is None:
        residue_numbers = np.arange(1, n + 1, dtype=int)
    atoms.arrays[RESIDUE_NUM_KEY] = np.array(residue_numbers, dtype=int)

    if layers is None:
        layers = np.zeros(n, dtype=int)
    atoms.arrays[LAYER_KEY] = np.array(layers, dtype=int)


def append_atom(atoms: Atoms, symbol: str, position, atom_type: str,
                residue_name: str, residue_number: int, layer: int = 0) -> Atoms:
    """Append a single atom to an Atoms object, preserving metadata arrays.

    Returns the modified Atoms object (atoms is modified in-place via
    concatenation and reassignment of arrays).
    """
    from ase import Atom as ASEAtom

    # Save existing metadata
    old_at = atoms.arrays.get(ATOM_TYPE_KEY, np.array([], dtype="U4"))
    old_rn = atoms.arrays.get(RESIDUE_NAME_KEY, np.array([], dtype="U3"))
    old_rnum = atoms.arrays.get(RESIDUE_NUM_KEY, np.array([], dtype=int))
    old_lay = atoms.arrays.get(LAYER_KEY, np.array([], dtype=int))

    # Append atom
    atoms.append(ASEAtom(symbol, position))

    # Restore and extend metadata
    atoms.arrays[ATOM_TYPE_KEY] = np.append(old_at, atom_type)
    atoms.arrays[RESIDUE_NAME_KEY] = np.append(old_rn, residue_name)
    atoms.arrays[RESIDUE_NUM_KEY] = np.append(old_rnum, residue_number)
    atoms.arrays[LAYER_KEY] = np.append(old_lay, layer)

    return atoms


def remove_atom(atoms: Atoms, idx: int) -> Atoms:
    """Remove atom at index idx, preserving metadata arrays.

    Returns a new Atoms object with the atom removed.
    """
    mask = np.ones(len(atoms), dtype=bool)
    mask[idx] = False

    new_atoms = atoms[mask]
    # atoms[mask] copies arrays automatically for standard arrays,
    # but custom arrays in atoms.arrays need explicit handling
    for key in [ATOM_TYPE_KEY, RESIDUE_NAME_KEY, RESIDUE_NUM_KEY, LAYER_KEY]:
        if key in atoms.arrays:
            new_atoms.arrays[key] = atoms.arrays[key][mask]

    return new_atoms


# ---------------------------------------------------------------------------
# Fast neighbor search
# ---------------------------------------------------------------------------

class PeriodicKDTree:
    """Wrapper around cKDTree that handles periodic boundary conditions.

    For periodic dimensions, positions are wrapped into [0, cell_length).
    For non-periodic dimensions, positions are shifted so min=0.
    The shift offsets are stored so new query points can be transformed
    consistently.
    """
    __slots__ = ('tree', 'pbc', 'cell_lengths', 'shifts', 'n')

    def __init__(self, positions, cell_lengths, pbc):
        self.pbc = np.asarray(pbc)
        self.cell_lengths = np.asarray(cell_lengths, dtype=float)
        self.n = len(positions)

        wrapped = positions.copy()
        self.shifts = np.zeros(3)

        for dim in range(3):
            if self.pbc[dim] and self.cell_lengths[dim] > 1e-6:
                wrapped[:, dim] = wrapped[:, dim] % self.cell_lengths[dim]
            else:
                lo = wrapped[:, dim].min() if self.n > 0 else 0.0
                if lo < 0:
                    self.shifts[dim] = -lo
                    wrapped[:, dim] += self.shifts[dim]

        # Build boxsize: periodic dims use cell length, non-periodic use huge value
        boxsize = np.zeros(3)
        for dim in range(3):
            if self.pbc[dim] and self.cell_lengths[dim] > 1e-6:
                boxsize[dim] = self.cell_lengths[dim]
            else:
                span = (wrapped[:, dim].max() - wrapped[:, dim].min()) if self.n > 0 else 0.0
                boxsize[dim] = max(span * 100 + 1000, 1000.0)

        self.tree = cKDTree(wrapped, boxsize=boxsize)

    @property
    def data(self):
        return self.tree.data

    def wrap_point(self, pos):
        """Transform a position into the tree's coordinate system."""
        p = np.array(pos, dtype=float)
        for dim in range(3):
            if self.pbc[dim] and self.cell_lengths[dim] > 1e-6:
                p[dim] = p[dim] % self.cell_lengths[dim]
            else:
                p[dim] += self.shifts[dim]
        return p

    def query_ball_point(self, point, r):
        return self.tree.query_ball_point(point, r)

    def mic_dist(self, pos_a, pos_b):
        """Minimum-image distance (periodic in flagged dimensions only)."""
        delta = pos_b - pos_a
        for dim in range(3):
            if self.pbc[dim] and self.cell_lengths[dim] > 1e-6:
                delta[dim] -= self.cell_lengths[dim] * round(delta[dim] / self.cell_lengths[dim])
        return np.linalg.norm(delta)


def _build_kdtree(positions: np.ndarray, cell=None, pbc=None):
    """Build a PeriodicKDTree for fast neighbor lookup.

    Parameters
    ----------
    positions : np.ndarray, shape (n, 3)
    cell : array-like or None
        Cell lengths [lx, ly, lz] or 3x3 cell matrix.
    pbc : array-like or None
        Periodic boundary flags [px, py, pz].

    Returns
    -------
    PeriodicKDTree or cKDTree
    """
    if pbc is not None and np.any(pbc) and cell is not None:
        cell_lengths = np.asarray(cell)
        if len(cell_lengths.shape) > 1:
            cell_lengths = np.array([cell_lengths[0, 0], cell_lengths[1, 1], cell_lengths[2, 2]])
        return PeriodicKDTree(positions, cell_lengths, pbc)
    else:
        return cKDTree(positions)



# ---------------------------------------------------------------------------
# Distance calculation (periodic-aware)
# ---------------------------------------------------------------------------

def get_distance(atoms: Atoms, idx1: int, idx2: int, periodic: bool = False) -> float:
    """Calculate distance between two atoms, optionally using minimum image convention."""
    return atoms.get_distance(idx1, idx2, mic=periodic)


def get_distances_from(atoms: Atoms, idx: int, periodic: bool = False) -> np.ndarray:
    """Get distances from atom idx to all other atoms.

    Returns array of shape (n_atoms,) with distance to self = 0.
    """
    n = len(atoms)
    if n == 0:
        return np.array([])

    indices = list(range(n))
    dists = atoms.get_distances(idx, indices, mic=periodic)
    return dists


# ---------------------------------------------------------------------------
# Bond checking
# ---------------------------------------------------------------------------

def _make_bond_id(atype1: str, rname1: str, atype2: str, rname2: str):
    """Create both possible bond identity tuples for a pair of atoms."""
    return (
        (atype1, rname1, atype2, rname2),
        (atype2, rname2, atype1, rname1),
    )


def _check_bond_by_types(atype1, rname1, atype2, rname2, dist, bond_defs):
    """Check if a bond is valid given atom types, residue names, and distance.

    Pure function — no Atoms object access needed.
    """
    id_fwd, id_rev = _make_bond_id(atype1, rname1, atype2, rname2)
    for bond_id in (id_fwd, id_rev):
        if bond_id in bond_defs:
            typical_len = bond_defs[bond_id]
            if BOND_TOL_LOW * typical_len <= dist <= BOND_TOL_HIGH * typical_len:
                return True
    return False


def check_bond(atoms: Atoms, idx1: int, idx2: int, bond_defs: dict,
               periodic: bool = False) -> bool:
    """Check if a valid bond exists between atoms idx1 and idx2.

    Uses the same logic as GOPY: the bond is valid if the atom-type/residue-name
    identity matches an entry in bond_defs AND the distance is within
    [0.975, 1.025] of the typical bond length.
    """
    atype1 = get_atom_type(atoms, idx1)
    rname1 = get_residue_name(atoms, idx1)
    atype2 = get_atom_type(atoms, idx2)
    rname2 = get_residue_name(atoms, idx2)

    dist = get_distance(atoms, idx1, idx2, periodic)
    return _check_bond_by_types(atype1, rname1, atype2, rname2, dist, bond_defs)


def identify_bonds(atoms: Atoms, idx: int, bond_defs: dict = None,
                   periodic: bool = False, cutoff: float = 2.0,
                   tree: cKDTree = None) -> list:
    """Find all valid bonds for atom at index idx.

    This is the core function from GOPY's identify_bonds, refactored for ASE.
    Uses cKDTree for O(log n) neighbor search instead of O(n) brute force.

    Parameters
    ----------
    atoms : Atoms
        The full atomic structure.
    idx : int
        Index of the atom to check.
    bond_defs : dict
        Bond definitions to use. Defaults to GO_BONDS.
    periodic : bool
        Whether to use minimum image convention for distances.
    cutoff : float
        Maximum distance to search for neighbors (Angstrom).
        Default 2.0 matches GOPY's search radius for non-hydrogen atoms.
    tree : cKDTree, optional
        Pre-built tree for neighbor search. If None, one is built on the fly.
        For atoms appended after the tree was built (idx >= tree.n), the tree
        is queried with the new atom's position to find neighbors among the
        existing atoms.

    Returns
    -------
    list of (int, float)
        List of (neighbor_index, distance) for each valid bond found.
        Returns empty list if number of nearby atoms != number of identified
        bonds (same logic as GOPY to detect invalid placements).
    """
    if bond_defs is None:
        bond_defs = GO_BONDS

    n = len(atoms)
    if n <= 1:
        return []

    atype = get_atom_type(atoms, idx)

    # Determine cutoff based on atom type (same logic as GOPY)
    hydrogens = {'H15', 'H14', 'H13', 'H12', 'H11', 'H10', 'H9', 'H8',
                 'H7', 'H6', 'H5', 'H4', 'H3', 'H2', 'H1'}
    rname = get_residue_name(atoms, idx)

    if atype in hydrogens:
        search_cutoff = 1.6
    elif rname == "P1A" and atype not in hydrogens:
        search_cutoff = 1.8
    else:
        search_cutoff = cutoff

    # Build tree on the fly if not provided
    if tree is None:
        cell = atoms.get_cell() if periodic else None
        pbc = np.array([True, True, False]) if periodic else None
        tree = _build_kdtree(atoms.get_positions(), cell, pbc)

    is_periodic_tree = hasattr(tree, 'wrap_point')  # PeriodicKDTree vs plain cKDTree

    if idx < tree.n:
        # Atom is in the tree — direct lookup
        query_point = tree.data[idx]
    else:
        # Atom was appended after tree was built — wrap its position
        pos_idx = atoms.positions[idx]
        if is_periodic_tree:
            query_point = tree.wrap_point(pos_idx)
        else:
            query_point = pos_idx

    # Query the tree for neighbors. This finds neighbors among atoms that
    # were in the tree when it was built. For newly appended atoms (idx >= tree.n),
    # this correctly finds existing atoms near the new position.
    nearby_indices = tree.query_ball_point(query_point, search_cutoff)
    # Remove self if present
    nearby_indices = [j for j in nearby_indices if j != idx]

    def _dist(pos_a, pos_b):
        """Distance, using MIC if periodic tree."""
        if is_periodic_tree:
            return tree.mic_dist(pos_a, pos_b)
        return np.linalg.norm(pos_b - pos_a)

    # Also check other recently appended atoms not in the tree
    # (e.g., C4 checking against OJ that was just appended)
    if tree.n < n:
        for j in range(tree.n, n):
            if j == idx:
                continue
            pos_j = atoms.positions[j]
            if is_periodic_tree:
                pos_j_wrapped = tree.wrap_point(pos_j)
            else:
                pos_j_wrapped = pos_j
            d = _dist(query_point, pos_j_wrapped)
            if d <= search_cutoff:
                nearby_indices.append(j)

    # Compute minimum-image distances for all nearby atoms
    nearby_with_dist = []
    for j in nearby_indices:
        if j < tree.n:
            # j is in the tree — use tree coordinates
            d = _dist(query_point, tree.data[j])
        else:
            # j is appended — wrap its position
            pos_j = atoms.positions[j]
            if is_periodic_tree:
                pos_j_wrapped = tree.wrap_point(pos_j)
            else:
                pos_j_wrapped = pos_j
            d = _dist(query_point, pos_j_wrapped)
        if 0 < d <= search_cutoff:
            nearby_with_dist.append((j, d))

    # Check which nearby atoms form valid bonds
    atypes = atoms.arrays[ATOM_TYPE_KEY]
    rnames = atoms.arrays[RESIDUE_NAME_KEY]

    identified = []
    for j, dist in nearby_with_dist:
        if _check_bond_by_types(atype, rname, str(atypes[j]), str(rnames[j]),
                                dist, bond_defs):
            identified.append((int(j), float(dist)))

    # GOPY logic: if number of nearby atoms != number of identified bonds,
    # return empty (indicates invalid placement)
    if len(nearby_with_dist) != len(identified):
        return []

    return identified


def is_connected_to_functional_group(atoms: Atoms, idx: int, bond_defs: dict = None,
                                     periodic: bool = False, tree: cKDTree = None) -> bool:
    """Check if atom at idx is connected to a functional group (C1A, E1A, H1A, P1A).

    Same logic as GOPY's check_connected.
    """
    if bond_defs is None:
        bond_defs = GO_BONDS

    bonds = identify_bonds(atoms, idx, bond_defs, periodic, tree=tree)
    rnames = atoms.arrays[RESIDUE_NAME_KEY]
    for neighbor_idx, _ in bonds:
        if str(rnames[neighbor_idx]) in ("C1A", "E1A", "H1A", "P1A"):
            return True
    return False


def get_edge_atoms(atoms: Atoms, bond_defs: dict = None,
                   periodic: bool = False) -> list:
    """Get indices of edge atoms (1 or 2 bonds, not connected to functional groups).

    Uses cKDTree for fast neighbor lookup. In periodic mode, boundary atoms
    will typically have 3 bonds through periodic images and won't appear as edges.
    """
    if bond_defs is None:
        bond_defs = GO_BONDS

    # Build tree once for all queries
    # Use xy-periodic PBC for graphene (z is always non-periodic)
    cell = atoms.get_cell() if periodic else None
    pbc = np.array([True, True, False]) if periodic else None
    tree = _build_kdtree(atoms.get_positions(), cell, pbc)

    atypes = atoms.arrays[ATOM_TYPE_KEY]
    rnames = atoms.arrays[RESIDUE_NAME_KEY]

    edge_indices = []
    for i in range(len(atoms)):
        if str(atypes[i]) != "CX":
            continue
        bonds = identify_bonds(atoms, i, bond_defs, periodic, tree=tree)
        if 0 < len(bonds) < 3:
            # Check not connected to functional group
            has_fg = False
            for neighbor_idx, _ in bonds:
                if str(rnames[neighbor_idx]) in ("C1A", "E1A", "H1A", "P1A"):
                    has_fg = True
                    break
            if not has_fg:
                edge_indices.append(i)
    return edge_indices


def get_available_atoms(atoms: Atoms, bond_defs: dict = None,
                        periodic: bool = False) -> list:
    """Get indices of all CX atoms not connected to functional groups.

    Uses cKDTree for fast neighbor lookup.
    """
    if bond_defs is None:
        bond_defs = GO_BONDS

    # Build tree once for all queries
    cell = atoms.get_cell() if periodic else None
    pbc = np.array([True, True, False]) if periodic else None
    tree = _build_kdtree(atoms.get_positions(), cell, pbc)

    atypes = atoms.arrays[ATOM_TYPE_KEY]
    rnames = atoms.arrays[RESIDUE_NAME_KEY]

    available = []
    for i in range(len(atoms)):
        if str(atypes[i]) != "CX":
            continue
        if not is_connected_to_functional_group(atoms, i, bond_defs, periodic, tree=tree):
            available.append(i)
    return available


def get_central_atoms(atoms: Atoms, bond_defs: dict = None,
                      periodic: bool = False) -> list:
    """Get indices of atoms with exactly 3 bonds, not connected to functional groups.

    Uses cKDTree for fast neighbor lookup.
    """
    if bond_defs is None:
        bond_defs = GO_BONDS

    cell = atoms.get_cell() if periodic else None
    pbc = np.array([True, True, False]) if periodic else None
    tree = _build_kdtree(atoms.get_positions(), cell, pbc)

    atypes = atoms.arrays[ATOM_TYPE_KEY]
    rnames = atoms.arrays[RESIDUE_NAME_KEY]

    central = []
    for i in range(len(atoms)):
        if str(atypes[i]) != "CX":
            continue
        bonds = identify_bonds(atoms, i, bond_defs, periodic, tree=tree)
        if len(bonds) == 3:
            has_fg = False
            for neighbor_idx, _ in bonds:
                if str(rnames[neighbor_idx]) in ("C1A", "E1A", "H1A", "P1A"):
                    has_fg = True
                    break
            if not has_fg:
                central.append(i)
    return central
