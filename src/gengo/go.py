"""
Graphene oxide (GO) generation.

Adds three types of functional groups to a pristine graphene sheet:
    - Carboxyl (-COOH): only on edge carbons (atoms with < 3 bonds)
    - Epoxy (-O-): bridging oxygen on the basal plane between two bonded C atoms
    - Hydroxyl (-OH): on the basal plane, above or below the sheet

The placement algorithm is the same as GOPY (Muraru et al., SoftwareX 2020):
    1. Randomly pick a functional group type that still needs to be placed.
    2. Randomly pick an eligible carbon atom.
    3. Place atoms one at a time using known bond lengths and angles.
    4. Validate each placement with identify_bonds (must form exactly the
       expected number of bonds).
    5. Retry up to 50 times per group; give up if placement fails.

For periodic systems, distance checks use minimum image convention so that
atoms near cell boundaries are correctly handled. Boundary atoms with 3 bonds
(through periodic images) are NOT classified as edge atoms, so COOH groups
cannot be placed there (physically correct: no edges in a periodic sheet).

Performance:
    Neighbor lookups use scipy.spatial.cKDTree. The tree is rebuilt after each
    successful group placement (when atoms are added). Edge/available atom
    lists are cached and updated incrementally.
"""

import math
import random

import numpy as np
from ase import Atoms

from .core import (
    GO_BONDS,
    CC_BOND,
    ATOM_TYPE_KEY,
    RESIDUE_NAME_KEY,
    RESIDUE_NUM_KEY,
    LAYER_KEY,
    get_atom_type,
    get_residue_name,
    get_residue_number,
    identify_bonds,
    is_connected_to_functional_group,
    get_edge_atoms,
    get_available_atoms,
    append_atom,
    set_metadata,
    init_metadata_arrays,
    _build_kdtree,
)

# Maximum placement attempts per functional group
MAX_ATTEMPTS = 50


def _top_or_down() -> int:
    """Random above (+1) or below (-1) the graphene plane."""
    return random.choice([1, -1])


def _next_residue_number(atoms: Atoms) -> int:
    """Get the next available residue number."""
    if RESIDUE_NUM_KEY in atoms.arrays and len(atoms) > 0:
        return int(atoms.arrays[RESIDUE_NUM_KEY].max()) + 1
    return 1


def _save_atoms_state(atoms: Atoms):
    """Save a snapshot of the atoms object for rollback on failure.

    Returns a tuple of (n_atoms, metadata_copies) that can be passed to
    _restore_atoms_state to undo any appended atoms and metadata changes.
    """
    n = len(atoms)
    saved_meta = {}
    for key in [ATOM_TYPE_KEY, RESIDUE_NAME_KEY, RESIDUE_NUM_KEY, LAYER_KEY]:
        if key in atoms.arrays:
            saved_meta[key] = atoms.arrays[key][:n].copy()
    return (n, saved_meta)


def _restore_atoms_state(atoms: Atoms, state) -> Atoms:
    """Restore atoms object to a previously saved state.

    Removes any atoms appended after the snapshot and restores metadata
    arrays. This guarantees no partial functional groups are left behind.
    """
    n, saved_meta = state
    if len(atoms) > n:
        atoms = _remove_last_n(atoms, len(atoms) - n)
    # Restore metadata in case any existing atom's metadata was modified
    for key, arr in saved_meta.items():
        if key in atoms.arrays:
            atoms.arrays[key][:n] = arr
    return atoms


def _add_carboxyl(atoms: Atoms, atom_idx: int, periodic: bool = False,
                  tree=None) -> tuple[Atoms, bool]:
    """Try to add a carboxyl group (-COOH) to the carbon at atom_idx.

    Adds 4 new atoms (C4, OJ, OK, HK) and converts the base CX to CY.
    The C4 atom is placed directly above/below the carbon, then OJ and OK
    are placed at angles, and HK is placed on OK.

    IMPORTANT: Always returns the atoms object (restored on failure, modified
    on success). The caller MUST use the returned atoms reference because
    append_atom mutates in-place but _restore creates a new object.

    Returns
    -------
    tuple of (Atoms, bool)
        (atoms_object, success). The caller must ALWAYS use the returned
        atoms object regardless of success/failure.
    """
    # Save state for rollback on failure
    state = _save_atoms_state(atoms)
    base_n = len(atoms)

    pos = atoms.positions[atom_idx]
    x, y, z = pos
    ct = _top_or_down()
    resnum = _next_residue_number(atoms)

    # Try 360 angle orientations (same as GOPY)
    alpha = random.randint(0, 359)
    for _ in range(360):
        alpha = (alpha + 1) % 360

        # Rollback any partial atoms from previous iteration
        if len(atoms) > base_n:
            atoms = _restore_atoms_state(atoms, state)

        # Place C4 directly above/below
        c4_pos = [x, y, z + ct * 1.52]
        atoms = append_atom(atoms, "C", c4_pos, "C4", "C1A", resnum)
        c4_idx = len(atoms) - 1

        bonds_c4 = identify_bonds(atoms, c4_idx, GO_BONDS, periodic, tree=tree)
        if len(bonds_c4) != 1 or bonds_c4[0][0] != atom_idx:
            continue

        # C4 placed successfully, now place OJ
        h = math.sin(math.radians(60)) * 1.20
        oj_pos = [
            c4_pos[0] - math.cos(math.radians(alpha)) * h,
            c4_pos[1] - math.sin(math.radians(alpha)) * h,
            c4_pos[2] + ct * math.cos(math.radians(60)) * 1.20,
        ]
        atoms = append_atom(atoms, "O", oj_pos, "OJ", "C1A", resnum)
        oj_idx = len(atoms) - 1

        bonds_oj = identify_bonds(atoms, oj_idx, GO_BONDS, periodic, tree=tree)
        if len(bonds_oj) != 1 or bonds_oj[0][0] != c4_idx:
            continue

        # OJ placed, now place OK
        h2 = math.sin(math.radians(60)) * 1.34
        ok_pos = [
            c4_pos[0] - math.cos(math.radians(alpha + 180)) * h2,
            c4_pos[1] - math.sin(math.radians(alpha + 180)) * h2,
            c4_pos[2] + ct * math.cos(math.radians(60)) * 1.34,
        ]
        atoms = append_atom(atoms, "O", ok_pos, "OK", "C1A", resnum)
        ok_idx = len(atoms) - 1

        bonds_ok = identify_bonds(atoms, ok_idx, GO_BONDS, periodic, tree=tree)
        if len(bonds_ok) != 1 or bonds_ok[0][0] != c4_idx:
            continue

        # OK placed, now place HK on OK
        hk_pos = [ok_pos[0], ok_pos[1], ok_pos[2] + ct * 0.98]
        atoms = append_atom(atoms, "H", hk_pos, "HK", "C1A", resnum)
        hk_idx = len(atoms) - 1

        bonds_hk = identify_bonds(atoms, hk_idx, GO_BONDS, periodic, tree=tree)
        if len(bonds_hk) != 1 or bonds_hk[0][0] != ok_idx:
            continue

        # All placed! Convert base CX to CY
        atoms.arrays[ATOM_TYPE_KEY][atom_idx] = "CY"
        atoms.arrays[RESIDUE_NAME_KEY][atom_idx] = "C1A"
        atoms.arrays[RESIDUE_NUM_KEY][atom_idx] = resnum
        return atoms, True

    # All 360 angles failed -- restore to original state
    atoms = _restore_atoms_state(atoms, state)
    return atoms, False


def _add_epoxy(atoms: Atoms, atom_idx: int, periodic: bool = False,
               tree=None) -> tuple[Atoms, bool]:
    """Try to add an epoxy group (-O-) bridging atom_idx and one of its neighbors.

    The oxygen is placed between two bonded C atoms at a height of
    1.46 * sin(60) above or below the plane, relative to the carbon z-position.

    IMPORTANT: Always returns the atoms object. The caller MUST use the
    returned atoms reference.

    Returns
    -------
    tuple of (Atoms, bool)
        (atoms_object, success).
    """
    state = _save_atoms_state(atoms)

    # Find neighbors of the chosen atom that are available (CX, not CY)
    bonds = identify_bonds(atoms, atom_idx, GO_BONDS, periodic, tree=tree)
    neighbors = [b[0] for b in bonds if get_atom_type(atoms, b[0]) == "CX"]

    if not neighbors:
        return atoms, False

    atom2_idx = random.choice(neighbors)
    ct = _top_or_down()
    resnum = _next_residue_number(atoms)

    pos1 = atoms.positions[atom_idx]
    pos2 = atoms.positions[atom2_idx]

    # For periodic systems, get the minimum-image position of atom2 relative to atom1
    if periodic and atoms.pbc.any():
        # Use the displacement vector with mic
        vec = atoms.get_distance(atom_idx, atom2_idx, mic=True, vector=True)
        midpoint = pos1 + vec / 2.0
    else:
        midpoint = (pos1 + pos2) / 2.0

    # IMPORTANT: z-offset is RELATIVE to the midpoint z (carbon sheet z),
    # not absolute. For multi-layer graphene the sheet can be at z=0, z=10, etc.
    oe_z = midpoint[2] + ct * 1.46 * math.sin(math.radians(60))
    oe_pos = [midpoint[0], midpoint[1], oe_z]

    atoms = append_atom(atoms, "O", oe_pos, "OE", "E1A", resnum)
    oe_idx = len(atoms) - 1

    bonds_oe = identify_bonds(atoms, oe_idx, GO_BONDS, periodic, tree=tree)
    if len(bonds_oe) == 2:
        bond_indices = {bonds_oe[0][0], bonds_oe[1][0]}
        if bond_indices == {atom_idx, atom2_idx}:
            # Convert both carbons to CY/CZ
            atoms.arrays[ATOM_TYPE_KEY][atom_idx] = "CY"
            atoms.arrays[RESIDUE_NAME_KEY][atom_idx] = "E1A"
            atoms.arrays[RESIDUE_NUM_KEY][atom_idx] = resnum

            atoms.arrays[ATOM_TYPE_KEY][atom2_idx] = "CZ"
            atoms.arrays[RESIDUE_NAME_KEY][atom2_idx] = "E1A"
            atoms.arrays[RESIDUE_NUM_KEY][atom2_idx] = resnum
            return atoms, True

    # Failed - restore to original state
    atoms = _restore_atoms_state(atoms, state)
    return atoms, False


def _add_hydroxyl(atoms: Atoms, atom_idx: int, periodic: bool = False,
                  tree=None) -> tuple[Atoms, bool]:
    """Try to add a hydroxyl group (-OH) to the carbon at atom_idx.

    Adds 2 new atoms (OL, HK) and converts the base CX to CY.

    IMPORTANT: Always returns the atoms object. The caller MUST use the
    returned atoms reference.

    Returns
    -------
    tuple of (Atoms, bool)
        (atoms_object, success).
    """
    state = _save_atoms_state(atoms)
    base_n = len(atoms)

    pos = atoms.positions[atom_idx]
    x, y, z = pos
    ct = _top_or_down()
    resnum = _next_residue_number(atoms)

    alpha = random.randint(0, 359)
    for _ in range(360):
        alpha = (alpha + 1) % 360

        # Rollback any partial atoms from previous iteration
        if len(atoms) > base_n:
            atoms = _restore_atoms_state(atoms, state)

        # Place OL directly above/below
        ol_pos = [x, y, z + ct * 1.49]
        atoms = append_atom(atoms, "O", ol_pos, "OL", "H1A", resnum)
        ol_idx = len(atoms) - 1

        bonds_ol = identify_bonds(atoms, ol_idx, GO_BONDS, periodic, tree=tree)
        if len(bonds_ol) != 1 or bonds_ol[0][0] != atom_idx:
            continue

        # OL placed, now place HK
        h_vert = math.sin(math.radians(19)) * 0.98
        h_horiz = math.cos(math.radians(19)) * 0.98
        hk_pos = [
            ol_pos[0] - math.cos(math.radians(alpha)) * h_horiz,
            ol_pos[1] - math.sin(math.radians(alpha)) * h_horiz,
            ol_pos[2] + ct * h_vert,
        ]
        atoms = append_atom(atoms, "H", hk_pos, "HK", "H1A", resnum)
        hk_idx = len(atoms) - 1

        bonds_hk = identify_bonds(atoms, hk_idx, GO_BONDS, periodic, tree=tree)
        if len(bonds_hk) != 1 or bonds_hk[0][0] != ol_idx:
            continue

        # Convert base CX to CY
        atoms.arrays[ATOM_TYPE_KEY][atom_idx] = "CY"
        atoms.arrays[RESIDUE_NAME_KEY][atom_idx] = "H1A"
        atoms.arrays[RESIDUE_NUM_KEY][atom_idx] = resnum
        return atoms, True

    # All 360 angles failed
    atoms = _restore_atoms_state(atoms, state)
    return atoms, False


def _remove_last_n(atoms: Atoms, n: int) -> Atoms:
    """Remove the last n atoms from an Atoms object, preserving metadata."""
    if n <= 0 or n > len(atoms):
        return atoms

    keep = len(atoms) - n
    new_atoms = atoms[:keep]

    # Preserve metadata arrays
    for key in [ATOM_TYPE_KEY, RESIDUE_NAME_KEY, RESIDUE_NUM_KEY, LAYER_KEY]:
        if key in atoms.arrays:
            new_atoms.arrays[key] = atoms.arrays[key][:keep].copy()

    # Preserve cell and PBC
    new_atoms.set_cell(atoms.get_cell())
    new_atoms.set_pbc(atoms.get_pbc())

    return new_atoms


def _report_composition(atoms: Atoms, placed_cooh: int = 0,
                        placed_epoxy: int = 0, placed_oh: int = 0):
    """Print composition report after GO generation."""
    from .calc import calculate_composition, print_composition

    comp = calculate_composition(atoms)
    print_composition(comp, header="Composition Report")

    # Additional GO-specific info
    if placed_cooh or placed_epoxy or placed_oh:
        print(f"  Placed groups: COOH={placed_cooh}, Epoxy={placed_epoxy}, OH={placed_oh}")

    if ATOM_TYPE_KEY in atoms.arrays:
        atypes = atoms.arrays[ATOM_TYPE_KEY]
        n_cx = np.sum(atypes == "CX")
        print(f"  Unmodified graphene C atoms (CX): {n_cx}")
    print()


def _build_tree_for_atoms(atoms, periodic):
    """Build a cKDTree from the current atoms structure."""
    import numpy as np
    cell = atoms.get_cell() if periodic else None
    # Graphene is periodic in xy only, never z
    pbc = np.array([True, True, False]) if periodic else None
    return _build_kdtree(atoms.get_positions(), cell, pbc)


def _pick_spatially_distributed(candidates: list, atoms: Atoms,
                                 functionalized_positions: list,
                                 min_spacing: float = 4.0,
                                 n_tries: int = 20) -> int | None:
    """Pick a candidate atom that is well-separated from recently placed groups.

    Tries up to n_tries random candidates. If a candidate's xy-distance to
    ALL functionalized positions is >= min_spacing, it is accepted immediately.
    Otherwise, falls back to the candidate with the largest minimum distance
    among those tried. This prevents tight clustering while still making
    progress.

    Parameters
    ----------
    candidates : list of int
        Atom indices to choose from.
    atoms : Atoms
        Current structure.
    functionalized_positions : list of np.ndarray
        xy-positions of already-placed functional groups.
    min_spacing : float
        Target minimum spacing in Angstrom (xy-plane).
    n_tries : int
        Number of random candidates to evaluate.

    Returns
    -------
    int or None
        Chosen atom index, or None if candidates is empty.
    """
    if not candidates:
        return None
    if not functionalized_positions:
        return random.choice(candidates)

    fg_xy = np.array(functionalized_positions)  # shape (m, 2)
    best_idx = None
    best_min_dist = -1.0

    for _ in range(min(n_tries, len(candidates))):
        idx = random.choice(candidates)
        pos_xy = atoms.positions[idx, :2]
        dists = np.linalg.norm(fg_xy - pos_xy, axis=1)
        d_min = dists.min()
        if d_min >= min_spacing:
            return idx  # Good enough, accept immediately
        if d_min > best_min_dist:
            best_min_dist = d_min
            best_idx = idx

    return best_idx


def create_go(atoms: Atoms, n_cooh: int, n_epoxy: int, n_oh: int,
              periodic: bool = False) -> Atoms:
    """Create graphene oxide by adding functional groups to pristine graphene.

    Parameters
    ----------
    atoms : Atoms
        Input pristine graphene structure with gengo metadata.
    n_cooh : int
        Number of carboxyl (-COOH) groups to add. Placed on edge atoms only.
    n_epoxy : int
        Number of epoxy (-O-) groups to add. Placed on basal plane.
    n_oh : int
        Number of hydroxyl (-OH) groups to add. Placed on basal plane.
    periodic : bool
        Whether to use periodic boundary conditions for distance checks.

    Returns
    -------
    Atoms
        Modified structure with functional groups added.

    Notes
    -----
    Distribution strategy:
    - **Layer-balanced**: For multi-layer structures, groups are distributed
      evenly across layers. The layer with the fewest placed groups is picked
      first, with random tie-breaking.
    - **Spatially spread**: Within a layer, candidate atoms are scored by their
      xy-distance to existing functional groups. Atoms far from existing groups
      are preferred (target spacing ~4 A). This avoids tight clustering while
      still allowing placement everywhere.
    - **Atomicity**: Group placement is all-or-nothing. If any atom in a group
      fails validation, the entire group is rolled back (no partial groups).
    """
    total = n_cooh + n_epoxy + n_oh
    placed_cooh = 0
    placed_epoxy = 0
    placed_oh = 0

    import time
    t_start = time.time()

    # Build tree and atom lists once, rebuild only after successful placement
    print(f"  Building neighbor tree for {len(atoms)} atoms ...", end=" ", flush=True)
    tree = _build_tree_for_atoms(atoms, periodic)
    print("done.")

    # --- Detect layers ---
    if LAYER_KEY in atoms.arrays:
        layer_ids = sorted(set(atoms.arrays[LAYER_KEY]))
    else:
        layer_ids = [0]
    n_layers = len(layer_ids)

    # Track how many groups are placed per layer for balancing
    groups_per_layer = {lid: 0 for lid in layer_ids}

    # Track xy-positions of placed functional groups for spatial spread
    fg_positions = []  # list of (x, y) arrays

    # Minimum spacing target for spatial distribution (in Angstrom).
    # ~3 CC bonds apart to avoid clustering.
    MIN_SPACING = CC_BOND * 3.0  # ~4.25 A

    # Cache edge and available atoms per layer
    edge_cache = {}      # layer_id -> list of atom indices
    available_cache = {}  # layer_id -> list of atom indices
    edge_valid = False
    available_valid = False

    def _get_layer_atoms(atom_indices, layer_id):
        """Filter atom indices to those belonging to a specific layer."""
        if n_layers == 1:
            return atom_indices
        layers = atoms.arrays[LAYER_KEY]
        return [i for i in atom_indices if layers[i] == layer_id]

    def _pick_target_layer(group_type: str) -> int:
        """Pick the layer with fewest placed groups (random tie-break).

        For COOH, only consider layers that have edge atoms.
        """
        candidates_layers = list(layer_ids)
        if group_type == "carboxyl" and edge_valid:
            # Only layers that have edge atoms
            candidates_layers = [lid for lid in layer_ids
                                 if edge_cache.get(lid)]
            if not candidates_layers:
                return layer_ids[0]  # fallback

        min_count = min(groups_per_layer[lid] for lid in candidates_layers)
        tied = [lid for lid in candidates_layers
                if groups_per_layer[lid] == min_count]
        return random.choice(tied)

    while total > 0:
        remaining = []
        if n_cooh > 0:
            remaining.append("carboxyl")
        if n_epoxy > 0:
            remaining.append("epoxy")
        if n_oh > 0:
            remaining.append("hydroxyl")

        if not remaining:
            break

        chosen = random.choice(remaining)
        placed_total = placed_cooh + placed_epoxy + placed_oh
        elapsed = time.time() - t_start
        print(f"  [{placed_total}/{placed_total + total} placed, {elapsed:.1f}s] "
              f"COOH={n_cooh} epoxy={n_epoxy} OH={n_oh}   ", end="\r", flush=True)

        if chosen == "carboxyl":
            if not edge_valid:
                t0 = time.time()
                all_edge = get_edge_atoms(atoms, GO_BONDS, periodic)
                for lid in layer_ids:
                    edge_cache[lid] = _get_layer_atoms(all_edge, lid)
                edge_valid = True
                dt = time.time() - t0
                if dt > 0.5 or placed_total == 0:
                    per_layer = ", ".join(f"L{lid}={len(edge_cache[lid])}"
                                         for lid in layer_ids)
                    print(f"\n  Edge atoms: {len(all_edge)} ({per_layer}) ({dt:.1f}s)")

            # All edge atoms across all layers
            all_edge = [i for lid in layer_ids for i in edge_cache.get(lid, [])]
            if not all_edge:
                print(f"\n  Warning: No edge atoms available for COOH placement.")
                n_cooh = 0
                total = n_cooh + n_epoxy + n_oh
                continue

            target_layer = _pick_target_layer("carboxyl")
            layer_candidates = edge_cache.get(target_layer, [])
            if not layer_candidates:
                layer_candidates = all_edge

            success = False
            for attempt in range(MAX_ATTEMPTS):
                idx = _pick_spatially_distributed(
                    layer_candidates, atoms, fg_positions, MIN_SPACING)
                if idx is None:
                    break
                atoms, placed = _add_carboxyl(atoms, idx, periodic, tree=tree)
                if placed:
                    n_cooh -= 1
                    placed_cooh += 1
                    total -= 1
                    success = True
                    fg_positions.append(atoms.positions[idx, :2].copy())
                    groups_per_layer[target_layer] = groups_per_layer.get(target_layer, 0) + 1
                    # Invalidate caches
                    tree = _build_tree_for_atoms(atoms, periodic)
                    edge_valid = False
                    available_valid = False
                    break

            if not success:
                print(f"\n  Warning: Could not place COOH after {MAX_ATTEMPTS} attempts. "
                      f"Stopping COOH placement ({n_cooh} remaining).")
                n_cooh = 0
                total = n_cooh + n_epoxy + n_oh

        elif chosen == "epoxy":
            if not available_valid:
                t0 = time.time()
                all_avail = get_available_atoms(atoms, GO_BONDS, periodic)
                for lid in layer_ids:
                    available_cache[lid] = _get_layer_atoms(all_avail, lid)
                available_valid = True
                dt = time.time() - t0
                if dt > 0.5 or placed_total == 0:
                    per_layer = ", ".join(f"L{lid}={len(available_cache[lid])}"
                                         for lid in layer_ids)
                    print(f"\n  Available atoms: {len(all_avail)} ({per_layer}) ({dt:.1f}s)")

            all_avail = [i for lid in layer_ids for i in available_cache.get(lid, [])]
            if not all_avail:
                print(f"\n  Warning: No available atoms for epoxy placement.")
                n_epoxy = 0
                total = n_cooh + n_epoxy + n_oh
                continue

            target_layer = _pick_target_layer("epoxy")
            layer_candidates = available_cache.get(target_layer, [])
            if not layer_candidates:
                layer_candidates = all_avail

            success = False
            for attempt in range(MAX_ATTEMPTS):
                idx = _pick_spatially_distributed(
                    layer_candidates, atoms, fg_positions, MIN_SPACING)
                if idx is None:
                    break
                atoms, placed = _add_epoxy(atoms, idx, periodic, tree=tree)
                if placed:
                    n_epoxy -= 1
                    placed_epoxy += 1
                    total -= 1
                    success = True
                    fg_positions.append(atoms.positions[idx, :2].copy())
                    groups_per_layer[target_layer] = groups_per_layer.get(target_layer, 0) + 1
                    # Invalidate caches
                    tree = _build_tree_for_atoms(atoms, periodic)
                    edge_valid = False
                    available_valid = False
                    break

            if not success:
                print(f"\n  Warning: Could not place epoxy after {MAX_ATTEMPTS} attempts. "
                      f"Stopping epoxy placement ({n_epoxy} remaining).")
                n_epoxy = 0
                total = n_cooh + n_epoxy + n_oh

        elif chosen == "hydroxyl":
            if not available_valid:
                t0 = time.time()
                all_avail = get_available_atoms(atoms, GO_BONDS, periodic)
                for lid in layer_ids:
                    available_cache[lid] = _get_layer_atoms(all_avail, lid)
                available_valid = True
                dt = time.time() - t0
                if dt > 0.5 or placed_total == 0:
                    per_layer = ", ".join(f"L{lid}={len(available_cache[lid])}"
                                         for lid in layer_ids)
                    print(f"\n  Available atoms: {len(all_avail)} ({per_layer}) ({dt:.1f}s)")

            all_avail = [i for lid in layer_ids for i in available_cache.get(lid, [])]
            if not all_avail:
                print(f"\n  Warning: No available atoms for OH placement.")
                n_oh = 0
                total = n_cooh + n_epoxy + n_oh
                continue

            target_layer = _pick_target_layer("hydroxyl")
            layer_candidates = available_cache.get(target_layer, [])
            if not layer_candidates:
                layer_candidates = all_avail

            success = False
            for attempt in range(MAX_ATTEMPTS):
                idx = _pick_spatially_distributed(
                    layer_candidates, atoms, fg_positions, MIN_SPACING)
                if idx is None:
                    break
                atoms, placed = _add_hydroxyl(atoms, idx, periodic, tree=tree)
                if placed:
                    n_oh -= 1
                    placed_oh += 1
                    total -= 1
                    success = True
                    fg_positions.append(atoms.positions[idx, :2].copy())
                    groups_per_layer[target_layer] = groups_per_layer.get(target_layer, 0) + 1
                    # Invalidate caches
                    tree = _build_tree_for_atoms(atoms, periodic)
                    edge_valid = False
                    available_valid = False
                    break

            if not success:
                print(f"\n  Warning: Could not place OH after {MAX_ATTEMPTS} attempts. "
                      f"Stopping OH placement ({n_oh} remaining).")
                n_oh = 0
                total = n_cooh + n_epoxy + n_oh

    elapsed = time.time() - t_start
    print()  # Clear the \r line
    print(f"Placed: COOH={placed_cooh}, epoxy={placed_epoxy}, OH={placed_oh} ({elapsed:.1f}s)")

    # Report per-layer distribution
    if n_layers > 1:
        print(f"  Per-layer distribution:")
        for lid in layer_ids:
            print(f"    Layer {lid}: {groups_per_layer[lid]} groups")

    _report_composition(atoms, placed_cooh, placed_epoxy, placed_oh)

    return atoms
