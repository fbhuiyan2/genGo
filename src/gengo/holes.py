"""
Hole generation in graphene sheets.

Creates holes by removing atoms from a pristine graphene layer.
Supports:
    - Unidirectional (u): removes one neighbor at a time, creating linear holes
    - Multidirectional (m): removes all neighbors, creating oval-shaped holes
    - Interior (i): holes do not touch edges or other holes
    - Exterior (e): holes can touch edges

The algorithm is the same as GOPY (Muraru et al., SoftwareX 2020).
For periodic systems, bonds wrap around boundaries so hole growth can
cross periodic boundaries.
"""

import random

import numpy as np
from ase import Atoms

from .core import (
    GO_BONDS,
    ATOM_TYPE_KEY,
    RESIDUE_NAME_KEY,
    RESIDUE_NUM_KEY,
    LAYER_KEY,
    get_atom_type,
    identify_bonds,
    is_connected_to_functional_group,
    init_metadata_arrays,
)


def _get_contour(atoms: Atoms, periodic: bool = False) -> list:
    """Get indices of atoms on the contour/edge of the graphene layer.

    Includes edge atoms and their close neighbors (within 2 bonds),
    same logic as GOPY's get_contour.
    """
    # Initial edge atoms: those with < 3 bonds and not connected to functional groups
    initial = []
    for i in range(len(atoms)):
        atype = get_atom_type(atoms, i)
        if atype != "CX":
            continue
        bonds = identify_bonds(atoms, i, GO_BONDS, periodic)
        if 0 < len(bonds) < 3:
            if not is_connected_to_functional_group(atoms, i, GO_BONDS, periodic):
                initial.append(i)

    initial_set = set(initial)

    # Extra atoms: those with a neighbor that has a neighbor in 'initial'
    extra_1 = set()
    for i in range(len(atoms)):
        atype = get_atom_type(atoms, i)
        if atype != "CX":
            continue
        bonds = identify_bonds(atoms, i, GO_BONDS, periodic)
        for neighbor_idx, _ in bonds:
            bonds2 = identify_bonds(atoms, neighbor_idx, GO_BONDS, periodic)
            for n2_idx, _ in bonds2:
                if n2_idx in initial_set:
                    extra_1.add(i)

    # Extra atoms: those with 2 or more neighbors in 'initial'
    extra_2 = set()
    for i in range(len(atoms)):
        atype = get_atom_type(atoms, i)
        if atype != "CX":
            continue
        bonds = identify_bonds(atoms, i, GO_BONDS, periodic)
        count = sum(1 for b_idx, _ in bonds if b_idx in initial_set)
        if count >= 2 and i not in initial_set:
            extra_2.add(i)

    return list(initial_set | extra_1 | extra_2)


def _get_available(atoms: Atoms, periodic: bool = False) -> list:
    """Get indices of all CX atoms not connected to functional groups."""
    available = []
    for i in range(len(atoms)):
        atype = get_atom_type(atoms, i)
        if atype != "CX":
            continue
        if not is_connected_to_functional_group(atoms, i, GO_BONDS, periodic):
            available.append(i)
    return available


def _hole_cleanup(atoms: Atoms, periodic: bool = False) -> Atoms:
    """Remove isolated fragments of <= 6 atoms after hole creation.

    Same algorithm as GOPY's hole_cleanup.
    """
    # Get all CX atom indices
    cx_indices = [i for i in range(len(atoms)) if get_atom_type(atoms, i) == "CX"]
    remaining = set(cx_indices)
    remove_set = set()

    while remaining:
        # BFS from an arbitrary remaining atom
        start = next(iter(remaining))
        component = {start}
        frontier = [start]

        while frontier:
            next_frontier = []
            for idx in frontier:
                bonds = identify_bonds(atoms, idx, GO_BONDS, periodic)
                for neighbor_idx, _ in bonds:
                    if neighbor_idx in remaining and neighbor_idx not in component:
                        atype = get_atom_type(atoms, neighbor_idx)
                        if atype == "CX":
                            component.add(neighbor_idx)
                            next_frontier.append(neighbor_idx)
            frontier = next_frontier

        # If component is too small, mark for removal
        if len(component) < 6:
            remove_set.update(component)

        remaining -= component

    if not remove_set:
        return atoms

    # Remove marked atoms
    keep_mask = np.ones(len(atoms), dtype=bool)
    for idx in remove_set:
        keep_mask[idx] = False

    new_atoms = atoms[keep_mask]
    for key in [ATOM_TYPE_KEY, RESIDUE_NAME_KEY, RESIDUE_NUM_KEY, LAYER_KEY]:
        if key in atoms.arrays:
            new_atoms.arrays[key] = atoms.arrays[key][keep_mask].copy()

    new_atoms.set_cell(atoms.get_cell())
    new_atoms.set_pbc(atoms.get_pbc())

    print(f"  Cleanup: removed {len(remove_set)} isolated atoms")
    return new_atoms


def generate_holes(atoms: Atoms, n_holes: int, size_range: tuple,
                   mode: str = "u", edge_mode: str = "i",
                   cleanup: bool = False,
                   periodic: bool = False) -> Atoms:
    """Generate holes in a graphene sheet.

    Parameters
    ----------
    atoms : Atoms
        Input graphene structure with gengo metadata.
    n_holes : int
        Number of holes to create.
    size_range : tuple of (int, int)
        Min and max number of atoms to remove per hole.
    mode : str
        "u" for unidirectional (linear holes) or "m" for multidirectional (oval holes).
    edge_mode : str
        "i" for interior (holes don't touch edges) or "e" for exterior (can touch edges).
    cleanup : bool
        If True, remove isolated fragments <= 6 atoms after hole creation.
    periodic : bool
        Whether to use periodic boundary conditions.

    Returns
    -------
    Atoms
        Modified structure with holes.
    """
    contour = set(_get_contour(atoms, periodic))
    holes_placed = 0
    atoms_to_remove = set()

    while holes_placed < n_holes:
        print(f"  Holes: {holes_placed} placed out of {n_holes}", end="\r")

        hole_size = random.randint(size_range[0], size_range[1])
        success = False

        for attempt in range(50):
            # Get available atoms
            if edge_mode == "i":
                available = [i for i in _get_available(atoms, periodic)
                             if i not in contour and i not in atoms_to_remove]
            else:
                available = [i for i in _get_available(atoms, periodic)
                             if i not in atoms_to_remove]

            if not available:
                print(f"\n  Warning: No available atoms for hole placement.")
                holes_placed = n_holes  # Force exit
                break

            hole_atoms = []
            start_idx = random.choice(available)
            hole_atoms.append(start_idx)

            if mode == "u":
                # Unidirectional: follow one path
                source = start_idx
                remaining_to_remove = hole_size - 1

                while remaining_to_remove > 0:
                    bonds = identify_bonds(atoms, source, GO_BONDS, periodic)
                    neighbors = [b[0] for b in bonds
                                 if b[0] not in hole_atoms and b[0] not in atoms_to_remove]

                    if edge_mode == "i":
                        neighbors = [n for n in neighbors if n not in contour]

                    if neighbors:
                        choice = random.choice(neighbors)
                        hole_atoms.append(choice)
                        remaining_to_remove -= 1
                        source = choice
                    else:
                        # Try to find another branch point
                        found = False
                        for ha in hole_atoms:
                            b = identify_bonds(atoms, ha, GO_BONDS, periodic)
                            nb = [x[0] for x in b
                                  if x[0] not in hole_atoms and x[0] not in atoms_to_remove]
                            if edge_mode == "i":
                                nb = [n for n in nb if n not in contour]
                            if nb:
                                source = ha
                                found = True
                                break
                        if not found:
                            break

            elif mode == "m":
                # Multidirectional: grow in all directions
                frontier = [start_idx]
                remaining_to_remove = hole_size - 1

                while remaining_to_remove > 0 and frontier:
                    next_frontier = []
                    for source_atom in frontier:
                        bonds = identify_bonds(atoms, source_atom, GO_BONDS, periodic)
                        neighbors = [b[0] for b in bonds
                                     if b[0] not in hole_atoms
                                     and b[0] not in atoms_to_remove]

                        if edge_mode == "i":
                            neighbors = [n for n in neighbors if n not in contour]

                        for nb in neighbors:
                            if remaining_to_remove > 0:
                                hole_atoms.append(nb)
                                next_frontier.append(nb)
                                remaining_to_remove -= 1

                    frontier = next_frontier

            if len(hole_atoms) == hole_size:
                atoms_to_remove.update(hole_atoms)

                # Update contour for interior mode
                if edge_mode == "i":
                    # Find atoms bordering the new hole
                    for ha in hole_atoms:
                        bonds = identify_bonds(atoms, ha, GO_BONDS, periodic)
                        for neighbor_idx, _ in bonds:
                            if neighbor_idx not in atoms_to_remove:
                                contour.add(neighbor_idx)
                                # Also add neighbors of contour atoms
                                b2 = identify_bonds(atoms, neighbor_idx, GO_BONDS, periodic)
                                for n2, _ in b2:
                                    if n2 not in atoms_to_remove:
                                        contour.add(n2)

                holes_placed += 1
                success = True
                break

        if not success and holes_placed < n_holes:
            print(f"\n  Warning: Could not place hole {holes_placed + 1}. "
                  f"Stopping hole generation.")
            break

    print(f"  Holes: {holes_placed} placed out of {n_holes}")

    # Remove atoms
    if atoms_to_remove:
        keep_mask = np.ones(len(atoms), dtype=bool)
        # Only remove CX atoms that are in the contour-allowed set
        for idx in atoms_to_remove:
            if idx < len(atoms) and idx not in contour or edge_mode == "e":
                keep_mask[idx] = False

        # Actually just remove all atoms marked for removal
        keep_mask = np.ones(len(atoms), dtype=bool)
        for idx in atoms_to_remove:
            keep_mask[idx] = False

        new_atoms = atoms[keep_mask]
        for key in [ATOM_TYPE_KEY, RESIDUE_NAME_KEY, RESIDUE_NUM_KEY, LAYER_KEY]:
            if key in atoms.arrays:
                new_atoms.arrays[key] = atoms.arrays[key][keep_mask].copy()

        new_atoms.set_cell(atoms.get_cell())
        new_atoms.set_pbc(atoms.get_pbc())
        atoms = new_atoms

        print(f"  Removed {len(atoms_to_remove)} atoms total")

    # Renumber atoms
    if RESIDUE_NUM_KEY in atoms.arrays:
        atoms.arrays[RESIDUE_NUM_KEY] = np.arange(1, len(atoms) + 1, dtype=int)

    if cleanup:
        print("  Running cleanup...")
        atoms = _hole_cleanup(atoms, periodic)

    print(f"  Final atom count: {len(atoms)}")
    return atoms
