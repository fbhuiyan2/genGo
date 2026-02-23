"""
I/O helpers using ASE for reading and writing atomic structures.

Handles format detection, metadata preservation for PDB files,
LAMMPS data file styles, and provides a clean interface for all
supported ASE formats.
"""

import os
import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from .core import (
    ATOM_TYPE_KEY, RESIDUE_NAME_KEY, RESIDUE_NUM_KEY, LAYER_KEY,
    init_metadata_arrays,
)

# Map of common extensions that ASE doesn't auto-detect
FORMAT_HINTS = {
    ".lmp": "lammps-data",
    ".data": "lammps-data",
    ".lammps": "lammps-data",
}

# gengo format shorthands for LAMMPS data files.
# Maps user-facing --format value to LAMMPS atom_style.
LAMMPS_STYLES = {
    "lmp": "atomic",
    "lmp-atomic": "atomic",
    "lmp-full": "full",
    "lmp-charge": "charge",
}


def _resolve_format(fname: str, user_format: str = None) -> tuple:
    """Resolve the user-facing format into an ASE format string and extra kwargs.

    Returns
    -------
    tuple of (str | None, dict)
        (ase_format, extra_write_kwargs).
        ase_format is the string to pass to ase.io.write(format=...).
        extra_write_kwargs contains atom_style, specorder, etc. for LAMMPS.
    """
    # Check gengo LAMMPS shorthands first
    if user_format and user_format in LAMMPS_STYLES:
        atom_style = LAMMPS_STYLES[user_format]
        return "lammps-data", {"atom_style": atom_style}

    # Fall back to extension / direct ASE format string
    fmt = _infer_format(fname, user_format)

    # If it resolved to lammps-data (from extension), default to atomic style
    if fmt == "lammps-data":
        return "lammps-data", {"atom_style": "atomic"}

    return fmt, {}


def _infer_format(fname: str, user_format: str = None) -> str | None:
    """Infer ASE format string from filename or user override.

    Returns None to let ASE auto-detect when possible.
    """
    if user_format:
        return user_format

    ext = os.path.splitext(fname)[1].lower()
    if ext in FORMAT_HINTS:
        return FORMAT_HINTS[ext]

    # Let ASE auto-detect for standard extensions (.pdb, .xyz, .cif, etc.)
    return None


def read_structure(fname: str, format: str = None) -> Atoms:
    """Read an atomic structure file and initialize gengo metadata.

    If the file is a PDB with GOPY-style naming (CX/CY atoms in GGG/C1A/E1A/H1A
    residues), metadata is parsed from the PDB fields. Otherwise, default
    metadata is initialized.

    Parameters
    ----------
    fname : str
        Path to the input file.
    format : str, optional
        ASE format string. If None, inferred from file extension.

    Returns
    -------
    Atoms
        ASE Atoms object with gengo metadata arrays initialized.
    """
    # Resolve gengo shorthands (lmp, lmp-full, etc.) to ASE format
    if format and format in LAMMPS_STYLES:
        read_fmt = "lammps-data"
    else:
        read_fmt = _infer_format(fname, format)

    kwargs = {}
    if read_fmt:
        kwargs["format"] = read_fmt

    atoms = ase_read(fname, **kwargs)

    # Remove LAMMPS-specific arrays that ASE stores on read.
    # The 'type' array in particular causes problems: when new atoms are
    # appended (e.g., during GO functionalization), they get type=0 by default.
    # On write, ASE sees 'type' in atoms.arrays and uses it instead of
    # specorder, producing invalid type 0 entries in the LAMMPS output.
    for lammps_key in ("type", "id", "masses"):
        if lammps_key in atoms.arrays:
            del atoms.arrays[lammps_key]

    # Check if metadata already exists (e.g., from a previously saved gengo file)
    if ATOM_TYPE_KEY in atoms.arrays:
        return atoms

    # Try to parse GOPY-style PDB metadata
    ext = os.path.splitext(fname)[1].lower()
    if ext in (".pdb",) and read_fmt in (None, "proteindatabank", "pdb"):
        atoms = _parse_pdb_metadata(atoms, fname)
    else:
        # Initialize default metadata for non-PDB files
        # Assume all atoms are graphene carbon unless otherwise specified
        _init_default_metadata(atoms)

    return atoms


def _parse_pdb_metadata(atoms: Atoms, fname: str) -> Atoms:
    """Parse GOPY-style atom names and residue names from a PDB file.

    GOPY PDB format:
    - Atom names: CX, CY, CZ, C4, OJ, OK, OE, OL, HK
    - Residue names: GGG (graphene), C1A (carboxyl), E1A (epoxy), H1A (hydroxyl)
    """
    n = len(atoms)
    atom_types = np.array(["CX"] * n, dtype="U4")
    residue_names = np.array(["GGG"] * n, dtype="U3")
    residue_numbers = np.arange(1, n + 1, dtype=int)
    layers = np.zeros(n, dtype=int)

    # Parse the raw PDB file to extract atom names and residue info
    try:
        with open(fname, "r") as f:
            atom_idx = 0
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if atom_idx >= n:
                        break
                    # PDB columns (standard):
                    # 12-16: atom name
                    # 17-20: residue name (or 17-19 in some variants)
                    # 22-26: residue sequence number
                    # But GOPY uses non-standard spacing, so parse by splitting
                    parts = line.split()
                    if len(parts) >= 6:
                        # GOPY format: ATOM num name resname resnum x y z ...
                        aname = parts[2].strip()
                        rname = parts[3].strip()
                        try:
                            rnum = int(parts[4])
                        except (ValueError, IndexError):
                            rnum = atom_idx + 1

                        # Handle VMD-generated graphene: "C   GRA X" -> "CX  GGG"
                        if aname == "C" and rname == "GRA":
                            aname = "CX"
                            rname = "GGG"

                        atom_types[atom_idx] = aname[:4]
                        residue_names[atom_idx] = rname[:3]
                        residue_numbers[atom_idx] = rnum

                    atom_idx += 1
    except Exception:
        # Fall back to defaults if parsing fails
        pass

    # Detect layers from z-coordinates (PDB doesn't store layer info)
    layers = _detect_layers_from_z(atoms)

    init_metadata_arrays(atoms, atom_types, residue_names, residue_numbers, layers)
    return atoms


def _detect_layers_from_z(atoms: Atoms) -> np.ndarray:
    """Detect graphene layers from z-coordinate clustering.

    For multi-layer graphene, atoms cluster at distinct z-values separated
    by the interlayer distance (typically >= 3.35 A). This function groups
    atoms into layers by clustering their z-coordinates.

    Returns an integer array of layer indices (0-based).
    """
    n = len(atoms)
    if n == 0:
        return np.zeros(0, dtype=int)

    z = atoms.positions[:, 2]
    # Sort unique z-values and cluster with tolerance
    z_sorted = np.sort(z)
    # Group z-values: two atoms are in the same layer if their z differs by < 2 A
    # (half the minimum interlayer distance of 3.35 A)
    tol = 2.0
    layer_z_values = [z_sorted[0]]
    for zi in z_sorted[1:]:
        if zi - layer_z_values[-1] > tol:
            layer_z_values.append(zi)
        # Update running average of current layer's z
        # (not needed for clustering, just for the reference z)

    # Re-cluster: find distinct z levels
    layer_z_values = []
    current_cluster = [z_sorted[0]]
    for zi in z_sorted[1:]:
        if zi - np.mean(current_cluster) > tol:
            layer_z_values.append(np.mean(current_cluster))
            current_cluster = [zi]
        else:
            current_cluster.append(zi)
    layer_z_values.append(np.mean(current_cluster))

    if len(layer_z_values) <= 1:
        return np.zeros(n, dtype=int)

    # Assign each atom to the nearest layer z-value
    layer_z_arr = np.array(layer_z_values)
    layers = np.zeros(n, dtype=int)
    for i in range(n):
        layers[i] = np.argmin(np.abs(layer_z_arr - z[i]))

    return layers


def _init_default_metadata(atoms: Atoms):
    """Initialize default metadata for non-PDB structures.

    Detects layers from z-coordinate clustering for multi-layer graphene.
    """
    n = len(atoms)
    atom_types = []
    for symbol in atoms.get_chemical_symbols():
        if symbol == "C":
            atom_types.append("CX")
        elif symbol == "O":
            atom_types.append("OL")  # generic oxygen
        elif symbol == "H":
            atom_types.append("HK")  # generic hydrogen
        elif symbol == "N":
            atom_types.append("N3")  # generic nitrogen
        else:
            atom_types.append(symbol[:4])

    layers = _detect_layers_from_z(atoms)

    init_metadata_arrays(
        atoms,
        atom_types=np.array(atom_types, dtype="U4"),
        residue_names=np.array(["GGG"] * n, dtype="U3"),
        residue_numbers=np.arange(1, n + 1, dtype=int),
        layers=layers,
    )


def write_structure(fname: str, atoms: Atoms, format: str = None):
    """Write an atomic structure to a file using ASE.

    For PDB output, ensures proper GOPY-compatible formatting with
    atom names and residue names from metadata arrays.

    For LAMMPS data output, sets specorder (alphabetically sorted species)
    and atom_style. Use --format with one of the LAMMPS shorthands:
        lmp          -> atom_style='atomic'  (default)
        lmp-atomic   -> atom_style='atomic'
        lmp-full     -> atom_style='full'
        lmp-charge   -> atom_style='charge'

    Parameters
    ----------
    fname : str
        Path to the output file.
    atoms : Atoms
        ASE Atoms object with gengo metadata.
    format : str, optional
        ASE format string or gengo shorthand. If None, inferred from extension.
    """
    ase_fmt, extra_kwargs = _resolve_format(fname, format)

    ext = os.path.splitext(fname)[1].lower()
    is_pdb = ext == ".pdb" or ase_fmt in ("proteindatabank", "pdb")
    is_lammps = ase_fmt == "lammps-data"

    if is_pdb and ATOM_TYPE_KEY in atoms.arrays:
        _write_pdb_with_metadata(fname, atoms)
    elif is_lammps:
        _write_lammps(fname, atoms, **extra_kwargs)
    else:
        # For extXYZ and other formats, strip ALL non-essential arrays before
        # writing. ASE serializes all atoms.arrays entries as extra columns in
        # extXYZ, which can cause parse errors on read-back when mixed types
        # appear (e.g., "GGG" strings next to integers).
        # Keep only the fundamental 'numbers' and 'positions' arrays.
        atoms_out = atoms.copy()
        keep_keys = {"numbers", "positions"}
        for key in list(atoms_out.arrays.keys()):
            if key not in keep_keys:
                del atoms_out.arrays[key]
        kwargs = {}
        if ase_fmt:
            kwargs["format"] = ase_fmt
        ase_write(fname, atoms_out, **kwargs)


def _write_lammps(fname: str, atoms: Atoms, atom_style: str = "atomic"):
    """Write a LAMMPS data file with deterministic atom type ordering.

    Species are sorted alphabetically to produce a consistent type mapping
    across runs (e.g., C=1, H=2, O=3 for graphene oxide).

    Ensures the cell is a valid 3D box (LAMMPS requires non-zero dimensions
    in all directions). For flat sheets (z=0), a vacuum slab is added.
    """
    atoms = atoms.copy()  # avoid mutating the original

    # Ensure a valid 3D cell for LAMMPS
    cell = atoms.get_cell()
    lengths = cell.lengths()
    pos = atoms.get_positions()

    needs_fix = False
    new_lengths = list(lengths)
    for dim in range(3):
        if new_lengths[dim] < 1e-6:
            # Dimension is degenerate -- set box to span atoms + vacuum
            lo = pos[:, dim].min() if len(pos) > 0 else 0.0
            hi = pos[:, dim].max() if len(pos) > 0 else 0.0
            span = hi - lo
            new_lengths[dim] = max(span + 20.0, 20.0)
            needs_fix = True

    if needs_fix:
        atoms.set_cell(new_lengths)
        # Don't set PBC -- keep whatever was set

    specorder = sorted(set(atoms.get_chemical_symbols()))
    ase_write(
        fname, atoms,
        format="lammps-data",
        specorder=specorder,
        atom_style=atom_style,
        masses=True,
    )


def _write_pdb_with_metadata(fname: str, atoms: Atoms):
    """Write a PDB file preserving GOPY-style atom/residue naming.

    Writes proper PDB format that ASE and other tools can read,
    while maintaining the GOPY naming conventions.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    with open(fname, "w") as f:
        # Write CRYST1 record if cell is defined and non-zero
        if pbc.any() and cell.lengths().any():
            a, b, c = cell.lengths()
            alpha, beta, gamma = cell.angles()
            f.write(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                    f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n")

        positions = atoms.get_positions()
        atom_types = atoms.arrays.get(ATOM_TYPE_KEY, None)
        residue_names = atoms.arrays.get(RESIDUE_NAME_KEY, None)
        residue_numbers = atoms.arrays.get(RESIDUE_NUM_KEY, None)

        for i in range(len(atoms)):
            serial = i + 1
            aname = atom_types[i] if atom_types is not None else atoms[i].symbol
            rname = residue_names[i] if residue_names is not None else "UNK"
            rnum = residue_numbers[i] if residue_numbers is not None else 1
            x, y, z = positions[i]
            element = atoms[i].symbol

            # Format atom name: left-justify in 4-char field if <= 3 chars,
            # otherwise fill all 4 chars
            aname_str = str(aname)
            if len(aname_str) <= 3:
                aname_fmt = f" {aname_str:<3s}"
            else:
                aname_fmt = f"{aname_str:<4s}"

            rname_str = str(rname)[:3]
            rnum_int = int(rnum) % 10000  # PDB limit

            # Standard PDB ATOM record format
            line = (
                f"ATOM  {serial:5d} {aname_fmt:4s} {rname_str:>3s}  "
                f"{rnum_int:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}  "
            )
            f.write(line + "\n")

        f.write("END\n")
