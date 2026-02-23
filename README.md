# gengo

A tool for generating graphene and graphene oxide (GO) structures with periodic boundary conditions and multi-layer support. Outputs to any file format supported by ASE (PDB, XYZ, CIF, LAMMPS-data, etc.).

Built on the GO functionalization algorithm from [GOPY](https://github.com/Iourarum/GOPY), refactored to use [ASE](https://wiki.fysik.dtu.dk/ase/) for atomic structure handling and file I/O.

## Features

- **Pristine graphene generation** with configurable X/Y dimensions
- **Periodic boundary conditions** (`-p`) in X and Y
- **Multi-layer graphene** (`-n`) with AB (Bernal) stacking, configurable interlayer distance
- **Graphene oxide** with three functional group types:
  - Carboxyl (-COOH) — edge atoms only
  - Epoxy (-O- / C-O-C) — basal plane
  - Hydroxyl (-OH) — basal plane
- **Layer-balanced distribution** — functional groups are distributed evenly across layers in multi-layer structures
- **Spatial spread** — functional groups are placed with minimum spacing to avoid clustering
- **Hole generation** — unidirectional or multidirectional, with optional cleanup
- **Composition calculator** — atomic % and weight % of C, O, H, with C/O ratios
- **Multiple output formats** — PDB, XYZ, extXYZ, CIF, LAMMPS-data, and anything else ASE supports
- **Examples** — Several example pristine graphene (PG) and graphene oxide (GO) structures are uploaded in the test_output folders

## Installation

Requires Python 3.10+.

### With pip

```bash
git clone <this-repo-url>
cd genGO
pip install -e .
```

### With Poetry

```bash
git clone <this-repo-url>
cd genGO
poetry install
poetry shell
```

### Dependencies

- [ase](https://wiki.fysik.dtu.dk/ase/) >= 3.23
- numpy >= 1.24
- scipy >= 1.10

## Usage

### Generate pristine graphene

```bash
# Non-periodic, 30x30 Angstrom sheet
gengo PG 30 30 graphene.pdb

# Periodic boundary conditions in X and Y
gengo PG 30 30 graphene.pdb -p

# Multi-layer (2 layers, AB stacking, default 3.35 A spacing)
gengo PG 30 30 bilayer.pdb -p -n 2

# 3 layers with custom interlayer distance (minimum 3.35 A)
gengo PG 50 50 trilayer.pdb -p -n 3 -d 4.0

# Output as XYZ instead of PDB
gengo PG 30 30 graphene.xyz
```

When `-p` is used, the sheet dimensions are snapped to exact multiples of the graphene lattice repeat unit so the structure tiles correctly under periodic boundary conditions. The tool reports both the requested and actual dimensions.

### Generate graphene oxide

```bash
# Add 10 COOH, 20 epoxy, 30 OH groups to a pristine graphene sheet
gengo GO graphene.pdb go.pdb --COOH 10 --COC 20 --OH 30

# Periodic GO — epoxy and OH only (a defect-free periodic sheet has no edges)
gengo GO graphene_periodic.pdb go_periodic.pdb --COC 20 --OH 30 -p

# Periodic GO with holes — COOH can be placed at internal edges around holes
gengo hole graphene_periodic.pdb holey_periodic.pdb -N 3 --range 5 10 --mode m --edge i -p
gengo GO holey_periodic.pdb go_holey.pdb --COOH 5 --COC 20 --OH 30 -p

# LAMMPS charge format output
gengo GO graphene.lmp go.lmp --COOH 30 --COC 200 --OH 500 -p --format lmp-charge
```

- `--COOH` groups are placed only on edge atoms (atoms with fewer than 3 C-C bonds). Edge detection is based on the actual bond count of each atom, not the `-p` flag. A defect-free periodic sheet has no edges (all boundary atoms have 3 bonds through periodic images), but a periodic sheet with holes, slits, or other defects will have internal edges where COOH groups can be placed.
- `--COC` (epoxy) and `--OH` (hydroxyl) groups are placed on the basal plane, above or below the sheet.
- **Spatial distribution**: Groups are placed with a target minimum spacing (~4.25 A) to avoid clustering. Candidate atoms far from existing functional groups are preferred.
- **Layer balancing**: For multi-layer structures, groups are distributed evenly across all layers. The layer with the fewest placed groups is always selected next.
- **Atomicity**: Group placement is all-or-nothing. If any atom in a functional group fails bond validation, the entire group is rolled back (no partial groups are left behind).
- After placement, a composition report is printed with atom counts, atomic %, weight %, and C/O ratios.

### Generate holes

```bash
# 5 holes, 10-20 atoms each, multidirectional, interior only, with cleanup
gengo hole graphene.pdb holey.pdb -N 5 --range 10 20 --mode m --edge i --cleanup
```

| Flag | Values | Description |
|------|--------|-------------|
| `-N` | integer | Number of holes |
| `--range` | min max | Atom count range per hole |
| `--mode` | `u` or `m` | Unidirectional (linear) or multidirectional (oval) |
| `--edge` | `i` or `e` | Interior (no edge touching) or exterior (can touch edges) |
| `--cleanup` | flag | Remove isolated fragments of 6 or fewer atoms |

### Calculate composition

```bash
# Current composition of an existing structure
gengo calc go.pdb

# Predict composition after hypothetically adding groups to a pristine sheet
gengo calc graphene.pdb --COOH 10 --COC 20 --OH 50
```

Reports both **atomic %** (atom count fraction) and **weight %** (mass fraction) for C, O, and H, along with C/O ratios in both atomic and mass terms. When `--COOH`, `--COC`, or `--OH` are specified, shows both the current and predicted compositions.

Example output:

```
--- Predicted Composition (with additions) ---
  Total atoms: 4896
  Total mass:  63644.8 amu

  Element   Count       at%       wt%
  -------- ------   -------   -------
  C          3006     61.4%     56.7%
  O          1710     34.9%     43.0%
  H           180      3.7%      0.3%

  C/O ratio (atomic): 1.76
  C/O ratio (mass):   1.32

  Hypothetical additions:
    COOH:  30 groups (+30C, +60O, +30H)
    Epoxy: 1500 groups (+1500O)
    OH:    150 groups (+150O, +150H)
```

**Atom contributions per functional group:**

| Group | Atoms added | Details |
|-------|-------------|---------|
| COOH  | +1 C, +2 O, +1 H | C4, OJ (=O), OK (-O-), HK |
| Epoxy (COC) | +1 O | OE (bridging oxygen between two C) |
| OH    | +1 O, +1 H | OL, HK |

### Output format

The output format is inferred from the file extension for standard types (`.pdb`, `.xyz`, `.cif`, etc.). To override or to use LAMMPS data files, pass `--format`:

```bash
gengo PG 30 30 graphene.xyz                    # inferred from .xyz
gengo PG 30 30 graphene.cif                    # inferred from .cif
gengo PG 30 30 graphene.lmp --format lmp       # LAMMPS data, atomic style
```

#### LAMMPS data files

ASE does not auto-detect `.lmp` as a LAMMPS data file, so a `--format` flag is always required. gengo provides shorthand format values that map to LAMMPS atom styles:

| `--format` value | LAMMPS `atom_style` | Columns |
|------------------|---------------------|---------|
| `lmp`            | `atomic`            | id type x y z |
| `lmp-atomic`     | `atomic`            | id type x y z |
| `lmp-full`       | `full`              | id mol-id type charge x y z |
| `lmp-charge`     | `charge`            | id type charge x y z |

You can also pass `--format lammps-data` directly, which defaults to `atomic` style.

```bash
# Pristine graphene as LAMMPS atomic data
gengo PG 30 30 graphene.lmp --format lmp

# GO as LAMMPS charge style
gengo GO graphene.lmp go.lmp --COOH 10 --COC 20 --OH 30 --format lmp-charge
```

**Atom type ordering:** Species are sorted alphabetically and assigned LAMMPS type IDs in that order. For graphene oxide this gives: C=1, H=2, O=3. For pristine graphene (carbon only): C=1.

**Charges:** gengo does not assign partial charges. The charge column in `lmp-full` and `lmp-charge` output will be zero for all atoms. Assign charges in your LAMMPS input script or with a separate tool.

**Converting between atom styles:** If you need to convert an existing file to a different atom style, you can use the ASE command-line tool directly:

```bash
ase convert -i lammps-data -o lammps-data \
    --write-args "atom_style='full'" \
    -- input.lmp output.lmp
```

## Periodic boundary conditions

When `-p` is enabled:

- The graphene lattice is tiled using an orthogonal supercell containing 4 atoms with cell vectors 2.456 x 4.254 A. This is **not** the primitive hexagonal unit cell of graphene (space group P6_3/mmc, a = 2.461 A, c = 6.708 A, Z = 4) but an equivalent rectangular representation that tiles seamlessly in Cartesian coordinates. The C-C bond length used is 1.418 A (from GOPY), giving a = 1.418 x sqrt(3) = 2.456 A. The experimental crystallographic value is a = 2.461 A (C-C = 1.421 A); the difference is ~0.2%.
- All bond/distance checks use the minimum image convention, meaning atoms near cell boundaries correctly detect bonds to their periodic images.
- Edge detection is based on actual bond counts, not on the `-p` flag. Boundary atoms with 3 bonds through periodic images are **not** classified as edges. However, atoms around holes, slits, notches, or other internal defects will still have fewer than 3 bonds and are correctly identified as edges -- even in a periodic sheet. This means COOH groups can be placed at internal defect edges under PBC.
- The PDB output includes a `CRYST1` record with the cell dimensions.

## Multi-layer support

When `-n` is greater than 1:

- Layers follow ABAB (Bernal) stacking, the same as graphite.
- Layer B is offset by (1.228, 0.709) A relative to layer A, placing B-sublattice atoms above the centers of A-layer hexagons.
- The interlayer C-C distance defaults to 3.35 A (graphite equilibrium). Values below 3.35 A are rejected.
- For GO generation on multi-layer structures, functional groups are **distributed evenly across all layers**. The algorithm tracks the number of groups placed per layer and always targets the layer with the fewest groups. A per-layer distribution summary is printed after placement.
- Layer detection works across file format round-trips. When reading a structure from any format (PDB, XYZ, LAMMPS), layers are automatically detected by clustering atom z-coordinates. This means you can generate a multi-layer PG, save it, read it back, and generate GO with correct layer balancing.

## References

The graphene oxide functionalization algorithm (geometric placement of COOH, epoxy, and OH groups with distance-based bond validation) is based on:

> S. Muraru, J.S. Burns, M. Ionita, "GOPY: A tool for building 2D graphene-based computational models," *SoftwareX*, vol. 12, 100586, 2020. [doi:10.1016/j.softx.2020.100586](https://doi.org/10.1016/j.softx.2020.100586)

Original GOPY source code: [github.com/Iourarum/GOPY](https://github.com/Iourarum/GOPY)
