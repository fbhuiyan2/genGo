"""
Composition calculator for graphene / graphene oxide structures.

Can compute:
    - Current atomic composition (C, O, H percentages) of an existing structure
    - Predicted composition after hypothetically adding COOH, epoxy, and/or OH groups

Reports both atomic % (atom count fraction) and weight % (mass fraction).
"""

from ase import Atoms

# Standard atomic masses (amu / g/mol)
ATOMIC_MASS = {
    "C": 12.011,
    "O": 15.999,
    "H": 1.008,
    "N": 14.007,
}


def calculate_composition(atoms: Atoms, n_cooh: int = 0, n_epoxy: int = 0,
                          n_oh: int = 0) -> dict:
    """Calculate atomic and weight composition of a structure.

    If n_cooh, n_epoxy, n_oh are provided, predicts the composition
    that would result from adding those functional groups.

    Parameters
    ----------
    atoms : Atoms
        Input atomic structure.
    n_cooh : int
        Number of COOH groups to hypothetically add.
        Each COOH adds: 1 C, 2 O, 1 H (and converts 1 CX -> CY, no net C change there)
    n_epoxy : int
        Number of epoxy groups to hypothetically add.
        Each epoxy adds: 1 O (and converts 2 CX -> CY/CZ, no net C change)
    n_oh : int
        Number of OH groups to hypothetically add.
        Each OH adds: 1 O, 1 H (and converts 1 CX -> CY, no net C change)

    Returns
    -------
    dict
        Composition data with keys for atom counts (n_c, n_o, n_h, n_n,
        n_total), atomic percentages (at_pct_c, at_pct_o, at_pct_h, at_pct_n),
        weight percentages (wt_pct_c, wt_pct_o, wt_pct_h, wt_pct_n),
        total mass, C/O ratios (atomic and mass), and group counts.
    """
    symbols = atoms.get_chemical_symbols()
    n_c = symbols.count("C")
    n_o = symbols.count("O")
    n_h = symbols.count("H")
    n_n = symbols.count("N")

    # Add hypothetical functional groups
    # COOH: -C(=O)-OH adds C4 + OJ + OK + HK = 1C + 2O + 1H
    n_c += n_cooh * 1
    n_o += n_cooh * 2
    n_h += n_cooh * 1

    # Epoxy: -O- bridging adds OE = 1O
    n_o += n_epoxy * 1

    # OH: -OH adds OL + HK = 1O + 1H
    n_o += n_oh * 1
    n_h += n_oh * 1

    n_total = n_c + n_o + n_h + n_n

    # Atomic percentages (atom count fractions)
    at_pct_c = 100 * n_c / n_total if n_total > 0 else 0
    at_pct_o = 100 * n_o / n_total if n_total > 0 else 0
    at_pct_h = 100 * n_h / n_total if n_total > 0 else 0
    at_pct_n = 100 * n_n / n_total if n_total > 0 else 0

    # Weight (mass) percentages
    mass_c = n_c * ATOMIC_MASS["C"]
    mass_o = n_o * ATOMIC_MASS["O"]
    mass_h = n_h * ATOMIC_MASS["H"]
    mass_n = n_n * ATOMIC_MASS["N"]
    mass_total = mass_c + mass_o + mass_h + mass_n

    wt_pct_c = 100 * mass_c / mass_total if mass_total > 0 else 0
    wt_pct_o = 100 * mass_o / mass_total if mass_total > 0 else 0
    wt_pct_h = 100 * mass_h / mass_total if mass_total > 0 else 0
    wt_pct_n = 100 * mass_n / mass_total if mass_total > 0 else 0

    result = {
        "n_c": n_c,
        "n_o": n_o,
        "n_h": n_h,
        "n_n": n_n,
        "n_total": n_total,
        # Atomic %
        "at_pct_c": at_pct_c,
        "at_pct_o": at_pct_o,
        "at_pct_h": at_pct_h,
        "at_pct_n": at_pct_n,
        # Weight %
        "mass_total": mass_total,
        "wt_pct_c": wt_pct_c,
        "wt_pct_o": wt_pct_o,
        "wt_pct_h": wt_pct_h,
        "wt_pct_n": wt_pct_n,
        # Ratios
        "co_ratio_atomic": n_c / n_o if n_o > 0 else float("inf"),
        "co_ratio_mass": mass_c / mass_o if mass_o > 0 else float("inf"),
        # Group counts
        "n_cooh": n_cooh,
        "n_epoxy": n_epoxy,
        "n_oh": n_oh,
    }

    return result


def print_composition(comp: dict, header: str = "Composition"):
    """Print a formatted composition report with atomic % and weight %."""
    print(f"\n--- {header} ---")
    print(f"  Total atoms: {comp['n_total']}")
    print(f"  Total mass:  {comp['mass_total']:.1f} amu")

    # Table header
    print(f"\n  {'Element':<8} {'Count':>6}   {'at%':>7}   {'wt%':>7}")
    print(f"  {'-'*8} {'-'*6}   {'-'*7}   {'-'*7}")

    print(f"  {'C':<8} {comp['n_c']:>6d}   {comp['at_pct_c']:>6.1f}%   {comp['wt_pct_c']:>6.1f}%")
    print(f"  {'O':<8} {comp['n_o']:>6d}   {comp['at_pct_o']:>6.1f}%   {comp['wt_pct_o']:>6.1f}%")
    print(f"  {'H':<8} {comp['n_h']:>6d}   {comp['at_pct_h']:>6.1f}%   {comp['wt_pct_h']:>6.1f}%")
    if comp["n_n"] > 0:
        print(f"  {'N':<8} {comp['n_n']:>6d}   {comp['at_pct_n']:>6.1f}%   {comp['wt_pct_n']:>6.1f}%")

    # C/O ratios
    if comp["n_o"] > 0:
        print(f"\n  C/O ratio (atomic): {comp['co_ratio_atomic']:.2f}")
        print(f"  C/O ratio (mass):   {comp['co_ratio_mass']:.2f}")
    else:
        print(f"\n  C/O ratio: inf (no oxygen)")

    # Hypothetical additions breakdown
    if comp["n_cooh"] > 0 or comp["n_epoxy"] > 0 or comp["n_oh"] > 0:
        print(f"\n  Hypothetical additions:")
        if comp["n_cooh"] > 0:
            print(f"    COOH:  {comp['n_cooh']} groups (+{comp['n_cooh']}C, "
                  f"+{comp['n_cooh']*2}O, +{comp['n_cooh']}H)")
        if comp["n_epoxy"] > 0:
            print(f"    Epoxy: {comp['n_epoxy']} groups (+{comp['n_epoxy']}O)")
        if comp["n_oh"] > 0:
            print(f"    OH:    {comp['n_oh']} groups (+{comp['n_oh']}O, +{comp['n_oh']}H)")
    print()
