"""
Command-line interface for gengo.

Usage:
    gengo PG <x> <y> <fname> [--format fmt] [-p] [-n 1] [-d 3.35]
    gengo GO <input> <output> --COOH <n> --COC <n> --OH <n> [--format fmt] [-p]
    gengo hole <input> <output> -N <n> --range <min> <max> --mode u|m
              --edge i|e [--cleanup] [--format fmt]
    gengo calc <input> [--COOH <n>] [--COC <n>] [--OH <n>]
"""

import argparse
import sys

from .core import GRAPHITE_INTERLAYER


def _add_format_arg(parser: argparse.ArgumentParser):
    """Add --format argument to a parser."""
    parser.add_argument(
        "--format", "-f", type=str, default=None,
        help="ASE file format string (e.g., pdb, xyz, extxyz, lammps-data, cif). "
             "If not specified, inferred from file extension. "
             "Note: for .lmp files, use --format lammps-data."
    )


def _add_periodic_arg(parser: argparse.ArgumentParser):
    """Add -p/--periodic argument to a parser."""
    parser.add_argument(
        "-p", "--periodic", action="store_true", default=False,
        help="Enable periodic boundary conditions in x and y."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="gengo",
        description="Generate graphene and graphene oxide structures. "
                    "Based on GOPY (Muraru et al., SoftwareX 2020), "
                    "refactored with ASE, periodic BC, and multi-layer support.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- PG subcommand ---
    pg_parser = subparsers.add_parser(
        "PG", help="Generate pristine graphene sheet."
    )
    pg_parser.add_argument("x", type=float, help="X dimension in Angstrom")
    pg_parser.add_argument("y", type=float, help="Y dimension in Angstrom")
    pg_parser.add_argument("fname", type=str, help="Output file path")
    _add_format_arg(pg_parser)
    _add_periodic_arg(pg_parser)
    pg_parser.add_argument(
        "-n", "--nlayer", type=int, default=1,
        help="Number of graphene layers (default: 1)"
    )
    pg_parser.add_argument(
        "-d", "--distance", type=float, default=GRAPHITE_INTERLAYER,
        help=f"Interlayer C-C distance in Angstrom (default: {GRAPHITE_INTERLAYER}, "
             f"minimum: {GRAPHITE_INTERLAYER})"
    )

    # --- GO subcommand ---
    go_parser = subparsers.add_parser(
        "GO", help="Generate graphene oxide from pristine graphene."
    )
    go_parser.add_argument("input", type=str, help="Input pristine graphene file")
    go_parser.add_argument("output", type=str, help="Output GO file")
    go_parser.add_argument(
        "--COOH", type=int, default=0,
        help="Number of carboxyl groups (edge only)"
    )
    go_parser.add_argument(
        "--COC", dest="COC", type=int, default=0,
        help="Number of epoxy groups (basal plane)"
    )
    go_parser.add_argument(
        "--OH", type=int, default=0,
        help="Number of hydroxyl groups (basal plane)"
    )
    _add_format_arg(go_parser)
    _add_periodic_arg(go_parser)

    # --- hole subcommand ---
    hole_parser = subparsers.add_parser(
        "hole", help="Generate holes in a graphene sheet."
    )
    hole_parser.add_argument("input", type=str, help="Input graphene file")
    hole_parser.add_argument("output", type=str, help="Output file")
    hole_parser.add_argument(
        "-N", type=int, required=True,
        help="Number of holes to create"
    )
    hole_parser.add_argument(
        "--range", type=int, nargs=2, required=True, metavar=("MIN", "MAX"),
        help="Size range (min max) in number of atoms per hole"
    )
    hole_parser.add_argument(
        "--mode", type=str, choices=["u", "m"], default="u",
        help="'u' for unidirectional (linear) or 'm' for multidirectional (oval)"
    )
    hole_parser.add_argument(
        "--edge", type=str, choices=["i", "e"], default="i",
        help="'i' for interior (no edge touching) or 'e' for exterior (can touch edges)"
    )
    hole_parser.add_argument(
        "--cleanup", action="store_true", default=False,
        help="Remove isolated fragments (<= 6 atoms) after hole creation"
    )
    _add_format_arg(hole_parser)
    _add_periodic_arg(hole_parser)

    # --- calc subcommand ---
    calc_parser = subparsers.add_parser(
        "calc", help="Calculate atomic composition of a structure."
    )
    calc_parser.add_argument("input", type=str, help="Input structure file")
    calc_parser.add_argument(
        "--COOH", type=int, default=0,
        help="Number of COOH groups to hypothetically add"
    )
    calc_parser.add_argument(
        "--COC", dest="COC", type=int, default=0,
        help="Number of epoxy groups to hypothetically add"
    )
    calc_parser.add_argument(
        "--OH", type=int, default=0,
        help="Number of OH groups to hypothetically add"
    )
    _add_format_arg(calc_parser)

    return parser


def cmd_pg(args):
    """Handle PG subcommand."""
    from .graphene import generate_pristine_graphene
    from .core import GRAPHITE_INTERLAYER as MIN_DIST
    from .io import write_structure

    # Validate inputs early (before any output)
    if args.x <= 0 or args.y <= 0:
        raise ValueError(f"Dimensions must be positive, got x={args.x}, y={args.y}")
    if args.nlayer < 1:
        raise ValueError(f"Number of layers must be >= 1, got {args.nlayer}")
    if args.distance < MIN_DIST:
        raise ValueError(
            f"Interlayer distance {args.distance:.3f} A is less than "
            f"graphite minimum {MIN_DIST:.3f} A"
        )

    print(f"Generating pristine graphene: {args.x} x {args.y} A")
    atoms = generate_pristine_graphene(
        x_dim=args.x,
        y_dim=args.y,
        periodic=args.periodic,
        nlayer=args.nlayer,
        distance=args.distance,
    )

    write_structure(args.fname, atoms, format=args.format)
    print(f"Saved to: {args.fname}")


def cmd_go(args):
    """Handle GO subcommand."""
    from .io import read_structure, write_structure
    from .go import create_go

    if args.COOH == 0 and args.COC == 0 and args.OH == 0:
        print("Error: At least one of --COOH, --COC, --OH must be > 0", file=sys.stderr)
        sys.exit(1)

    print(f"Reading input: {args.input}")
    atoms = read_structure(args.input)
    print(f"  Input atoms: {len(atoms)}")

    print(f"Creating GO: COOH={args.COOH}, epoxy(COC)={args.COC}, OH={args.OH}")
    atoms = create_go(
        atoms,
        n_cooh=args.COOH,
        n_epoxy=args.COC,  # epoxy / C-O-C bridge
        n_oh=args.OH,
        periodic=args.periodic,
    )

    write_structure(args.output, atoms, format=args.format)
    print(f"Saved to: {args.output}")


def cmd_hole(args):
    """Handle hole subcommand."""
    from .io import read_structure, write_structure
    from .holes import generate_holes

    print(f"Reading input: {args.input}")
    atoms = read_structure(args.input)
    print(f"  Input atoms: {len(atoms)}")

    print(f"Generating {args.N} holes, size range {args.range[0]}-{args.range[1]}")
    atoms = generate_holes(
        atoms,
        n_holes=args.N,
        size_range=tuple(args.range),
        mode=args.mode,
        edge_mode=args.edge,
        cleanup=args.cleanup,
        periodic=args.periodic,
    )

    write_structure(args.output, atoms, format=args.format)
    print(f"Saved to: {args.output}")


def cmd_calc(args):
    """Handle calc subcommand."""
    from .io import read_structure
    from .calc import calculate_composition, print_composition

    print(f"Reading: {args.input}")
    atoms = read_structure(args.input)

    has_hypothetical = args.COOH > 0 or args.COC > 0 or args.OH > 0

    # Always show current composition
    comp = calculate_composition(atoms)
    print_composition(comp, header="Current Composition")

    # Show predicted composition if hypothetical groups are specified
    if has_hypothetical:
        comp_pred = calculate_composition(atoms, args.COOH, args.COC, args.OH)
        print_composition(comp_pred, header="Predicted Composition (with additions)")


def main():
    """Entry point for the gengo CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "PG":
            cmd_pg(args)
        elif args.command == "GO":
            cmd_go(args)
        elif args.command == "hole":
            cmd_hole(args)
        elif args.command == "calc":
            cmd_calc(args)
        else:
            parser.print_help()
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
