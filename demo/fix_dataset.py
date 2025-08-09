#!/usr/bin/env python3
"""
Recompute 'decoded_output' from 'computation_outcomes_count' by majority vote
of the leftmost bit (index 0). Update 'match' to ✓ if decoded_output == correct_value else ✗.

Usage:
    python fix_decoded_output.py input.csv output.csv
"""

import sys
import csv
import ast

CHECK = "✓"
CROSS = "✗"

def majority_leftmost_bit(counts):
    """
    counts: dict mapping bitstrings -> int counts (e.g., {'101': 34, '001': 49, ...})
    Returns: 0, 1, or None (for tie)
    """
    ones = 0
    zeros = 0
    for bitstr, cnt in counts.items():
        if not bitstr:  # skip empty keys defensively
            continue
        if bitstr[0] == "1":
            ones += int(cnt)
        elif bitstr[0] == "0":
            zeros += int(cnt)
        # ignore any unexpected first chars
    if ones > zeros:
        return 1
    if zeros > ones:
        return 0
    return None  # tie

def parse_counts(cell):
    """
    Safely parse the dict-like string from the CSV. The file uses single quotes,
    so we use ast.literal_eval instead of json.
    """
    if cell is None or cell == "":
        return {}
    try:
        obj = ast.literal_eval(cell)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # As a fallback, try stripping whitespace and re-attempt minimal fixes
    cell = cell.strip()
    try:
        obj = ast.literal_eval(cell)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        raise ValueError(f"Could not parse computation_outcomes_count: {cell!r}") from e

def to_int_or_none(x):
    if x is None or x == "" or str(x).lower() == "none":
        return None
    try:
        return int(x)
    except Exception:
        return None

def main(inp, out):
    with open(inp, newline="", encoding="utf-8") as f_in, \
         open(out, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        # Ensure required columns exist
        required = {"computation_outcomes_count", "correct_value", "decoded_output", "match"}
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise SystemExit(f"Missing required columns: {missing}. Found: {fieldnames}")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            counts = parse_counts(row.get("computation_outcomes_count", ""))

            decoded = majority_leftmost_bit(counts)
            # Store as bare 0/1 or 'None' for tie to match your spec
            row["decoded_output"] = "None" if decoded is None else str(decoded)

            correct_value = to_int_or_none(row.get("correct_value"))
            row["match"] = CHECK if (decoded is not None and decoded == correct_value) else CROSS

            writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_decoded_output.py input.csv output.csv", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
