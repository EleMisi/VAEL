"""

Mainly inspired from how ParlAI shows results

"""

import re

import pandas as pd

LINE_WIDTH = 700


def float_formatter(f) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f:
        # instead of returning nan, return "" so it shows blank in table
        return ""
    if isinstance(f, int):
        # don't do any rounding of integers, leave them alone
        return str(f)
    if f >= 1000:
        # numbers > 1000 just round to the nearest integer
        s = f'{f:.0f}'
    else:
        # otherwise show 4 significant figures, regardless of decimal spot
        s = f'{f:.4g}'
    # replace leading 0's with blanks for easier reading
    # example:  -0.32 to -.32
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    # Add the trailing 0's to always show 4 digits
    # example: .32 to .3200
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s


def prettify_statistics(statistics):
    df = pd.DataFrame([statistics])
    result = "   " + df.to_string(
        na_rep="",
        line_width=LINE_WIDTH - 3,  # -3 for the extra spaces we add
        float_format=float_formatter,
        index=df.shape[0] > 1,
    ).replace("\n\n", "\n").replace("\n", "\n   ")
    result = re.sub(r"\s+$", "", result)

    return result
