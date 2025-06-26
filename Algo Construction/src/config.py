# src/config.py

# Fraction of median line height used to decide if two words share a line
Y_TOL_FACTOR = 0.7

# Fraction of median word width used to decide if two words glue into one (small gap)
X_GAP_TOL_FACTOR = 0.5

# Fraction of median line height used to decide if two lines stay in same paragraph
LINE_GAP_TOL_FACTOR = 1.5

# Fraction of median char width used to allow small indent shifts within a paragraph
INDENT_TOL_FACTOR = 0.2
