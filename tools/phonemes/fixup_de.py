# Replacements:
# ʏ -> y
def fixup(ps: str) -> str:
    ps = ps.replace("\u028f", "y")
    return ps
