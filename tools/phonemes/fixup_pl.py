import re

# Replacements:
# ʑ + i -> z + i
# ʑ -> ʒ
def fixup(ps: str) -> str:
    ps = re.sub(r'ʑ(?=[ˈˌ]*i)', 'z', ps)
    ps = ps.replace('ʑ', 'ʒ')
    return ps
