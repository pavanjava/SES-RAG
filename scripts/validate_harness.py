import json, re, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
errors = []
skills = {}
for f in (root / ".claude/skills").glob("*/SKILL.md"):
    text = f.read_text()
    m = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
    if not m or m[1] != f.parent.name:
        errors.append(f"{f}: bad frontmatter or name != dir")
        continue
    skills[m[1]] = m[2]
    for h in ("## Process", "## Verification"):
        if h not in text:
            errors.append(f"{f}: missing {h}")
    if "Use " not in m[2]:
        errors.append(f"{f}: description lacks a 'Use ...' trigger")

for f in (root / ".claude/commands").glob("*.md"):
    for s in re.findall(r"(?:Invoke|with) ([a-z-]+(?:-[a-z]+)+)", f.read_text()):
        if s in skills or s.endswith(("-auditor", "-reviewer", "-engineer")):
            continue
        errors.append(f"{f}: unknown skill {s}")

for f in (root / ".claude/agents").glob("*.md"):
    if not re.match(r"---\nname: .+\ndescription: .+\n---\n", f.read_text()):
        errors.append(f"{f}: bad frontmatter")

for c in json.loads((root / "evals/cases/routing.json").read_text()):
    if c["skill"] not in skills:
        errors.append(f"eval: unknown skill {c['skill']}")

json.loads((root / ".claude/settings.json").read_text())
print(f"skills={len(skills)} errors={len(errors)}")
for e in errors:
    print(" -", e)
sys.exit(1 if errors else 0)
