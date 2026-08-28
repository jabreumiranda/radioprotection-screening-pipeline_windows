from pathlib import Path
import importlib

# automatically import any Python files in the criterions/ directory
# print("automatically import any Python files in the criterions/ directory")
for file in sorted(Path(__file__).parent.glob("*.py")):
    # print(file)
    if not file.name.startswith("_"):
        importlib.import_module("unimol.tasks." + file.name[:-3])
        # print("unimol.tasks." + file.name[:-3])
