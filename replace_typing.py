import os
import re

# Define the path to your package directory
package_dir = "./src/insight"  # Replace with the path to your package directory

# Regex patterns to match and replace the imports and typing objects
import_pattern = re.compile(r"^from typing import (.+)$", re.MULTILINE)
import_as_pattern = re.compile(r"^import typing as ty$", re.MULTILINE)
typing_objects_pattern = re.compile(
    r"\b(?:List|Dict|Tuple|Set|FrozenSet|Optional|Union|Callable|Iterator|Sequence|Any|cast)\b"
)

# Mapping of original types to their ty. equivalents
typing_objects_map = {
    "List": "ty.List",
    "Dict": "ty.Dict",
    "Tuple": "ty.Tuple",
    "Set": "ty.Set",
    "FrozenSet": "ty.FrozenSet",
    "Optional": "ty.Optional",
    "Union": "ty.Union",
    "Callable": "ty.Callable",
    "Iterator": "ty.Iterator",
    "Sequence": "ty.Sequence",
    "Any": "ty.Any",
    "cast": "ty.cast",
}

# Function to replace the typing objects with ty.ObjectName
def replace_typing_objects(match):
    return typing_objects_map[match.group()]


# Walk through the directory, find .py files, and perform the replacements
for subdir, dirs, files in os.walk(package_dir):
    for filename in files:
        filepath = os.path.join(subdir, filename)
        if filepath.endswith(".py"):
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            # Skip files that already have 'import typing as ty'
            if import_as_pattern.search(content):
                continue

            # Replace 'from typing import' with 'import typing as ty' if not already done
            if "import typing as ty" not in content:
                content = import_pattern.sub("import typing as ty", content)

            # Replace typing objects with ty.ObjectName
            new_content = typing_objects_pattern.sub(replace_typing_objects, content)

            # Only write if changes have been made
            if new_content != content:
                with open(filepath, "w", encoding="utf-8") as file:
                    file.write(new_content)

print("Typing syntax replaced across all .py files in the package.")
