"""Classification reference loader — pluggable taxonomy support.

Loads classification hierarchy, syntax rules, and auxiliary tables from a YAML
reference file.  This makes the classifier universal — swap UDC for DDC, LCC,
or any custom taxonomy by providing a different reference file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ClassificationReference:
    """Pluggable classification taxonomy loaded from a YAML reference file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path) as f:
            raw = yaml.safe_load(f)

        self.name: str = raw.get("name", "Unknown")
        self.version: str = raw.get("version", "0.0")
        self.description: str = raw.get("description", "")

        syntax = raw.get("syntax", {})
        self.compound_separator: str = syntax.get("compound_separator", ":")
        self.compound_label: str = syntax.get("compound_label", "in connection with")
        self.std_sub_prefix: str = syntax.get("standard_subdivision_prefix", "-")
        self.geo_facet_prefix: str = syntax.get("geographic_facet_prefix", "(")

        self.main_classes: dict[str, str] = raw.get("main_classes", {})
        self.classes: dict[str, str] = raw.get("classes", {})
        self.standard_subdivisions: dict[str, str] = raw.get("standard_subdivisions", {})
        self.form_numbers: dict[str, str] = raw.get("form_numbers", {})
        self.geographic_table: dict[str, str] = raw.get("geographic_table", {})
        self.temporal_table: dict[str, str] = raw.get("temporal_table", {})

        self.syntax_rules: list[str] = [
            f"Compound separator: '{self.compound_separator}' means '{self.compound_label}'",
            f"Standard subdivisions: prefix '{self.std_sub_prefix}' (e.g. -002, -055)",
            f"Geographic facets: parentheses '{self.geo_facet_prefix}' (e.g. (4), (73))",
            "Temporal facets: parentheses with time codes (e.g. (08), (/08))",
            "Form numbers: hyphen + digits (e.g. (104.2) for language)",
        ]

        logger.info(
            "Loaded %s v%s (%d classes, %d main classes)",
            self.name, self.version, len(self.classes), len(self.main_classes),
        )

    def get_label(self, number: str) -> str:
        components = self.parse_compound(number)
        component_labels = []
        for comp in components:
            base = self.strip_facets(comp)
            label = self._lookup(base)
            if label and label != base:
                label = self._attach_facet_labels(comp, label)
            component_labels.append(label or base)
        return f" {self.compound_label} ".join(component_labels)

    def _lookup(self, number: str) -> str | None:
        if number in self.classes:
            return self.classes[number]
        best = None
        best_len = 0
        for key, label in self.classes.items():
            if number.startswith(key) and len(key) > best_len:
                best = label
                best_len = len(key)
        return best

    def _attach_facet_labels(self, number: str, base_label: str) -> str:
        import re
        result = base_label
        for match in re.finditer(r"\(([^)]+)\)", number):
            facet = f"({match.group(1)})"
            if facet in self.geographic_table:
                result += f" ({self.geographic_table[facet]})"
        return result

    def get_main_class(self, number: str) -> str:
        return number[0] if number else "0"

    def get_children(self, parent: str) -> list[tuple[str, str]]:
        children = []
        for num, label in self.classes.items():
            if num == parent:
                continue
            if num.startswith(parent):
                remainder = num[len(parent):]
                if not remainder or remainder[0] == "." or (len(remainder) == 1 and remainder.isdigit()):
                    children.append((num, label))
        return sorted(children)

    def get_ancestors(self, number: str) -> list[tuple[str, str]]:
        ancestors = []
        parts = number.split(".")
        for i in range(len(parts) - 1, 0, -1):
            ancestor = ".".join(parts[:i])
            label = self._lookup(ancestor)
            if label:
                ancestors.append((ancestor, label))
        main = self.get_main_class(number)
        if main in self.main_classes:
            ancestors.append((main, self.main_classes[main]))
        ancestors.reverse()
        return ancestors

    def parse_compound(self, number: str) -> list[str]:
        if self.compound_separator in number:
            return number.split(self.compound_separator)
        return [number]

    def first_component(self, number: str) -> str:
        return self.parse_compound(number)[0]

    def strip_facets(self, number: str) -> str:
        import re
        base = re.sub(r"-[\d.]+", "", number)
        base = re.sub(r"\([^)]*\)", "", base)
        return base

    def format_compound(self, numbers: list[str], labels: list[str] | None = None) -> str:
        if labels:
            parts = [f"{num} ({label})" for num, label in zip(numbers, labels)]
            return f" {self.compound_label} ".join(parts)
        return self.compound_separator.join(numbers)

    def validate(self, number: str) -> bool:
        import re
        if not number:
            return False
        return bool(re.match(r"^[\d\.\-\:\(\)]+$", number))

    def get_tree(self) -> dict:
        tree: dict = {}
        for main_cls, label in sorted(self.main_classes.items()):
            tree[main_cls] = {"label": label, "children": {}}
        for num, label in sorted(self.classes.items()):
            main = self.get_main_class(num)
            if main not in tree:
                tree[main] = {"label": self.main_classes.get(main, ""), "children": {}}
            current = tree[main]["children"]
            parts = num.split(".")
            for i, part in enumerate(parts):
                partial = ".".join(parts[:i + 1])
                if partial not in current:
                    current[partial] = {"label": self._lookup(partial) or "", "children": {}}
                if i < len(parts) - 1:
                    current = current[partial]["children"]
                else:
                    current[partial]["label"] = label
        return tree

    def udc_to_path(self, number: str) -> str:
        base = self.first_component(number)
        base_stripped = self.strip_facets(base)
        parts_stripped = base_stripped.split(".")
        main = parts_stripped[0][0] if parts_stripped else "0"
        mid = parts_stripped[0] if len(parts_stripped) > 1 else main
        # Collapse redundant nesting when classification is just the main class digit
        if base_stripped == main:
            return f"{main}/{base_stripped}"
        return f"{main}/{mid}/{base}"

    def system_prompt(self) -> str:
        return self.to_prompt_reference()

    def to_prompt_reference(self) -> str:
        lines = []
        lines.append(f"## {self.name} Classification Reference")
        lines.append(f"{self.description}")
        lines.append("")
        for main_cls in sorted(self.main_classes.keys()):
            label = self.main_classes[main_cls]
            lines.append(f"### {main_cls} — {label}")
            for num, num_label in sorted(self.classes.items()):
                if self.get_main_class(num) == main_cls:
                    lines.append(f"- {num}: {num_label}")
            lines.append("")
        if self.standard_subdivisions:
            lines.append("### Standard Subdivisions (attach with hyphen)")
            for key, val in sorted(self.standard_subdivisions.items()):
                lines.append(f"- {key}: {val}")
            lines.append("")
        if self.form_numbers:
            lines.append("### Form Numbers (attach with hyphen)")
            for key, val in sorted(self.form_numbers.items()):
                lines.append(f"- {key}: {val}")
            lines.append("")
        if self.geographic_table:
            lines.append("### Geographic Facets (attach with parentheses)")
            for key, val in sorted(self.geographic_table.items()):
                lines.append(f"- {key}: {val}")
            lines.append("")
        if self.temporal_table:
            lines.append("### Temporal Facets (attach with parentheses)")
            for key, val in sorted(self.temporal_table.items()):
                lines.append(f"- {key}: {val}")
            lines.append("")
        return "\n".join(lines)
