"""UniversalDecimalInator: Universal Decimal Classification-based document catalog system.

Pluggable classification reference — swap UDC for DDC, LCC, or any custom
taxonomy by providing a different reference YAML file.
"""

__version__ = "0.1.0"

from UniversalDecimalInator.reference import ClassificationReference
from UniversalDecimalInator.card_writer import CardWriter
from UniversalDecimalInator.models import Card, UDCClassification
