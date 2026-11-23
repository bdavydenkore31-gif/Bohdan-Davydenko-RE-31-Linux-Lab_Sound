"""Шляхи проєкту, незалежні від поточної робочої папки."""
from pathlib import Path

# Усі ресурси шукаємо від кореня проєкту, незалежно від папки запуску.
ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/'ai/models'
STATIC_DIR=ROOT/'web/static'
EXAMPLES_DIR=ROOT/'examples'
