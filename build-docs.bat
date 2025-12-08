@echo off
echo Building Black Box Precision Documentation...
pip install -r requirements-docs.txt
mkdocs build
echo Documentation built successfully!
echo To serve locally, run: mkdocs serve

