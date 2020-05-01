# Pymoten website

## Requirements

```
numpydoc
sphinx
sphinx_gallery
sphinx_bootstrap_theme
```

## Build the website

```bash
cd docs
sphinx-apidoc -o source/autodoc ../moten -e
sphinx-autogen source/index.rst
make html
firefox build/html/index.html
```
