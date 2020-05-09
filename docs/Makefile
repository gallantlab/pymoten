# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

API_DIR = source/autodoc
HTMLOUTPATH = build/html
PROJECTNAME = moten

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

githubio-docs:
	@mkdir -p $(HTMLOUTPATH)
	@mkdir -p $(HTMLOUTPATH)/_downloads
	@touch $(HTMLOUTPATH)/.nojekyll
	@echo "Making API..."
	@sphinx-apidoc -o $(API_DIR) ../$(PROJECTNAME) -e
	@sphinx-autogen $(SOURCEDIR)/index.rst
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(HTMLOUTPATH)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(HTMLOUTPATH)."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
