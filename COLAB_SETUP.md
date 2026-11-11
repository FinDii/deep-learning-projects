# Colab setup notes

1. In Colab, choose **Runtime → Change runtime type → GPU** if your experiment requires accelerated training. Availability varies.
2. Open a project notebook, and install its dependencies exactly as required by that notebook; some projects use different framework versions.
3. When importing modules from `src/`, add the project `src` directory to Python's module search path or change the Colab working directory.
4. Mount Google Drive only when needed and adjust any hardcoded `/content/drive` paths; never commit access tokens or private files.
5. Use the original notebook as a reference when a `%%writefile` cell creates an alternate module variant. Avoid executing all versions into the same output filename without checking overwrite order.
6. Save small representative images to `assets/`, not entire datasets or huge training checkpoints.

The included standalone `.py` files have had notebook-only leading `%%writefile` magic removed where present, so they can be imported or executed as normal Python modules. This cleanup was syntax-checked but not run against external datasets.
