"""Grammar scoring engine for spoken English.

Note (macOS): PyTorch and LightGBM each bundle their own OpenMP runtime and crash when both are active in one
process. Feature extraction (PyTorch) therefore runs as a separate process (`python -m src.extract_all`), and the
modelling code only reads the cached features.
"""
