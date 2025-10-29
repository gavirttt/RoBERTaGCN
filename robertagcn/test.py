import sys
print(f"Python version: {sys.version}")
print(f"Version info: {sys.version_info}")
print(f"Platform: {sys.platform}")
print(f"Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")
