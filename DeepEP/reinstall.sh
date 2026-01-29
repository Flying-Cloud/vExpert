#!/bin/bash

echo "=== DeepEP Reinstall Script ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""

# Check if deep_ep is already installed
echo "Checking current deep_ep installation..."
if python -c "import deep_ep" 2>/dev/null; then
    echo "Found installed deep_ep at:"
    python -c "import deep_ep; print(deep_ep.__file__)"
else
    echo "No installed deep_ep found"
fi

echo ""
echo "Starting cleanup and reinstall..."

# Clean old build files
echo "1. Cleaning old build files..."
rm -rf build/
rm -rf dist/
rm -rf deep_ep.egg-info/
rm -rf *.egg-info/

# Rebuild and install
echo "2. Rebuilding and installing..."
python setup.py build
python setup.py install --force

echo ""
echo "3. Verifying installation..."
if python -c "import deep_ep; print('✓ deep_ep imported successfully'); print('Installation location:', deep_ep.__file__)" 2>/dev/null; then
    echo "✓ Installation successful!"
else
    echo "✗ Installation failed, trying manual uninstall and reinstall..."
    
    # Try uninstalling
    pip uninstall deep_ep -y 2>/dev/null || true
    
    # Reinstall
    python setup.py install --force
    
    # Verify again
    if python -c "import deep_ep; print('✓ deep_ep imported successfully'); print('Installation location:', deep_ep.__file__)" 2>/dev/null; then
        echo "✓ Reinstall successful!"
    else
        echo "✗ Still failed, please check error messages"
        python -c "import deep_ep" 2>&1 || true
    fi
fi

echo ""
echo "=== Installation Complete ==="




