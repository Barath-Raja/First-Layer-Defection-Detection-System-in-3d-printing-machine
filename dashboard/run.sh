#!/bin/zsh
cd $(dirname $0)

# Unset zsh traps to prevent trace trap
unsetopt traps

# Use Python 3.11 specifically for stability
if ! command -v python3.11 &> /dev/null; then
    echo \"Installing Python 3.11...\"
    brew install python@3.11 || echo \"Please install python3.11 via brew\"
fi

# Create/update venv
python3.11 -m venv venv
source venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Disable resource warnings
export PYTHONWARNINGS=ignore::ResourceWarning

echo \"🚀 Starting stable 3D Print Monitor...\"
echo \"Open: http://localhost:5000\"

python app.py
