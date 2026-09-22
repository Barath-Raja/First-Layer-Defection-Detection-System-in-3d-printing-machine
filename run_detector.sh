#!/bin/zsh
# Pure Python 3D Print Defect Detector

# Use existing venv_new or create
if [ ! -d "venv_new" ]; then
  python3 -m venv venv_new
fi

source venv_new/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Disable warnings
export PYTHONWARNINGS=ignore::ResourceWarning

echo "🚀 Starting Pure Python 3D Print Defect Detector"
echo "📹 Live OpenCV window - Press 'q' to quit"
echo "🔔 Defects logged to defect_images/ and logs/"

python detect_defect.py
