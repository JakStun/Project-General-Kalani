# Lucrehulk

- brain of all operations that require thinking/analysis

Run this command if you want all devices on local network to send requests:
```uvicorn main:app --host 0.0.0.0 --port 8000```

## Installation:

Tricky install-> torch (check newer versions), for this project I used: torch==2.6.0+cu124
Install it like this:  ```python.exe -m pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124```

For faster-wisper to work you need to have Visual Studio Build Tools for C++ installed (https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Only then install: ```pip install "faster-whisper==1.2.1"```

After these are installed, install: ```pip install -r requirements.txt```

CUDA Tests (if available):
```python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"```