@echo off
echo Setting temporary directories to D:\GPT_instinct\tmp
mkdir D:\GPT_instinct\tmp
set TMP=D:\GPT_instinct\tmp
set TEMP=D:\GPT_instinct\tmp
set TMPDIR=D:\GPT_instinct\tmp
echo Installing PyTorch CUDA...
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
echo PyTorch installation complete.
