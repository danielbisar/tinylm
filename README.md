py -m venv .venv
.venv\Scripts\activate.ps1
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
pip install numpy

#verify
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"

# run

```
python main.py
0: loss=5.8837
100: loss=1.3015
200: loss=0.9806
300: loss=0.8876
400: loss=0.6690
500: loss=0.7349
600: loss=0.6832
700: loss=0.5732
800: loss=0.4545
900: loss=0.5376
the cat don dog likes t t t t t lilililililikes the like likes fod.





the lke dog likes fod.
the fod.
th
```
