py -m venv .venv
.venv\Scripts\activate.ps1
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
pip install numpy

#verify
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"

# design

```mermaid
flowchart LR
A["byte"] --> B["byte embedding"]
C["position"] --> D["position embedding"]
B --> E["add"]
D --> E
E --> F["attention"]
F --> G["MLP"]
G --> H["next-byte prediction"]
```

use more transformers

```mermaid
flowchart LR
A["bytes + position"] --> B["Block 1"]
B --> C["Block 2"]
C --> D["Block 3"]
D --> E["Block 4"]
E --> F["output"]
```

# run

```
python main.py
0: loss=5.7992
100: loss=0.9663
200: loss=0.8042
300: loss=0.4229
400: loss=0.2531
500: loss=0.1092
600: loss=0.1075
700: loss=0.1189
800: loss=0.0412
900: loss=0.1080
the cat likes milk.
the dog likes food.

the cat sat on the mat.
the dog sat on the flooor.
the cat likes m
```
