py -m venv .venv
.venv\Scripts\activate.ps1
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
pip install numpy
pip install datasets

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

## simple recursion
```
This is a useful result.
Model	Params	VRAM	Time	Val loss
4 unique blocks	867,328	32.9 MB	140.9 s	1.2925
1 shared block ×4	272,512	23.0 MB	141.0 s	1.4159


So recurrence gave us:
- ~69% fewer parameters
- ~30% less VRAM
- essentially identical training speed
- only ~0.12 worse validation loss
That is actually promising.
```

## 2 blocks each used 2 times

```
So compared with 4 unique blocks, A→B→A→B uses about 46% fewer parameters and ~21% less VRAM, while validation loss only worsens from 1.2925 to 1.3403.

The repetitive generation is still a weakness, but that affects all these tiny models.
```

## 2 blockes each used 3 times

| Model | Params | VRAM | Time | Val loss |
|---|---:|---:|---:|---:|
| 4 unique blocks | 867k | 32.9 MB | 141 s | **1.2925** |
| A→B ×2 | 471k | 26.0 MB | 139 s | 1.3403 |
| **A→B ×3** | **471k** | **27.2 MB** | **195 s** | **1.3039** |
| A ×4 | 273k | 23.0 MB | 141 s | 1.4159 |

I wouldn't worry much about the repetitive generated text yet. We're using greedy argmax, only 64 bytes of context, and a ~0.5M parameter model.

## 2 blocks each 4 times

| Model | Params | Time | VRAM | Val loss |
|---|---:|---:|---:|---:|
| A→B ×3 | 470,784 | 194.8s | 27.2 MB | **1.3039** |
| A→B ×4 | 470,784 | 248.0s | 28.4 MB | 1.3072 |

# context=128 instead of 64

| Model | Context | Val loss | Time | VRAM |
|---|---:|---:|---:|---:|
| A→B ×3 | 64 | 1.3039 | 194.8s | 27.2 MB |
| A→B ×3 | 128 | **1.2649** | 189.7s | 31.8 MB |

# context=256 instead of 64

The next thing I’d test is context 256 before touching experts or anything more complex. This experiment strongly suggests the model is still context-limited.

| Context | Val loss | VRAM | Time |
|---:|---:|---:|---:|
| 64 | 1.3039 | 27.2 MB | 194.8s |
| 128 | **1.2649** | 31.8 MB | 189.7s |
| 256 | **1.2619** | 43.5 MB | 195.5s |

So 128 → 256 gives only 0.003 better loss but costs ~37% more VRAM.
I’d keep context 128 as the better efficiency point.

## output

```
0: train loss=5.6598
2000: train loss=2.3193
4000: train loss=2.3595
6000: train loss=1.8979
8000: train loss=1.7419
10000: train loss=1.7720
12000: train loss=1.8245
14000: train loss=1.2902
16000: train loss=1.5261
18000: train loss=1.5704

Parameters:      495,360
Training time:   195.54s
Peak VRAM:       43.5 MB
Steps/sec:       102.3
Validation loss: 1.2619

Once upon a time the beautiful the ball and said, "I wanted to the be some the started to play with the started to the better. The bird was so happy and said, "That you we can to the park. I was so so happy and the s
```
