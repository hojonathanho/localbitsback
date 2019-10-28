# Compression with Flows via Local Bits-Back Coding
Jonathan Ho, Evan Lohn, Pieter Abbeel

Neural Information Processing Systems, 2019

https://arxiv.org/abs/1905.08500

Contains a PyTorch implementation of [Flow++](https://arxiv.org/abs/1902.00275).

Models available [here](https://drive.google.com/open?id=1g_UMRnAruT5SpsvzhU3bJADsp9OdSAOp).

## Dependencies

- Python 3.6.7
- PyTorch 1.1.0 (CUDA 10.0)

## Installation instructions

1. Install Anaconda with Python 3.6.7
2. `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
3. `pip install tqdm`
4. Extract this codebase into a directory called `compression`
5. Build the underlying C++ library (need a compiler with OpenMP support):

```sh
cd compression/ans
mkdir build
cd build
cmake ..
make
```
