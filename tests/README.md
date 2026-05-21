# PPISP Tests

This directory contains tests for the PPISP CUDA implementation.

## Test Files

- **test_cuda_vs_torch.py**: Core correctness tests
  - Compares CUDA implementation against pure PyTorch reference
  - Forward pass equivalence (multiple configurations)
  - Backward pass gradient equivalence (all parameter groups)
  - Tests for disabled effects (camera_idx=-1, frame_idx=-1)
  - Stress tests (large batch, many cameras/frames)

- **test_gradcheck.py**: Numerical gradient verification
  - Uses `torch.autograd.gradcheck` for independent gradient validation
  - Gradient magnitude sanity checks

- **test_regularization_loss.py**: Regularization loss tests
  - Compares CUDA regularization loss and gradients against the PyTorch formula

- **torch_reference.py**: PyTorch reference implementation (not tests)
  - Pure PyTorch implementation of PPISP pipeline
  - Used as ground truth for correctness testing

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_cuda_vs_torch.py -v
```

### Run specific test:
```bash
pytest tests/test_cuda_vs_torch.py::test_forward_basic -v
```

## Requirements

- PyTorch with CUDA support
- pytest
- CUDA-capable GPU

## Test Coverage

- Forward pass correctness (CUDA vs PyTorch reference)
- Backward pass correctness (all gradient outputs)
- Numerical gradient verification via finite differences
- Different batch sizes (1 to 4096 pixels)
- Multiple cameras and frames configurations
- Disabled effects (camera_idx=-1, frame_idx=-1)
- CUDA regularization loss and gradient correctness
- Gradient magnitude sanity checks
