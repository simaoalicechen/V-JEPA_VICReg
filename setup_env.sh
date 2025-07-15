#!/bin/bash
# first time run, need the next two following lines: 
# module load cuda/11.6.2
# module load cudnn/8.6.0.163-cuda11
cd /scratch/netId/video_training 
conda activate OpenSTL_fixed
echo "Ready to go!"

# nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 
# nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.0.2.54 
# nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 
# nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.1.105 nvidia-nvtx-cu12-12.1.105 sympy-1.13.1 
# torch-2.5.1+cu121 torchaudio-2.5.1+cu121 torchvision-0.20.1+cu121 triton-3.1.0

python -c"
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
    print('Device capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(100, 100).cuda()
    print('CUDA test successful!')"