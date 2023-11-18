#eval "$(conda shell.bash hook)" && \
conda create -y -p /scratch/gpfs/mj2976/.conda/envs/llamao && \
conda activate llamao && \

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia && \
# need to install this transformers version from scratch currently
pip install transformers==4.34.0.dev0 datasets accelerate ipython wandb pysbd sentencepiece packaging ninja peft && \
export CUDA_HOME=/usr/local/cuda-11.7 && \
pip install flash-attn==2.1.0 --no-build-isolation && \
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary && \
echo "Success!"