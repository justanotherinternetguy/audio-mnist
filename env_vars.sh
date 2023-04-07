CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=/home/optimus/anaconda3/pkgs/libtiff-4.5.0-h6a678d5_2/lib/:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=/home/optimus/anaconda3/pkgs/cudatoolkit-11.8.0-h37601d7_11/lib/:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/cuda

#/home/optimus/anaconda3/pkgs/cudatoolkit-11.8.0-h37601d7_11/lib/
