#!/bin/sh
RST=bench_breadth_result.txt

echo "Hidden Size Test" > $RST

echo "======= Only Forward =======" >> $RST
echo "MLP-NumPy" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_numpy_cpu.py --only-forward --hidden-size=$h
done | tee -a $RST
echo "MLP-MinPy-CPU" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_minpy_cpu.py --only-forward --hidden-size=$h
done | tee -a $RST
echo "MLP-MXNet-GPU" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_mxnet_gpu.py --only-forward --hidden-size=$h
done | tee -a $RST
echo "MLP-MinPy-GPU" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_minpy_gpu.py --only-forward --hidden-size=$h
done | tee -a $RST
echo "MLP-MinPy-GPU-Sym" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_minpy_gpu_sym.py --only-forward --hidden-size=$h
done | tee -a $RST


echo "======= With Backward =======" >> $RST
echo "MLP-NumPy" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_numpy_cpu.py --hidden-size=$h
done | tee -a $RST
echo "MLP-MinPy-CPU" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_minpy_cpu.py --hidden-size=$h
done | tee -a $RST
echo "MLP-MXNet-GPU" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_mxnet_gpu.py --hidden-size=$h
done | tee -a $RST
echo "MLP-MinPy-GPU" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_minpy_gpu.py --hidden-size=$h
done | tee -a $RST
echo "MLP-MinPy-GPU-Sym" >> $RST
for h in 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096
do
  python mlp_minpy_gpu_sym.py --hidden-size=$h
done | tee -a $RST
