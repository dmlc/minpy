#!/bin/sh
RST=bench_depth_result.txt
HS=1024

echo "Multiple Layers Test" > $RST

echo "======= Only Forward =======" >> $RST
echo "MLP-NumPy" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_numpy_cpu.py --only-forward --hidden-size=$HS --num-hidden=$i --num-loops=30
done | tee -a $RST
echo "MLP-MinPy-CPU" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_minpy_cpu.py --only-forward --hidden-size=$HS --num-hidden=$i --num-loops=30
done | tee -a $RST
echo "MLP-MXNet-GPU" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_mxnet_gpu.py --only-forward --hidden-size=$HS --num-hidden=$i --num-loops=100
done | tee -a $RST
echo "MLP-MinPy-GPU" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_minpy_gpu.py --only-forward --hidden-size=$HS --num-hidden=$i --num-loops=100
done | tee -a $RST
echo "MLP-MinPy-GPU-Sym" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_minpy_gpu_sym.py --only-forward --hidden-size=$HS --num-hidden=$i --num-loops=100
done | tee -a $RST

echo "======= With Backward =======" >> $RST
echo "MLP-NumPy" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_numpy_cpu.py --hidden-size=$HS --num-hidden=$i --num-loops=30
done | tee -a $RST
echo "MLP-MinPy-CPU" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_minpy_cpu.py --hidden-size=$HS --num-hidden=$i --num-loops=30
done | tee -a $RST
echo "MLP-MXNet-GPU" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_mxnet_gpu.py --hidden-size=$HS --num-hidden=$i --num-loops=100
done | tee -a $RST
echo "MLP-MinPy-GPU" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_minpy_gpu.py --hidden-size=$HS --num-hidden=$i --num-loops=100
done | tee -a $RST
echo "MLP-MinPy-GPU-Sym" >> $RST
for i in 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
  python mlp_minpy_gpu_sym.py --hidden-size=$HS --num-hidden=$i --num-loops=100
done | tee -a $RST
