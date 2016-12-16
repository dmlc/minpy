#!/bin/sh
RST=bench_mlp_result.txt

echo "MLP GPU Test" > $RST

for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    echo "B=$b H=$h"
  done
done | tee -a $RST

echo ">>>>>>>> Only Forward >>>>>>>>" >> $RST
echo "MXNet" >> $RST
for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    python mlp_mxnet_gpu.py --only-forward --batch-size=$b --hidden-size=$h --num-hidden=50 --num-loops=50
  done
done | tee -a $RST

echo "MinPy" >> $RST
for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    python mlp_minpy_gpu.py --only-forward --batch-size=$b --hidden-size=$h --num-hidden=50 --num-loops=50
  done
done | tee -a $RST

echo "MinPy-Sym" >> $RST
for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    python mlp_minpy_gpu_sym.py --only-forward --batch-size=$b --hidden-size=$h --num-hidden=50 --num-loops=50
  done
done | tee -a $RST

echo ">>>>>>>> With BP >>>>>>>>" >> $RST
echo "MXNet" >> $RST
for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    python mlp_mxnet_gpu.py --batch-size=$b --hidden-size=$h --num-hidden=50 --num-loops=50
  done
done | tee -a $RST

echo "MinPy" >> $RST
for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    python mlp_minpy_gpu.py --batch-size=$b --hidden-size=$h --num-hidden=50 --num-loops=50
  done
done | tee -a $RST

echo "MinPy-Sym" >> $RST
for b in 512 1024 2048 4096
do
  for h in 512 1024 2048 4096
  do
    python mlp_minpy_gpu_sym.py --batch-size=$b --hidden-size=$h --num-hidden=50 --num-loops=50
  done
done | tee -a $RST
