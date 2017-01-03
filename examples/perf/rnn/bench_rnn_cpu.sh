#!/bin/sh
RST=bench_rnn_cpu_result.txt

echo "RNN CPU Test" > $RST

for b in 32 64 128
do
  for h in 512 1024
  do
    echo "B=$b H=$h"
  done
done | tee -a $RST

echo ">>>>>>>> Only Forward >>>>>>>>" >> $RST
echo "MinPy" >> $RST
for b in 32 64 128
do
  for h in 512 1024
  do
    python rnn_minpy_cpu.py --only-forward --batch-size=$b --hidden-size=$h --num-unroll-steps=50 --num-loops=20
  done
done | tee -a $RST

echo "NumPy" >> $RST
for b in 32 64 128
do
  for h in 512 1024
  do
    python rnn_numpy.py --only-forward --batch-size=$b --hidden-size=$h --num-unroll-steps=50 --num-loops=20
  done
done | tee -a $RST

echo ">>>>>>>> With BP >>>>>>>>" >> $RST
echo "MinPy" >> $RST
for b in 32 64 128
do
  for h in 512 1024
  do
    python rnn_minpy_cpu.py --batch-size=$b --hidden-size=$h --num-unroll-steps=50 --num-loops=20
  done
done | tee -a $RST

echo "NumPy" >> $RST
for b in 32 64 128
do
  for h in 512 1024
  do
    python rnn_numpy.py --batch-size=$b --hidden-size=$h --num-unroll-steps=50 --num-loops=20
  done
done | tee -a $RST
