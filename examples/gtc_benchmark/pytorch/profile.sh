# Records are dumped to ./time/naive-lstm-batch-BATCH-hidden-HIDDEN
# or ./time/sophisticated-lstm-batch-BATCH-hidden-HIDDEN (using pickle)
# After unpickling a record, you will get a tuple (tft, ift, bt), where
# tft = forward time (training)
# ift = forward time (inference)
# bt = backward time

if [ ! -d "time" ]; then
  mkdir time
fi

for batch in 16 32 64 128 256
do
for hidden in 512 1024 2048 4096
  do
    ipython naive_lstm.py -- --batch_size=$batch --num_hidden=$hidden
    ipython sophisticated_lstm.py -- --batch_size=$batch --num_hidden=$hidden
  done
done
