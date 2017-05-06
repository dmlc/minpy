'''
get_datasets.sh is duplicated from mxnet/example/image-classification/data/imagenet1k-val.sh.
Usage: bash get_datasets.sh URL_OF_ILSVRC2012_img_val.tar
'''

if [ ! -e ILSVRC2012_img_val.tar ]; then
    wget $1
fi

mkdir -p val
tar -xf ILSVRC2012_img_val.tar -C val
wget http://data.mxnet.io/models/imagenet/resnet/val.lst -O imagenet1k-val.lst

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MX_DIR=${CUR_DIR}/../../../

python im2rec.py --resize 256 --quality 90 --num-thread 16 imagenet1k-val val/

rm -rf val
