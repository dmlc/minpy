# Docker images for Minpy

Pre-built docker images are available at https://hub.docker.com/r/dmlc/minpy/

## How to use

1. First pull the pre-built image

   ```bash
   docker pull dmlc/minpy
   ```
2. Then we can run the python shell in the docker

   ```bash
   nvidia-docker run -ti dmlc/minpy python
   ```
   For example
   ```bash
   $ nvidia-docker run -ti dmlc/minpy python
   Python 2.7.6 (default, Jun 22 2015, 17:58:13)
   [GCC 4.8.2] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import minpy as np
   >>> ...
   ```

   Note: One may get the error message `libdc1394 error: Failed to initialize
   libdc1394`, which is due to opencv and can be ignored.

3. Train a model on MNIST to check everything works

   To launch the docker, we need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.
   ```
   nvidia-docker run dmlc/minpy python dmlc/minpy/examples/basics/logistic.py --gpus 0
   ```

## How to build

```bash
docker build -t dmlc/minpy .
```
