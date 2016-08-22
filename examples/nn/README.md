MinPy Tutorial Code Folder
---------------------

* Make sure you have installed MXNet and MinPy. If not, see [here](https://minpy.readthedocs.io/en/latest/get-started/install.html).
* Download the `cifar10` dataset:
  - Enter into `/path/to/minpy/examples/dataset/cifar10/` folder.
  - Call `./get_dtasets.sh`.
  - You should be able to see a `cifar-10-batches-py` folder.
* Run example like this:

  ```bash
  python mlp.py --data_dir=/path/to/minpy/examples/dataset/cifar10/cifar-10-batches-py/
  ```
  
* You could try different examples by replacing the python script you call.
