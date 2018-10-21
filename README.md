cuNVSM
======

:warning: You need a CUDA-compatible GPU (compute capability 5.2+) to use this software.

cuNVSM is a C++/CUDA implementation of state-of-the-art [NVSM](https://arxiv.org/abs/1708.02702) and [LSE](https://arxiv.org/pdf/1608.07253.pdf) representation learning algorithms. It also supports injecting a priori knowledge of document/document similarity, as was the main subject of study in the [CIKM2018](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/van-gysel-mix-n-match-2018.pdf) paper on product substitutability.

It integrates conveniently with the [Indri search engine](https://www.lemurproject.org/indri.php) and model parameters are estimated directly from indexes created by Indri. Model parameters are stored in the open [HDF5](https://support.hdfgroup.org/HDF5) format. A lightweight Python module `nvsm`, provided as part of this toolkit, allows querying the models and more.

For more information, see Section 3.3 of the 2018 TOIS paper [<i>"Neural Vector Spaces for Unsupervised Information Retrieval"</i>](https://arxiv.org/abs/1708.02702).

Requirements
------------

To build the cuNVSM training binary and manage dependencies, we use [CMake](https://cmake.org/) (version 3.8 and higher). In addition, we rely on the following libraries for the cuNVSM training binary:

   * [Boost](http://www.boost.org) (>= 1.65.1)
   * [CUDA](https://developer.nvidia.com/cuda) (>= 8.0)
   * [cuDNN](https://developer.nvidia.com/cudnn) (>= 5.1.3)
   * [Glog](https://github.com/google/glog) (>= 0.3.4)
   * [HDF5](https://support.hdfgroup.org/HDF5) (>= 1.6.10)
   * [Indri](https://www.lemurproject.org/indri.php) (>= 5.11)
   * [gperftools](https://github.com/gperftools/gperftools) (>= 2.5)
   * [protobuf](https://github.com/google/protobuf) (>= 3.5.1)

The [cnmem](https://github.com/NVIDIA/cnmem) library is used for memory management. The tests are implemented using the [googletest and googlemock](https://github.com/google/googletest) frameworks. CMake will fetch and compile these libraries automatically as part of the build pipeline. Finally, you need a CUDA-compatible GPU in order to perform any computations.

Dependencies for the `nvsm` Python (>= 3.5) library used for loading and querying trained models can be installed as follows:

	pip install -r requirements.txt

Note that the Python library depends on [pyndri](https://github.com/cvangysel/pyndri), which in turn also depends on [Indri](https://www.lemurproject.org/indri.php).

Installation
------------

To install cuNVSM, the following instructions should get you started. Note that the installation will fail if dependencies cannot be found.

	git clone https://github.com/cvangysel/cuNVSM
	cd cuNVSM
	mkdir build
	cd build
	cmake ..
	make
	make install

Please refer to the [CMake documentation](https://cmake.org/documentation) for advanced options.

cuNVSM also comes with a rich test harness to verify its implementation, see [TESTS](TESTS.md) for more information.

Examples
--------

See [TUTORIAL](TUTORIAL.md) for examples.

Frequently Asked Questions
--------------------------

### How do I run NVSM or LSE?

Different models can be trained/queried by passing the appropriate flags to the `cuNVSMTrainModel` and `cuNVSMQuery` executables.

   * For LSE, pass `--batch_size 4096`, `--nonlinearity tanh` and `--bias_negative_samples` to `cuNVSMTrainModel`.
   * For NVSM, pass `--batch_size 51200`, `--nonlinearity hard_tanh` and `--batch_normalization` to `cuNVSMTrainModel` and pass `--linear` to `cuNVSMQuery`.

For more information, see the [`train_nvsm`](scripts/functions.sh#L369) function in [scripts/functions.sh](scripts/functions.sh) and the invocation of `cuNVSMQuery` in [rank-cranfield-collection.sh](rank-cranfield-collection.sh#L173).

Citation
--------

If you use cuNVSM to produce results for your scientific publication, please refer to our [TOIS](https://arxiv.org/abs/1708.02702) and [CIKM 2018](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/van-gysel-mix-n-match-2018.pdf) papers:

```
@article{VanGysel2018nvsm,
  title={Neural Vector Spaces for Unsupervised Information Retrieval},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  publisher={ACM},
  journal={TOIS},
  year={2018},
}

@inproceedings{VanGysel2018substitutability,
  title={Mix ’n Match: Integrating Text Matching and Product Substitutability within Product Search},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={CIKM},
  volume={2018},
  year={2018},
  organization={ACM}
}
```

The validate/test splits used in the 2018 TOIS paper can be found [here](resources/adhoc-splits). The test collections for the 2018 CIKM paper can be found [here](PRODUCT_SUBSTITUTABILITY.md).

The toolkit also contains an implementation of the LSE model described in the following [CIKM paper](https://arxiv.org/pdf/1608.07253.pdf):

```
@inproceedings{VanGysel2016lse,
  title={Learning Latent Vector Spaces for Product Search},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={CIKM},
  volume={2016},
  pages={165--174},
  year={2016},
  organization={ACM}
}
```

License
-------

cuNVSM is licensed under the [MIT license](LICENSE). CUDA is a licensed trademark of NVIDIA. Please note that [CUDA](https://developer.nvidia.com/cuda-zone) and [Indri](http://www.lemurproject.org/indri.php) are licensed separately. Some of the CMake scripts in the [third_party](third_party) directory are licensed under BSD-3.

If you modify cuNVSM in any way, please link back to this repository.
