Integrating Text Matching and Product Substitutability
======================================================

Data collection
----

We use the [Amazon product data](http://jmcauley.ucsd.edu/data/amazon) published by McAuley et al. at SIGIR 2015. You can obtain the data by following the [provided instructions](http://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt).

We make use of the `Pet_Supplies`, `Sports_and_Outdoors`, `Toys_and_Games` and `Electronics` reviews and metadata data files (full, not the 5-core). For more information, see the [CIKM 2016 paper](https://arxiv.org/pdf/1608.07253.pdf) where the evaluation sets where first introduced.

|      | Pet Supplies | Sports & Outdoors | Toys & Games | Electronics |
| ---- | ------------------ | ---------------------------- | -------------- | --------------------- |
| Product lists | [product_list](resources/product-substitutability/pet_supplies/product_list) | [product_list](resources/product-substitutability/sports_and_outdoors/product_list) | [product_list](resources/product-substitutability/toys_and_games/product_list) | [product_list](resources/product-substitutability/electronics/product_list) |
| Product substitute relations | [substitutes](resources/product-substitutability/pet_supplies/substitutes) | [substitutes](resources/product-substitutability/sports_and_outdoors/substitutes) | [substitutes](resources/product-substitutability/toys_and_games/substitutes) | [substitutes](resources/product-substitutability/electronics/substitutes) |
| Topics | [topics](resources/product-substitutability/pet_supplies/topics) | [topics](resources/product-substitutability/sports_and_outdoors/topics) | [topics](resources/product-substitutability/toys_and_games/topics) | [topics](resources/product-substitutability/electronics/topics) |
| Relevance | [qrel_test](resources/product-substitutability/pet_supplies/qrel_test) [qrel_validation](resources/product-substitutability/pet_supplies/qrel_validation) | [qrel_test](resources/product-substitutability/sports_and_outdoors/qrel_test) [qrel_validation](resources/product-substitutability/sports_and_outdoors/qrel_validation) | [qrel_test](resources/product-substitutability/toys_and_games/qrel_test) [qrel_validation](resources/product-substitutability/toys_and_games/qrel_validation) | [qrel_test](resources/product-substitutability/electronics/qrel_test) [qrel_validation](resources/product-substitutability/electronics/qrel_validation) |

Usage
-----

To replicate the experiments of the paper on **integrating text matching and product substitutability**, build an Indri index where every product is respresented by the union of its description and its reviews as described in the experimental setup section of the [CIKM 2018 paper](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/van-gysel-mix-n-match-2018.pdf). Secondly, follow the tutorial for the [2018 TOIS paper](https://arxiv.org/abs/1708.02702) on neural vector spaces that can be found [here](TUTORIAL.md).

The substitution relations (or, in fact, any document/document similarity relations) can be passed to [`cuNVSMTrainModel`](https://github.com/cvangysel/cuNVSM/blob/master/cpp/main.cu#L657) as a second positional argument that follows the path to the Indri index. The [`--entity_similarity_weight`](https://github.com/cvangysel/cuNVSM/blob/master/cpp/main.cu#L73) flag controls the mixture weight between text and substitutability signals.

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