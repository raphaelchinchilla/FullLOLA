# Full LOLA

## Theory
In this technical report, we derive an algorithm we call Full LOLA, an extension of the [Learning with Opponent Learning Awerness](https://arxiv.org/abs/1709.04326) (LOLA) algorithm for solving minmax optimization. The algorithm is motivated by properties of what we call a full descent ascent direction, in which the minimizer uses a descent direction that takes into account the response of the maximizer. We show that this algorithm is a generalization of the concept of descent direction from standard minimization. Finally, we state conditions on the step sizes of the minimizer and maximizer such that the limit points of the algorithm satisfy the first order necessary condition for a local minmax.

## codes
In the codes section, we implement the Full LOLA in the context of adversarial training for the MNIST and CIFAR10 datasets. We were able to achieve relatively high levels of protection against PGD attack. While we were not able to outperform the state of the art, maybe it could be achieved with better tuning of the hyperparameters on a faster GPU.

The codes are a collaboration between Metehan Cekic, who wrote most of the general structure of the module, and Raphael Chinchilla, who implemented the algorithm and did most of the testing and debugging.

The adversarial attack is implemented in the file `train_test.py`

Run the codes by typing: `python -m codes.mnist.main -tr -sm -at` and `python -m codes.cifar.main -tr -sm -at`.

Besides pytorch and some other standard packages, the code also requires the deepillusion package, developped by Metehan Cekic and avaiable on pip.
