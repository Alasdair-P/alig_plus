# ALI-G+
This is the repo containing alig_plus code, an effective single hyperparameter optimisation algorithm for non interpolating deep learning problems. Please see [Faking Interpolation Until You Make It](https://openreview.net/pdf?id=OslAMMF4ZP) for full details. If this code is useful please cite as:

```
@Article{paren2022stochastic,
  author       = {Paren, Alasdair and Poudel, Rudra PK and Kumar, M Pawan},
  title        = {A Stochastic Bundle Method for Interpolating Networks},
  journal      = {Transactions on Machine Learning Research},
  year         = {2022},
}
```

# Abstract

Deep over-parameterized neural networks exhibit the interpolation property on many data sets. That
is, these models are able to achieve approximately zero loss on all training samples simultaneously.
Recently, this property has been exploited to develop novel optimisation algorithms for this setting.
These algorithms use the fact that the optimal loss value is known to employ a variation of a Polyak
Step-size calculated on a stochastic batch of data. In this work, we introduce an algorithm that
extends this idea to tasks where the interpolation property does not hold. As we no longer have
access to the optimal loss values a priori, we instead estimate them for each sample online. To
realise this, we introduce a simple but highly effective heuristic for approximating the optimal value
based on previous loss evaluations. Through rigorous experimentation we show the effectiveness of
our approach, which outperforms state of the art adaptive gradient and line search methods.




# Usage

the ALI-G+ algorthum is written in PyTorch >= 1.0 in python3. ALI-G+ requires indexed dataset be used, this can be easily achieved using the following dataset wrapper:

```python
class IndexedDataset(data.Dataset):
    def __init__(self, dataset):
        self._dataset = dataset
    def __getitem__(self, idx):
        return idx, self._dataset[idx]
    def __len__(self):
        return self._dataset.__len__()
```

with this dataset ALI-G+ can be used like any other optimiser with the exception that requires being pass a vector for the indexed of thee xamples in each batch and their corresponding loss values:

```python
optimizer = AligPlus(model.parameters(), args.lr, args.train_size, args.epochs momentum=0.9)

for idx, data in data_loader:
  x, y = data

  optimizer.zero_grad()
  losses = loss_fn(reduction='none')(model(x), y)
  loss_value = losses.mean()
  loss_value.backward()
  optimizer.step(lambda: (idx,losses))

% additionally alig+ require the following method be called after each epoch 
optimizer.epoch_()
```

# Code Requirements and Installation

This code should work for PyTorch >= 1.0 in python3. Please install the necessary packages using:

```
install -r requirements.txt

git clone --recursive https://github.com/oval-group/ali-g
cd ali-g && pip install -r requirements.txt
python setup.py install

if required, download relievent dataset and set the paths in the data/loaders.py.
svhn, cifar10 and cifar 100 should download automatically.
```

# Reproducing the Results

Please first complete the code installation as described above. The following command lines assume that the current working directory is "/experiments" . 

```python
cd ../experiments 
```
To reproduce image classifacation results from paper run one of the following commands:

```
python reproduce/svhn.py
python reproduce/cifar10.py
python reproduce/cifar100.py
python reproduce/tiny_imagenet.py
python reproduce/imagenet.py
```


# Acknowledgments

Code based of work by Leonard Berrada, available at https://github.com/oval-group/ali-g
We additionally use the authors implemnetion of Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates
of found a https://github.com/IssamLaradji/sls
