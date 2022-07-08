To reproduce the image classication results please follow these steps.

step 1. 

create enviroment with requirements.txt

>>> conda create --name alig_plus python=3.7
>>> conda activate alig_plus
>>> pip install -r requirements.txt

step 2.

install alig module

>>> git clone --recursive https://github.com/oval-group/ali-g
>>> cd ali-g && pip install -r requirements.txt
>>> python setup.py install

for more infomation on this step please visit: https://github.com/oval-group/ali-g

step 3.

>>> cd ../experiments 

all experiement should be run with this as the working directoray.

step 4. 

if required, download relievent dataset and set the paths in the data/loaders.py.
svhn, cifar10 and cifar 100 should download automatically.

step 5.

to reproduce results from paper run one of the following commands:

>>> python reproduce/svhn.py
>>> python reproduce/cifar10.py
>>> python reproduce/cifar100.py
>>> python reproduce/tiny_imagenet.py
>>> python reproduce/imagenet.py

we do not provide code for the other experiements.

CODE USEAGE:
Code based of work by Leonard Berrada, available at https://github.com/oval-group/ali-g
We additionally use the authors implemnetion of Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates
of found a https://github.com/IssamLaradji/sls

