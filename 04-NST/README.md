# Section 4: Neural network style transfer (NNST)

This section has two parts: 
* **Part-I**: Here we will look at style transfer in the CNN way and use a jupyter notebook with Tensorflow eager execution.
* **Part-II**: In this part we do the NST the GAN way. This kind of project will give good intuition on using GAN, which is a new paradigm in AI.

## **Part-I: NNST with CNN**
---

First navigate to `04-NST/NST-CNN/` and type the following command to open the jupyter notebook

```sh
$ conda activate ai
$ jupyter notebook
```

* You should see the your browser open the address `http://localhost:8888/tree`
* Click on `NST.ipynb`
* After this execute each of the blocks in the notebook using `Shift + Enter`

## **Part-II: NNST with Cycle-GAN**
---

First navigate to `04-NST/NST-CGAN/` and type the following command to open the jupyter notebook

```sh
$ conda activate ai
$ jupyter notebook
```

* Create a folder `data` and navigate to it
* Download the Monet dataset inside `data` folder from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)
* Extract the Monet dataset
* Open the address `http://localhost:8888/tree`, if not already opened
* Click on `cycleGAN.ipynb`
* After this execute each of the blocks in the notebook using `Shift + Enter`
