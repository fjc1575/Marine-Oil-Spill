# EUDA-PLR

### Official pytorch implementation of the paper: "Enhanced Unsupervised Domain Adaptation with Iterative Pseudo-Label Refinement for Inter-Event Oil Spill Segmentation in SARImages".

## Code

### Pre-requsites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

### Model training
To train EUDA-PLR on your own training images, put the training images under ```data/Source``` and ```data/Target```.
To reach to the comparable performance you may need to train a few times.
By default, logs and snapshots are stored in ```<root_dir>/experiments/``` with this structure.

**Step 1.** Conduct inter-domain adaptation by runing: 
```bash
$ cd <root_dir>/ADVENT/advent/scripts
$ python train.py --cfg ./config/advent.yml 
$ python train.py --cfg ./config/advent.yml --tensorboard % using tensorboard
```
After inter-domain training, it is needed to get best IoU iteration by runing:
```bash
$ cd <root_dir>/ADVENT/advent/scripts
$ python test.py --cfg ./config/advent.yml
```
The best IoU iteration ```BEST_ID``` will be a parameter to **step 2 and step 3**. 

**Step 2.** The initial pseudo-label of the target domain is generated using the optimal weights for inter-domain adaptation.
Ranking based on knowledge decision to split training set of target data into easy split and hard split: 
```bash
$ python pseudo.py --best_iter BEST_ID
```
You will see the pseudo-label in ```color_masks```, the easy split file names in ```easy_split.txt```, and the hard split file names in ```hard_split.txt```.

**Step 3.** Conduct intra-domain adaptation by runing:
```bash
$ cd <root_dir>/intrada
$ python train.py --cfg ./intrada.yml
$ python train.py --cfg ./intrada.yml --tensorboard % using tensorboard
```
After intra-domain training, it is needed to get best IoU iteration by runing:
```bash
$ cd <root_dir>/intrada
$ python test.py --cfg ./intrada.yml
```

### The trained model
Download link of the test EUDA-PLR model for Oil spill segmentation:
```bash
链接: https://pan.baidu.com/s/1QYUOY3xB4ZoaW1txMCPqrw 
提取码: 7hew
```