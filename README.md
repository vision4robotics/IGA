# [Incremental Grey-box Attack]

## Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## Attack on Trackers

## [SiamRPN++] 
The pre-trained model of SiamRPN++ with AlexNet can be found at /experiments/
and the pre-trained model of our attack method can be found at /checkpoints/Grey_box_attack/model.pth


## Datasets Setting
We evaluate our attack method on four well-known UAV tracking benchmark, i.e., UAV123, UAVTrack112, DTB70 and UAVDT

## Test Attack
```
vim ~/.bashrc
export PYTHONPATH=/home/user/source_code:$PYTHONPATH
export PYTHONPATH=/home/user/source_code/snot:$PYTHONPATH
export PYTHONPATH=/home/user/source_code/attack_utils:$PYTHONPATH
export PYTHONPATH=/home/user/source_code/pix2pix:$PYTHONPATH
source ~/.bashrc
```

```
python test_siamrpn.py
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.



