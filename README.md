# WGAN-pytorch
WGAN paper implementation code using pytorch

## Requirement
python >= 3.6  
pytorch >= 1.2.0  
matplotlib >= 3.0.0

## Training model 
```
python3 main.py --model=WGAN --dataset=CelebA --clipping=0.01 --num_critic=5
```

## Show results
```
python3 visualize.py
```
