

Official implementation of the paper [Sequential Modelling for Medical Image Segmentation] (https://doi.org/10.1101/2023.11.01.565075)

This work adapts MEGABYTE and Pix2Seq to perform medical image segmentation.

This work is heavily adapted on Chen Ting's Pix2Seq in tensorflow. Ours is written in torch.


For train, batch size is set in data/{task dataset}.py , for parallel training, wandb is automatically initiated.
However, no image will be uploaded for parallel training.   
```
. mega_train.sh
```

For test,

If instance segmentation is involved in the config / task , set all batch size to 1 for evaluation purpose.
For example, to run a test of trained model on instance and semantic segmentation task, give the prepared config yaml
```
python mega_multitask_main.py --mode test --device 0 --cfg configs/instance_semantic.yaml
```



