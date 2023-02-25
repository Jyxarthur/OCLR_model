# Test-time adaptation by RGB-based mask propagation
Flow-predicted masks from OCLR models can be further refined by introducing RGB information. Here, we follow the similar idea proposed in [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), and utilise DINO features to propagate **good** flow-predicted masks with additional processes such as dynamic refinements and finetuning DINO features.

The code is copied and modified from the pytorch implementation of [Self-Supervised Vision Transformers with DINO](https://github.com/facebookresearch/dino).<br/>

## Inference
After preparing test dataset following the steps in the main repository, carry out the RGB-based mask propagation process with dynamic refinements by
```Shell
python eval_adaptation.py --dataset DAVIS17m --data_path /path/to/DAVIS17m/ --seg_path /path/to/seg_dir/ --output_path /path/to/save_dir/
```
where ```--data_path```,  ```--seg_path``` and ```--output_path``` correspond to the location of test dataset, flow-predicted masks and the saving path, respectively.

Detailed parameters regarding the DINO transformer model and the propagation process can be checked and modified in ```eval_adaptation.py```

#### To setup your own data: 
* Add you own dataset information in  ```eval_adaptation.py```
* Replace the arguments with the dataset name and correponding paths


## DINO feature finetuning
Coming soon.

## License
This part of repository is released under the same Apache 2.0 license as the original code, which can found in the ```LICENSE``` file.