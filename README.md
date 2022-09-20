compressedBART
===

<img src="https://raw.githubusercontent.com/AlaFalaki/compressedBART/main/figures/proposed.png" width="700"/>


### A Robust Approach to Fine-tune Pre-trained Transformer-based models for Text Summarization through Latent Space Compression

This repository is the official PyTorch implementation of the mentioned paper that is consists of:

- Architecture codes (BART + AE) (architectures.py)
- Inference process (main.py)
- Experiments checkpoints

## Results
The results using the CNN/DM dataset.

| Latent Space Size | R-1 | R-2 | R-3 | R-L | Checkpoint
|:---:|:---:|:---:|:---:|:---:|:---:|
| 504 | 0.401 | 0.182 | 0.106 | 0.375 | [link](https://uwin365-my.sharepoint.com/:u:/g/personal/alamfal_uwindsor_ca/Ef42Eldx5HBCs9yXZ_xXAwgBS8-KbRMjvQZ8KG9AZuu60w?e=g8KUnZ) |
| 384 | 0.400 | 0.181 | 0.105 | 0.374 | [link](https://uwin365-my.sharepoint.com/:u:/g/personal/alamfal_uwindsor_ca/EcFD97dBoj9AjhbehFCZv5kBJ_nIKoABnhJ-PhJooNDGSw?e=V9n3DL) |

There is a minor difference between the two compression sizes. But, there is a great advantage in using 
the 384 checkpoint since it is faster, and smaller. (Refer to the paper for more details)

*The link will expire after 367 days, please open up an issue so I replace them with new ones.*

## Usage

First, you need to download the checkpoint that you like to work with, put it in the 'cbart-checkpoints' directory, and lastly, uncompress the file using the following command.

`tar -xvf <checkpoint_name>.tar.gz`

This is how your directory tree should look like.

 * cbat-checkpoint
    * 394 [or 504]
        * ae-checkpoint.pth
        * config.json
        * pytorch_model.bin

Now, you can the following commands to run the inference and get the results on the CNN/DM dataset "test" set.

**384 checkpoint**

```
python main.py \
    --checkpoint_dir ./cbart-checkpoints/384 \
    --exp_name 384 \
    --batch_size 32 \
    --first 576 \
    --second 480 \
    --third 384
```

**504 checkpoint**

```
python main.py \
    --checkpoint_dir ./cbart-checkpoints/504 \
    --exp_name 504 \
    --batch_size 32 \
    --first 640 \
    --second 576 \
    --third 504
```

You will find a CSV file containing the generated summaries in the `results` directory when the process finishes.

**Note:** There is a `--test` parameter you can append to the mentioned commands to just test the code with only one sample.
## Requirements

* Python 3.8.10
* torch 1.10.0
* transformers 4.19.2
* datasets 1.18.3
* rouge-score 0.0.4

## Citations
If you wish to cite the paper, you may use the following:
```
The paper is accepted in the ICMLA22 conference.
```

GL!