# [Feedback Prize - evaluating Student Writing] (https://www.kaggle.com/c/feedback-prize-2021)

## Top 2% solution (36/2060) based on an initial model developed by [Abhishek Thakur](https://github.com/abhishekkrthakur/long-text-token-classification).

## Requirements

- Python 3.9.7
- [Pytorch](https://pytorch.org/) 1.10.1
- [Transformers](https://huggingface.co/docs/transformers/index) 4.15.0


# Architecture

- The solution is an ensemble of 5 models, each trained on 5 folds: 2x deberta-large, 2x deberta-v3-large and 1x longformer-large.

## Preparation

- Put `./data` directory in the root level and unzip the files [downloaded](https://www.kaggle.com/c/feedback-prize-2021/data) from Kaggle there. 
- In order to use deberta v2 or v3, you need to patch transformers library to create a new fast tokenizer using instructions from this [kaggle kernel](https://www.kaggle.com/nbroad/deberta-v2-3-fast-tokenizer)
- Download 'microsoft/deberta-large', 'microsoft/deberta-v3-large' and 'allenai/transformer-large' or any other models using [nbs/download_model.ipynb](https://github.com/akuritsyn/feedback-prize-2021/blob/main/nbs/download_model.ipynb) and save them in './model' folder.
- Create 5 training folds using [nbs/creating_folds.ipynb](https://github.com/akuritsyn/feedback-prize-2021/blob/main/nbs/creating_folds.ipynb).

## Training

Please make sure you run the script from parent directory of `./bin`. 

~~~
$ sh ./bin/train.sh
~~~

To train different models and different folds (0...4) change inside train.sh files. The two version of deberta models were trained using 'dropout=0.1' and 'dropout=0.15' and 'max_len=1024', while for longformer-large 'dropout=0.1' and 'max_len=1536' parameters were used. 

The training should fit into 15GB GPU memory.


## Inference

Use [ensemble_inference_oof.ipynb](https://github.com/akuritsyn/feedback-prize-2021/blob/main/nbs/ensemble_inference_oof.ipynb)