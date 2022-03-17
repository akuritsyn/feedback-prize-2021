# [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/c/feedback-prize-2021)

### Top 2% solution (36/2060) based on a baseline model developed by [Abhishek Thakur](https://github.com/abhishekkrthakur/long-text-token-classification).

## [Kaggle write-up](https://www.kaggle.com/c/feedback-prize-2021/discussion/313452)


## Requirements

- Python 3.9.7
- [Pytorch](https://pytorch.org/) 1.10.1
- [Transformers](https://huggingface.co/docs/transformers/index) 4.15.0


## Architecture

- The solution is an ensemble of 5 transformer models, each trained on 5 folds: 2x `deberta-large`, 2x `deberta-v3-large` and 1x `longformer-large`. The two versions of deberta models were trained using `dropout=0.1` and `dropout=0.15` and `max_len=1024` parameters, while for longformer-large model `dropout=0.1` and `max_len=1536` parameters were used. 

## Preparation

- Put `./data` directory in the root level and unzip the files downloaded from [Kaggle](https://www.kaggle.com/c/feedback-prize-2021/data) there. 
- In order to use deberta v2 or v3, you need to patch transformers library to create a new fast tokenizer using data and instructions from [this](https://www.kaggle.com/nbroad/deberta-v2-3-fast-tokenizer) kaggle dataset.
- Download `microsoft/deberta-large`, `microsoft/deberta-v3-large` and `allenai/transformer-large` or any other transformer models using [nbs/download_model.ipynb](https://github.com/akuritsyn/feedback-prize-2021/blob/main/nbs/download_model.ipynb) and save them in `./model` folder.
- Create 5 training folds using [nbs/creating_folds.ipynb](https://github.com/akuritsyn/feedback-prize-2021/blob/main/nbs/creating_folds.ipynb).

## Training

Please make sure you run the script from parent directory of `./bin`. 

~~~
$ sh ./bin/train.sh
~~~

To train different models on different folds (0...4) make changes inside the `train.sh` file. 

The training of each fold should fit into 15GB GPU memory.


## Inference

Use [ensemble_inference_oof.ipynb](https://github.com/akuritsyn/feedback-prize-2021/blob/main/nbs/ensemble_inference_oof.ipynb)
