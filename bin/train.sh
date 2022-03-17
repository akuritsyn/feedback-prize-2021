fold=0
gpu=0

model=./model/microsoft/deberta-large
# model=./model/microsoft/deberta-v3-large
# model=./model/allenai/transformer-large-4096

output=./output/deberta-large-test
# output=./output/deberta-v3-large-1
# output=./output/transformer-large-1

# checkpoint=./output/deberta-large-0
lr=2e-5
dropout=0.15
epochs=5
max_len=512
batch_size=1
accumulation_steps=4
valid_batch_size=1

python src/train.py --fold $fold --model $model --output $output --lr $lr \
--dropout $dropout --epochs $epochs --max_len $max_len \
--accumulation_steps $accumulation_steps --batch_size $batch_size \
--valid_batch_size $valid_batch_size --gpu $gpu
# --checkpoint $checkpoint