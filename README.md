# BERT with multi-GPU

## Install horovod

[Horovod](https://github.com/uber/horovod)

## Run pretraining with multi-GPU

```shell
export BERT_BASE_DIR=/path/to/bert/xxxxxxxx_L-12_H-768_A-12

mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python run_pretraining.py \
      --input_file=/tmp/tf_examples.tfrecord \
      --output_dir=/tmp/pretraining_output \
      --do_train=True \
      --do_eval=True \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --train_batch_size=32 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --num_train_steps=20 \
      --num_warmup_steps=10 \
      --learning_rate=2e-5 \
      --gpus=0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 \
```
Here, --gpus indicate the gpu used by each worker.
