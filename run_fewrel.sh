
rm -rf /home/wb-dyw512452/fewrel/pytorch_output
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/home/wb-dyw512452/fewrel

python /home/wb-dyw512452/fewrel/bert_pytorch/examples/run_classifier.py --data_dir=/home/wb-dyw512452/fewrel/data --task_name=fewrel \
    --do_train \
	--do_eval \
    --bert_model=/home/wb-dyw512452/fewrel/pre_trained \
    --max_seq_length=100 \
	--Nways=10 \
	--Kshot=5 \
	--meta_steps=64 \
	--meta_batch=16 \
    --lr_a=5e-4 \
	--learning_rate=5e-5 \
    --test_train_epochs=8 \
    --output_dir=/home/wb-dyw512452/fewrel/pytorch_output