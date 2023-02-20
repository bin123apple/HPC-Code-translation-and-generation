# HPC-Code-translation-and-generation

The main task of this github is to test the translation and generation performance of fortran HPC code using some existing open source projects
CodeXGLUE : https://github.com/microsoft/CodeXGLUE  \n
ChatGPT : https://openai.com/blog/chatgpt/

## Task1: Code to Code Translation

### Fortran to C translation by using Fine-tuned CodeBERT model
The fine-tuned CodeBERT model is in https://drive.google.com/file/d/177B19VLstHLXQYAdjce29EDZ1AWEUTV-/view?usp=share_link

Test BLEU score

```
cd path/to/code-to-code-trans/code/
```

```
python run.py \
    	--do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path /path/to/fine-tuned model \
	--dev_filename /path/to/valid.fortran2C.txt.f90,/path/to/valid.fortran2C.txt.C \
	--test_filename /path/to/test.fortran2C.txt.f90,/path/to/test.fortran2C.txt.C \
	--output_dir /path/to/your output file \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 
```

Test CodeBLEU score

```
cd path/to/CodeBLEU/
```

```
python calc_code_bleu.py --refs /path/to/your output file/test_1.gold --hyp /path/to/your output file/test_1.output --lang c_sharp --params 0.25,0.25,0.25,0.25
```

### Fortran to C translation by using ChatGPT
The question provided to chatGPT: Please help me to translate the following C code(The C code in our test dataset) to Fortran code.
NOTE: ChatGPT may generate different answers each time. The answer I got is shown in /Code to Code Translation dataset/ChatGPT_test_answer.output 

Test BLEU score and CodeBLEU score

```
python calc_code_bleu.py --refs /path/to/test_1.gold --hyp /path/to/ChatGPT_test_answer.output --lang c_sharp --params 0.25,0.25,0.25,0.25
```

### Java to C# translation by using CodeBERT model
The data is from this paper: https://arxiv.org/abs/2102.04664


## Task2: Code Generation Based on Text

### Text to Java Code generation by using CodeGPT
The data is from this paper: https://arxiv.org/abs/2102.04664

### Text to Fortran HPC Code generation by using fine-tuned CodeGPT model.

Fine tuned model
```
DATADIR=../dataset/Fortran
OUTPUTDIR=../save/Fortran
PRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2    # will download pre-trained CodeGPT model
LOGFILE=text2code_concode.log
PER_NODE_GPU=1
python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=6 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=30 \
        --logging_steps=100 \
        --save_steps=100 \
        --overwrite_output_dir \
        --seed=42
```

inference

```
DATADIR=../dataset/Fortran
OUTPUTDIR=../save/Fortran
PRETRAINDIR=../save/Fortran/checkpoint-last
LOGFILE=text2code_concode_infer.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_infer \
        --logging_steps=100 \
        --seed=42
```

NOTE: Our fortran HPC is not sufficient to support training this large model, Therefore the generated results are not ideal. If you want to test the text-code generation in a large Java dataset, Check the original project https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code. Although their result is also not ideal :).

### Text to Fortran HPC Code generation by using ChatGPT.

```
python calc_code_bleu.py --refs /path/to/ChatGPT_results/result.gold --hyp /path/to/ChatGPT_results/ChatGPT_result.output --lang c_sharp --params 0.25,0.25,0.25,0.25
```














