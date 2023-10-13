# HPC-Code-translation-and-generation

The main task of this github is to test the translation and generation performance of fortran HPC code using some existing open source projects  
CodeXGLUE : https://github.com/microsoft/CodeXGLUE  
ChatGPT : https://openai.com/blog/chatgpt/

The origianl Fortran HPC dataset can be downloaded from https://github.com/OMPI-X/epcc-mixedmode

## Task1: Code to Code Translation

### Fortran to C++ translation by using Fine-tuned CodeBERT model
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

### Fortran to C++ translation by using ChatGPT
The question provided to chatGPT: Please help me to translate the following C code (The C code in our test dataset) to Fortran code.
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
Question provided to ChatGPT, take FT calculation as an example: 

```
Please help me to write some Fortran HPC code to implements the time integration of a three-dimensional partial differential equation using the Fast Fourier Transform.
Please add OpenMP (Open Multi-Processing) directives into the code to make it run in parallel.
Please add MPI (Message Passing Interface) calls into the code to make it run in parallel in a cluster.
```

```
python calc_code_bleu.py --refs /path/to/ChatGPT_results/result.gold --hyp /path/to/ChatGPT_results/ChatGPT_result.output 
```

## Task3: Create our own model for the HPC code translation

Our paper is avaliable at [http://arxiv.org/abs/2307.07686](http://arxiv.org/abs/2307.07686).

This folder contains training and testing dataset and a simple test script.

We collect data form three different source: 

[Polybench](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/)

[NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html)

[dataracebench](https://github.com/LLNL/dataracebench)

You can also download the dataset from : [My Huggingface](https://huggingface.co/datasets/Bin12345/HPC_Fortran_CPP)

Here is one data pair example:

![Here is one data pair example:](https://github.com/bin123apple/OpenMP-Fortran-CPP-Translation/blob/main/Figures/Data%20Pair%20Example.png)

We will add more data pairs in the future and will add a new "nature language" column for code generation task.

## Reproduce our results

1. Finetune the model by using deepspeed
```
deepspeed --master_port 12345 main.py \
   --data_path Bin12345/HPC_Fortran_CPP \
   --model_name_or_path path/to/starcoder_model \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 128 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
```
If you want to finetune the other models (for example OPT models), you just need to change the `--model_name_or_path` from `path/to/starcoder_model` to `path/to/OPT_models`.

2. Use the finetuned model to generate the prompts. Change the 
```
model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b").to('cuda:2')
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
```
Inside the `Simple_test_script.py` to 
```
model = OPTForCausalLM.from_pretrained("path/to/the/fintuned_model").to('cuda:2')
tokenizer = AutoTokenizer.from_pretrained("path/to/the/fintuned_model")
```
Then run:
```
python Simple_test_script.py
```
You can try our simple test scripts. And for different models, there might be slightly difference.

3. Then test the CodeBlue Score
```
cd CodeBLUE
python calc_code_bleu.py --refs path/to/groundtruth.txt --hyp path/to/the_generated_answers/by_the_finetuned_model
```

## Reference 

```
@article{lu2021codexglue,
  title={Codexglue: A machine learning benchmark dataset for code understanding and generation},
  author={Lu, Shuai and Guo, Daya and Ren, Shuo and Huang, Junjie and Svyatkovskiy, Alexey and Blanco, Ambrosio and Clement, Colin and Drain, Dawn and Jiang, Daxin and Tang, Duyu and others},
  journal={arXiv preprint arXiv:2102.04664},
  year={2021}
}

@inproceedings{lei2023creating,
      title={Creating a Dataset for High-Performance Computing Code Translation using LLMs: A Bridge Between OpenMP Fortran and C++}, 
      author={Bin Lei and Caiwen Ding and Le Chen and Pei-Hung Lin and Chunhua Liao},
  booktitle={High Performance Extreme Computing Conference (HPEC), 2023 IEEE},
  year={2023},
  organization={IEEE}
}
```

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg













