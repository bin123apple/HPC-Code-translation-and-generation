# HPC-Code-translation-and-generation


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














