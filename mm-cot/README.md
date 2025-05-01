# Multimodal Chain-of-Thought Reasoning in Language Models

<h5 align="center"><i>"Imagine learning a textbook without figures or tables."</i></h5>

Multimodal-CoT incorporates vision features in a decoupled training framework. The framework consists of two training stages: (i) rationale generation and (ii) answer inference. Both stages share the same model architecture but differ in the input and output.

![](vision_features/mm-cot.png)



## Datasets

Download the dataset from the following repository:

```
https://github.com/lupantech/ScienceQA/tree/main/data
```
The vision features (detr, resnet, clip, vit) are available at https://huggingface.co/cooelf/vision_features/tree/main

Alternatively, you may download the extracted vision features (detr, resnet, clip) from [vision_features](https://drive.google.com/file/d/13B0hc_F_45-UlqPLKSgRz-ALtFQ8kIJr/view?usp=share_link) and unzip the files under `vision_features`

OR you can get it from my google drive: 
`vision_features: https://drive.google.com/drive/folders/10jdTp3lDxWC3e14_5bqtHFUrSslAHVTK?usp=sharing`
`models: https://drive.google.com/drive/folders/1YtIzygjcgs0dmsxQMfOpbwEiOVD2M59Q?usp=sharing`

## Trained Models
The trained rationale generation and answer generation files from experiments 1-8 are uploaded to HuggingFace.

```
https://huggingface.co/collections/kanhon/cs5260-mmcot-6812f3d15a456cf3a40b04d9

```


## Replicating Experiments conducted 

### Setup
```
1. conda create -n mmcot python=3.10
2. pip install -r requirements.txt
3. download models, vision features
4. pip install "huggingface-hub<0.26"
5. pip install transformers[torch]
```

If nltk punkt is not available:
```
python
>>> import nltk
>>> nltk.download('punkt')
```

```
export HF_TOKEN="hf_XXXXX"
```

IMPORTANT: When running SFT script `main_llama_sft.py`, please run the following
```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"
```

### Commands

Experiment 1:
```
pip install transformers==4.30.0

python main.py   --data_root data     --caption_file data/instruct_captions.json     --model declare-lab/flan-alpaca-base     --user_msg rationale --img_type vit     --bs 6 --eval_bs 8 --epoch 20 --lr 5e-5 --output_len 512     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments --final_eval

python main.py     --data_root data      --caption_file data/instruct_captions.json     --model declare-lab/flan-alpaca-base     --user_msg answer --img_type vit     --bs 8 --eval_bs 8 --epoch 20 --lr 5e-5 --output_len 64     --use_caption --use_generate --prompt_format QCMG-A     --output_dir experiments     --eval_le experiments/rationale_declare-lab-flan-alpaca-base_vit_QCM-E_lr5e-05_bs6_op512_ep20/predictions_ans_eval.json    --test_le experiments/rationale_declare-lab-flan-alpaca-base_vit_QCM-E_lr5e-05_bs6_op512_ep20/predictions_ans_test.json --final_eval
```

Experiment 2:
```
pip install transformers==4.30.0

python main.py   --data_root data     --caption_file data/instruct_captions.json     --model declare-lab/flan-alpaca-large     --user_msg rationale --img_type vit     --bs 2 --eval_bs 2 --epoch 20 --lr 5e-5 --output_len 512     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments --final_eval

python main.py     --data_root data      --caption_file data/instruct_captions.json     --model declare-lab/flan-alpaca-large     --user_msg answer --img_type vit     --bs 2 --eval_bs 2 --epoch 20 --lr 5e-5 --output_len 64     --use_caption --use_generate --prompt_format QCMG-A     --output_dir experiments     --eval_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs2_op512_ep20/predictions_ans_eval.json    --test_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs2_op512_ep20/predictions_ans_test.json --final_eval
```

Experiment 3:
Run `scrub_rationale_json.ipynb` to create json files that contain empty strings for rationale.

After that, run the following command to start training answer generation directly:

```
pip install transformers==4.30.0

python main.py     --data_root data      --caption_file data/instruct_captions.json     --model declare-lab/flan-alpaca-base     --user_msg answer --img_type vit     --bs 8 --eval_bs 8 --epoch 20 --lr 5e-5 --output_len 64     --use_caption --use_generate --prompt_format QCMG-A     --output_dir experiments_no_rationale     --eval_le "experiments/rationale_scrubbed_QCM-E/predictions_ans_test.json"    --test_le "experiments/rationale_scrubbed_QCM-E/predictions_ans_test.json" --final_eval

```

Experiment 4:
Run `scrub_rationale_json.ipynb` to create json files that contain empty strings for rationale.

After that, run the following command to start training answer generation directly:

```
pip install transformers==4.30.0

python main.py     --data_root data      --caption_file data/instruct_captions.json     --model declare-lab/flan-alpaca-large     --user_msg answer --img_type vit     --bs 2 --eval_bs 2 --epoch 20 --lr 5e-5 --output_len 64     --use_caption --use_generate --prompt_format QCMG-A     --output_dir experiments_no_rationale     --eval_le "experiments/rationale_scrubbed_QCM-E/predictions_ans_test.json"    --test_le "experiments/rationale_scrubbed_QCM-E/predictions_ans_test.json" --final_eval
```

Experiment 5:
```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"

python main_llama_sft.py --data_root data --caption_file data/instruct_captions.json --model meta-llama/Llama-3.2-1B --user_msg rationale --img_type vit --bs 2 --eval_bs 2 --epoch 15 --lr 1e-4 --output_len 512 --use_caption --use_generate --prompt_format QCM-E --output_dir experiments --final_eval
```
```
pip install transformers==4.30.0

python main.py --data_root data --caption_file data/instruct_captions.json --model declare-lab/flan-alpaca-base --user_msg answer --img_type vit --bs 1 --eval_bs 1 --epoch 10 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCMG-A --output_dir experiments --final_eval --eval_le experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr0.0001_bs4_lr0.0001_bs2_op512_ep15_full/predictions_ans_eval.json --test_le experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr0.0001_bs4_lr0.0001_bs2_op512_ep15_full/predictions_ans_test.json 
```

Experiment 6:
```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"

python main_llama_sft.py --data_root data --caption_file data/instruct_captions.json --model meta-llama/Llama-3.2-1B --user_msg rationale --img_type vit --bs 2 --eval_bs 2 --epoch 15 --lr 1e-4 --output_len 512 --use_caption --use_generate --prompt_format QCM-E --output_dir experiments --final_eval
```
```
pip install transformers==4.30.0
python main.py --data_root data --caption_file data/instruct_captions.json --model declare-lab/flan-alpaca-large --user_msg answer --img_type vit --bs 1 --eval_bs 1 --epoch 10 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCMG-A --output_dir experiments --final_eval --eval_le experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr0.0001_bs4_lr0.0001_bs2_op512_ep15_full/predictions_ans_eval.json --test_le experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr0.0001_bs4_lr0.0001_bs2_op512_ep15_full/predictions_ans_test.json 
```


Experiment 7:

```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"

python main_llama_sft.py --data_root data --caption_file data/instruct_captions.json --model meta-llama/Llama-3.2-1B --user_msg rationale --img_type vit --bs 4 --eval_bs 4 --epoch 30 --lr 5e-5 --output_len 512 --use_caption --use_generate --prompt_format QCM-E --output_dir experiments --final_eval --use_lora --lora_r 32 
```
```
pip install transformers==4.30.0
python main.py --data_root data --caption_file data/instruct_captions.json --model declare-lab/flan-alpaca-base --user_msg answer --img_type vit --bs 4 --eval_bs 4 --epoch 10 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCMG-A --output_dir experiments --final_eval --eval_le experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr5e-05_bs4_op512_ep30_lora32/predictions_ans_eval.json --test_le experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr5e-05_bs4_op512_ep30_lora32/predictions_ans_test.json 
```

Experiment 8:

```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"
```

```
python llama_inference.py --data_root data --caption_file data/instruct_captions.json --model meta-llama/Llama-3.2-1B-Instruct --user_msg answer --bs 4 --eval_bs 32 --epoch 10 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCMG-A --output_dir experiments --final_eval --use_lora --lora_r 32 
```

### Commands for MM-Star

MM-Star inference can be run as follow:
Download `data_mmstar` data and `vision_features/mmstar` data from Google Drive. 
`data_mmstar: https://drive.google.com/drive/folders/1I2dYq0h9LTMW4-bcXgW8AHkd1Nxi_tb6?usp=sharing`

`vision_features/mmstar: https://drive.google.com/drive/folders/10jdTp3lDxWC3e14_5bqtHFUrSslAHVTK?usp=sharing`

#### To run generate rationale using llama on Pretrained model:

```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"
```

Change --model to the ANSWER model to be tested
```
python llama_generate_rationale_multimodal.py --data_root data_mmstar --caption_file data/instruct_captions.json --model experiments/rationale_meta-llama-Llama-3.2-1B_vit_QCM-E_lr0.0001_bs2_op512_ep30_lora32_run2 --user_msg rationale --img_type vit --bs 4 --eval_bs 8 --epoch 1 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCM-E --output_dir experiments --dataset mmstar
```
#### To run inference on Encoder-Decoder model:

```
pip install transformers==4.30.0
pip install "huggingface_hub<0.26"
```

Change --model to the ANSWER model to be tested, and --test_le to the rationale json file
```
python alpaca_inference.py --data_root data_mmstar --caption_file data/instruct_captions.json --model models/mm-cot-base-ans --user_msg answer --img_type vit --bs 4 --eval_bs 8 --epoch 1 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCMG-A --output_dir experiments --eval_le models/mm-cot-base-ans\\predictions_ans_eval.json --test_le models/mm-cot-base-ans\\predictions_ans_test.json --dataset mmstar
```

#### To run inference on Pretrained model:

```
pip install peft==0.10.0 trl==0.11.4 "huggingface-hub<0.26"
```

```
python llama_inference.py --data_root data_mmstar --caption_file data/instruct_captions.json --model meta-llama/Llama-3.2-1B-Instruct --user_msg answer --img_type vit --bs 4 --eval_bs 8 --epoch 1 --lr 5e-5 --output_len 64 --use_caption --use_generate --prompt_format QCMG-A --output_dir experiments --dataset mmstar
```

## Extract Features (optional)

The processed vision features for ScienceQA are available at https://huggingface.co/cooelf/vision_features/tree/main. 

The following instructions show how we obtain those features.

Download the image files from [Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev?usp=sharing) and unzip all the images (train, dev, test) in the same folder (). The structure should be:

```
images
├── 1
│   └── image.png
├── 2
│   └── image.png
├── 3
│   └── image.png
├── 5
│   └── image.png
├── 7
│   └── image.png
```

Run ```extract_features.py --data_root images --output_dir vision_features --img_type vit```

If you hope to use your own images, please structure those images in the way above, or modify the script ```extract_features.py```.

## Extract Captions (optional)

The processed captions for ScienceQA are available at ```data/instruct_captions.json```. 

The following instructions show how we obtain those features.

Intall lavis and prepare Vicuna weights to use InstructBLIP for caption extraction.

https://github.com/salesforce/LAVIS/tree/f982acc73288408bceda2d35471a8fcf55aa04ca/projects/instructblip

Assume that the images are stored in the ```images``` folder. 

```
python extract_caption.py
```