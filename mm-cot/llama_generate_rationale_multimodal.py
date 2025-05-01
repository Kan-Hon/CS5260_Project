import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import LlamaForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig  # using base Trainer for causal LM
from trl import SFTTrainer, SFTConfig
from model import T5ForMultimodalGeneration, LlamaForMultimodalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStdInference, ScienceQADatasetStd, ScienceQADatasetImg, ScienceQADatasetImgSFT, ScienceQADatasetImgSFTHF
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face access token.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest', 'superminitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_lora', action='store_true', help='use lora or not')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--use_quant', action='store_true', help='use quantisation or not')
    parser.add_argument('--dataset', type=str, default='scienceqa')

    args = parser.parse_args()
    return args


def LlamaInferencer(
    dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir
    args.use_caption=False
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # IMPORTANT for decoder-only generation


    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    # train_qids = qids['train']
    test_qids = qids['test']
    # val_qids = qids['val']
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}_lora{args.lora_r}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)
    patch_size = img_shape[args.img_type]
    model = LlamaForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, token=hf_token) 

    image_features = dataframe['image_features']
    test_set = ScienceQADatasetImgSFT(
        problems,
        test_qids,
        name_maps,
        tokenizer,
        args.input_len,
        args.output_len,
        args,
        image_features,
    )
            # ----- Wrap Model with LoRA (if specified) -----
    if getattr(args, "use_lora", False):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,       # set to False for training
            r=getattr(args, "lora_r", 32),           # LoRA rank (default 8)
            lora_alpha=getattr(args, "lora_alpha", 32),  # scaling factor (default 32)
            lora_dropout=getattr(args, "lora_dropout", 0.1),  # dropout (default 0.1)
        )
        model = get_peft_model(model, lora_config)

        # print("Trainable parameters after applying LoRA:")
        # model.print_trainable_parameters()

    # datacollator =  DataCollatorWithPadding(tokenizer=tokenizer)
    # datacollator =  DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())
    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        # modified for zero shot 1b-instruct
        if len(res) >= 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED" 
        return answer  

    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        targets = np.where(targets != -100, targets, tokenizer.pad_token_id)

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct +=1 
        return {'accuracy': 1.0*correct/len(targets)}
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        pred_ls = []
        for pred in preds:
            try:
                pred = re.sub(r'^\.\s*', '', pred)
                token_pred = "\n".join(nltk.sent_tokenize(pred))
                pred_ls.append(token_pred)

            except Exception as e:
                print(f'ERROR:: Tokenisation of "{pred}" failed {e}')
                pred_ls.append('Solution: ')
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        preds = pred_ls
        return preds, labels


    from torch.utils.data import DataLoader

    loader = DataLoader(test_set, batch_size=args.eval_bs, shuffle=False)
    all_decoded_preds = []
    all_labels = []
    all_preds = []
    all_decoded_labels = []
    all_inputs = []

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    from tqdm import tqdm
    for batch in tqdm(loader):
        # --- generate preds ---
        inputs = {
            "input_ids":      batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
            "image_ids": batch["image_ids"].to(dtype=torch.float32).to(model.device),
        }
        outputs = model.generate(**inputs,
                                max_new_tokens =args.output_len,
                                do_sample=False)
        generated_outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_outputs = generated_outputs.cpu()

        
        texts = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        all_decoded_preds.extend(texts) 

        # --- decode the gold labels ---
        # batch["labels"] is shape (bs, seq_len) with -100 on padding
        labels = batch["labels"]
            # 2) if it’s not already a Tensor, convert it
        if not torch.is_tensor(labels):
            # shape will be (bs, seq_len)
            labels = torch.tensor(labels, dtype=torch.long, device=model.device)

        labels = torch.where(labels != -100,
                            labels,
                            torch.full_like(labels, tokenizer.pad_token_id))
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
        all_labels.extend(labels)
        all_preds.extend(generated_outputs)
        all_decoded_labels.extend(decoded_labels)  
        all_inputs.extend(inputs['input_ids'])
    # 3) Call your accuracy‐computing fn on the tuple (preds, labels):
    
    # metrics = compute_metrics_acc((all_preds, all_labels))
    # print(f"→ accuracy: {metrics['accuracy']:.4f}")

    output_data = {"preds": all_decoded_preds,
            "labels": all_decoded_labels}
    output_prediction_file = os.path.join(save_dir,"predictions_ans_test.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_data, indent=4))

    

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids}
    LlamaInferencer(
        dataframe=dataframe,
        args = args
    )