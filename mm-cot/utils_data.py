import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, f'{args.dataset}/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, f'{args.dataset}/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids,

def load_data_img(args):
    problems = json.load(open(os.path.join(args.data_root, f'{args.dataset}/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, f'{args.dataset}/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open(os.path.join(args.data_root, f'name_map.json')))

    # check
    if args.dataset == 'mmstar':
         image_features = torch.load("vision_features/mmstar/vit.pth")
    else:
        if args.img_type == "resnet:":
            image_features = np.load('vision_features/resnet.npy')
            image_features = np.expand_dims(image_features, axis=1)
            image_features = image_features.repeat(512, axis=1)
        elif args.img_type == "clip":
            image_features = np.load('vision_features/clip.npy')
        elif args.img_type == "detr":
            image_features = np.load('vision_features/detr.npy')
        elif args.img_type == "vit":
            image_features = torch.load("vision_features/vit.pth")
        else:
            image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", image_features.shape)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""
    if args.dataset == 'scienceqa':
        train_qids = pid_splits['%s' % (args.train_split)]
        val_qids = pid_splits['%s' % (args.val_split)]
        test_qids = pid_splits['%s' % (args.test_split)]
        print(f"number of train problems: {len(train_qids)}\n")
        print(f"number of val problems: {len(val_qids)}\n")
        print(f"number of test problems: {len(test_qids)}\n")
        qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    elif args.dataset == 'mmstar':

        test_qids = pid_splits['%s' % (args.test_split)]

        print(f"number of test problems: {len(test_qids)}\n")
        qids = {'test':test_qids}

    
    return problems, qids, name_maps, image_features
class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }

class ScienceQADatasetStdInference(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                # curr_le_data = None
                curr_le_data = '' # let rationale be empty for llama inference 
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text + "Give your answer in this format 'The answer is (X)' where X is the option."],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=False,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }



class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }
    
class ScienceQADatasetImgInference(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }
    
class ScienceQADatasetImgSFT(ScienceQADatasetImg):

    def __getitem__(self, index):
        # Get the original prompt and answer from the parent class.
        parent_dict = super().__getitem__(index)
        # These come from the parent: input_ids originally contains the prompt tokens,
        # and labels (currently a list) contains the answer tokens.
        prompt_ids = parent_dict["input_ids"]  # Tensor containing the prompt tokens
        answer_ids = parent_dict["labels"]       # List of answer token ids

        # Convert prompt_ids to a list for slicing.
        prompt_ids_list = prompt_ids.tolist()

        # Determine the token id for the string "Solution:".
        solution_token_id = self.tokenizer.encode("Solution:", add_special_tokens=False)[0]

        # Find the index of the "Solution:" token in the prompt.
        if solution_token_id in prompt_ids_list:
            split_index = prompt_ids_list.index(solution_token_id) + 1  # include "Solution:" token
        else:
            split_index = len(prompt_ids_list)  # fallback if not found

        # Set input_ids to only include tokens up to (and including) "Solution:".
        new_input_ids = torch.tensor(prompt_ids_list[:split_index], dtype=torch.long)

        # For labels, we want the full merged sequence (prompt + answer).
        merged_labels_list = prompt_ids_list + answer_ids  # no masking, full sequence
        new_labels = torch.tensor(merged_labels_list, dtype=torch.long)

        # Adjust the attention mask for input_ids (all ones for the prompt tokens).
        new_attention_mask = torch.ones_like(new_input_ids)

        return {
            "input_ids": new_input_ids,             # only prompt up to "Solution:"
            "attention_mask": new_attention_mask,
            "labels": new_labels,                   # complete sequence (prompt + answer)
            "image_ids": parent_dict["image_ids"]   # unchanged image features
        }
class ScienceQADatasetImgSFTHF(ScienceQADatasetImg):
    """
    Same logic as before, but __getitem__ now returns *Python lists*
    (which are automatically serialisable by 🤗 Datasets) instead of
    torch tensors.  No other change is needed.
    """

    def __getitem__(self, index):
        parent = super().__getitem__(index)

        # --- unpack original fields ---
        prompt_ids  = parent["input_ids"]          # tensor
        answer_ids  = parent["labels"]             # list[int]
        image_feats = parent["image_ids"]          # tensor

        # ---- trim prompt up to and inc. "Solution:" --------------
        prompt_ids_list = prompt_ids.tolist()
        sol_tok = self.tokenizer.encode(
            "Solution:", add_special_tokens=False
        )[0]

        cut = prompt_ids_list.index(sol_tok) + 1 if sol_tok in prompt_ids_list else len(prompt_ids_list)

        new_input_ids     = prompt_ids_list[:cut]                 
        merged_label_list = prompt_ids_list + answer_ids          
        new_attention     = [1] * len(new_input_ids)              

        return {
            "input_ids":       new_input_ids,         
            "attention_mask":  new_attention,         
            "labels":          merged_label_list,     
            "image_ids":       image_feats.tolist(),  
        }
