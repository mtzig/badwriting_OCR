import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
from datasets import load_metric
from transformers import VisionEncoderDecoderModel

import os
import json
import random
import numpy as np
from collections import defaultdict

cer_metric = load_metric("cer")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

def getModel(use_config=False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    if use_config:
        # set beam search parameters
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.max_length = 64
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

    return model

def compute_cer(pred_ids, label_ids, debug=False):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    if debug:
        print(pred_str, label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
        
class IAMDatasetAugmented(Dataset):
    def __init__(self, root_dir, df, processor, transform, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        image = self.transform(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


class IAM_fewshot_dataset(Dataset):
    def __init__(self,
                 image_dir,
                 meta_filename,
                 processor,
                 max_target_length=128,
                 episode_num=600,
                 shot=5,):

        self.image_dir = image_dir
        self.episode_num = episode_num
        self.shot = shot
        self.processor = processor
        self.max_target_length = max_target_length

        with open(meta_filename, 'r') as json_file:
            meta_data = json.load(json_file)

        for i in range(len(meta_data)):
            sample = meta_data[i]
            dir = os.path.join(image_dir, sample['image_dir'])
            if not os.path.exists(dir):
                print(dir, os.path.exists(dir))
                raise Exception

        self._writer_id_to_ind = {}
        writer_ind = 0
        for sample in meta_data:
            if sample['writer_id'] not in self._writer_id_to_ind:
                self._writer_id_to_ind[sample['writer_id']] = writer_ind
                writer_ind += 1

        self._ind_to_writer_id = {value: key for key, value in self._writer_id_to_ind.items()}

        self.writer_samples = [[] for ind in self._ind_to_writer_id]
        for sample in meta_data:
            writer_id = sample['writer_id']
            writer_ind = self._writer_id_to_ind[writer_id]
            self.writer_samples[writer_ind].append(sample)

        self.writer_num = len(self.writer_samples)

    def __len__(self,):
        return self.episode_num

    def get_encoding(self, sample):
        # get file name + text
        file_name = os.path.join(self.image_dir, sample['image_dir'])
        text = ' '.join(sample['transcription'])

        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


    def __getitem__(self, idx):
        # get writer
        while True:
            writer_ind = np.random.randint(0, self.writer_num)
            samples = self.writer_samples[writer_ind]
            if len(samples) > self.shot:
                break

        random.shuffle(samples)
        supports = samples[:self.shot]
        query = samples[self.shot]

        supports = [self.get_encoding(sample) for sample in supports]
        query = self.get_encoding(query)

        pixel_values = []
        labels = []
        for batch in supports:
            pixel_values.append(batch['pixel_values'])
            labels.append(batch['labels'])
        pixel_values = torch.stack(pixel_values, 0)
        labels = torch.stack(labels, 0)
        supports = {'pixel_values': pixel_values, "labels": labels}
        return supports, query
    

class IAM_global_dataset(Dataset):
    def __init__(self,
                 image_dir,
                 meta_filename,
                 processor,
                 max_target_length=128):

        self.image_dir = image_dir
        self.processor = processor
        self.max_target_length = max_target_length

        with open(meta_filename, 'r') as json_file:
            self.meta_data = json.load(json_file)

        for i in range(len(self.meta_data)):
            sample = self.meta_data[i]
            dir = os.path.join(image_dir, sample['image_dir'])
            if not os.path.exists(dir):
                print(dir, os.path.exists(dir))
                raise Exception

    def __len__(self,):
        return len(self.meta_data)

    def get_encoding(self, sample):
        # get file name + text
        file_name = os.path.join(self.image_dir, sample['image_dir'])
        text = ' '.join(sample['transcription'])

        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


    def __getitem__(self, idx):
        sample = self.meta_data[idx]
        sample = self.get_encoding(sample)
        return sample

class IAMDatasetFromList(Dataset):
    def __init__(self, data_list, processor, max_target_length=128):
        self.processor = processor
        self.max_target_length = max_target_length
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # get file name + text
        file_name, text = self.data_list[idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    
class IAM_MAML_Outer(Dataset):

    def __init__(self, image_dir, meta_filename, processor, max_target_length=128, inner_batch_size=4, sample_thresh=10):
        
        self.processor = processor
        self.max_target_length = max_target_length
        self.inner_batch_size = inner_batch_size
        self.wd = tuple(self.sort_by_writer(image_dir, meta_filename, sample_thresh).values())



    def sort_by_writer(self, image_dir, meta_filename, sample_thresh=10):
        '''
        Only keep writer id's with at least sample_thres examples
        '''
    
        with open(meta_filename, 'r') as json_file:
            meta_data = json.load(json_file)

        writer_datas = defaultdict(list)
        for sample in meta_data:
            file_path =  os.path.join(image_dir, sample['image_dir'])
            text = ' '.join(sample['transcription'])
            writer_datas[sample['writer_id']].append((file_path, text))
        
        for key in list(writer_datas.keys()):
            if len(writer_datas[key]) < 10:
                del writer_datas[key]

        return writer_datas
    
    def __len__(self,):
        return len(self.wd)
    
    def __getitem__(self, index):
        dataset = IAMDatasetFromList(self.wd[index], self.processor, self.max_target_length)
        return DataLoader(dataset, batch_size=self.inner_batch_size)

def get_dataloaders(dataset_type='t', test_size=0.2, batch_size=4, root='', test=False, transform=None):

    if dataset_type=='t':
        root = Path(__file__).parents[1]
        df = pd.read_csv(f'{root}/data/thomas_writing/t_data.csv')
        train_df, test_df = train_test_split(df, test_size=test_size)
        # we reset the indices to start from zero
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = IAMDataset(root_dir=f'{root}/data/thomas_writing/',
                                df=train_df,
                                processor=processor)
        eval_dataset = IAMDataset(root_dir=f'{root}/data/thomas_writing/',
                                df=test_df,
                                processor=processor)
    elif dataset_type=='a':
        root = Path(__file__).parents[1]
        df = pd.read_csv(f'{root}/data/alex_writing/a_data.csv')
        train_df, test_df = train_test_split(df, test_size=test_size)
        # we reset the indices to start from zero
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        if transform:
            train_dataset = IAMDatasetAugmented(root_dir=f'{root}/data/alex_writing/',
                                    df=train_df,
                                    processor=processor, transform=transform)
        else:
            train_dataset = IAMDataset(root_dir=f'{root}/data/alex_writing/',
                                    df=train_df,
                                    processor=processor)
        eval_dataset = IAMDataset(root_dir=f'{root}/data/alex_writing/',
                                df=test_df,
                                processor=processor)
    elif dataset_type=='iam':
        # if we are using the iam dataset, need to use the optinoal root param to point to where it is
        # df = pd.read_csv(root + csv) # root should be iam_path

        # If we are just testing, don't do any splitting, just return the whole df for both train and test
        if test:
            train_df = pd.read_csv(root + '/iam_data_test.csv')
            test_df = pd.read_csv(root + '/iam_data_test.csv')
            # train_df = df
            # test_df = df
        else:
            # train_df, test_df = train_test_split(df, test_size=test_size)
            train_df = pd.read_csv(root + '/iam_data_train.csv')
            test_df = pd.read_csv(root + '/iam_data_val1.csv')
            
        # we reset the indices to start from zero
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = IAMDataset(root_dir=root+'/lines/',
                                df=train_df,
                                processor=processor)
        eval_dataset = IAMDataset(root_dir=root+'/lines/', # sentences or lines ?
                                df=test_df,
                                processor=processor)
    elif dataset_type=='iam_fewshot':
        sentence_path = f'{root}/data/IAM/sentences'
        global_train_dataset = IAM_global_dataset(sentence_path, f'{root}/data/IAM/meta_train_data.json', processor=processor)

        fewshot_train_dataset = IAM_fewshot_dataset( sentence_path, f'{root}/data/IAM/meta_train_data.json', processor=processor)
        fewshot_test_dataset = IAM_fewshot_dataset(sentence_path, f'{root}/data/IAM/meta_test_data.json', processor=processor)
        fewshot_val_dataset = IAM_fewshot_dataset( sentence_path, f'{root}/data/IAM/meta_val_data.json', processor=processor)

        global_train_dataloader = DataLoader(global_train_dataset, batch_size=batch_size)

        fewshot_train_dataloader = DataLoader(fewshot_train_dataset, batch_size=1)
        fewshot_test_dataloader = DataLoader(fewshot_test_dataset, batch_size=1)
        fewshot_val_dataloader = DataLoader(fewshot_val_dataset, batch_size=1)

        return global_train_dataloader, fewshot_train_dataloader, fewshot_val_dataloader, fewshot_test_dataloader
    elif dataset_type =='iam_maml':

        # for now just hardcode these values as I am lazy
        image_dir = './data/IAM/sentences/'
        meta_filename = './data/IAM/aa_te.json'

        dataset = IAM_MAML_Outer(image_dir, meta_filename, processor)
        return DataLoader(dataset, batch_size=None, shuffle=True)

    else:
        raise ValueError('dataset_type must be "t" or ...')
    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader

def classify_img(img, model):
  model.eval()

  pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

  generated_ids = model.generate(pixel_values, max_length=10)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  print(generated_text)
