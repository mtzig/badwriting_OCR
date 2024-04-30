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

def compute_cer(pred_ids, label_ids, as_strings=False, debug=False):
    if not as_strings:
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    else:
        pred_str = pred_ids
        label_str = label_ids
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

def get_dataloaders(dataset_type='t', test_size=0.2, batch_size=4, root='', test=False, transform=None, extended_thomas=False, thomas_spliting_tricks=False):

    if dataset_type=='t':
        root = Path(__file__).parents[1]
        if extended_thomas:
            df = pd.read_csv(f'{root}/data/thomas_writing/t_data_extended.csv')
        else:
            df = pd.read_csv(f'{root}/data/thomas_writing/t_data.csv')
        
        if thomas_spliting_tricks:
            assert df.shape[0] == 121
            train_df, test_df = train_test_split(df, test_size=1/6, shuffle=False)
        else:
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

  pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(model.device)

  generated_ids = model.generate(pixel_values, max_length=10)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  return generated_text


from transformers import VisionEncoderDecoderModel
import torch

from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VISION_ENCODER_DECODER_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from torch.nn import CrossEntropyLoss
from copy import deepcopy

class MyDualDecoderVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
        >>> decoder_tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")
        >>> model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> img = Image.open(requests.get(url, stream=True).raw)
        >>> pixel_values = image_processor(images=img, return_tensors="pt").pixel_values  # Batch size 1

        >>> output_ids = model.generate(
        ...     pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True
        ... ).sequences

        >>> preds = decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        >>> preds = [pred.strip() for pred in preds]

        >>> assert preds == ["a cat laying on top of a couch next to another cat"]
        ```"""

        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            raise not NotImplementedError
        # At the moment fast initialization is not supported for composite models
        # if kwargs.get("_fast_init", False):
        #     logger.warning(
        #         "Fast initialization is currently not supported for VisionEncoderDecoderModel. "
        #         "Falling back to slow initialization..."
        #     )
        kwargs["_fast_init"] = False

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.decoder2 = deepcopy(model.decoder)
        return model

    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> # training
        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> text = "hello world"
        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(pixel_values=pixel_values, labels=labels)
        >>> loss = outputs.loss

        >>> # inference (generation)
        >>> generated_ids = model.generate(pixel_values)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        
        decoder2_outputs = self.decoder2(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )        
        

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits + decoder2_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits + decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def getDualDecoderModel(use_config=False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model = MyDualDecoderVisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
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

from transformers import VisionEncoderDecoderModel
import torch

from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VISION_ENCODER_DECODER_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from torch.nn import CrossEntropyLoss
from copy import deepcopy
from transformers.models.trocr.modeling_trocr import TrOCRAttention
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.trocr.configuration_trocr import TrOCRConfig

my_config = TrOCRConfig(activation_dropout=0.0,
                        activation_function="relu",
                        add_cross_attention= True,
                        attention_dropout= 0.0,
                        bos_token_id= 0,
                        classifier_dropout= 0.0,
                        cross_attention_hidden_size= 384,
                        d_model= 256,
                        decoder_attention_heads= 8,
                        decoder_ffn_dim= 1024,
                        decoder_layerdrop= 0.0,
                        decoder_layers= 6,
                        decoder_start_token_id= 2,
                        dropout= 0.1,
                        eos_token_id= 2,
                        init_std= 0.02,
                        is_decoder= True,
                        layernorm_embedding= True,
                        max_position_embeddings= 512,
                        model_type= "trocr",
                        pad_token_id= 1,
                        scale_embedding= True,
                        tie_word_embeddings= False,
                        transformers_version= "4.37.2",
                        use_cache= False,
                        use_learned_position_embeddings=True,
                        vocab_size= 64044)



class FinetuneAdaptor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_mapping = nn.Linear(dim, dim//2)
        # self.middle_mapping = nn.Linear(dim*2, dim*2)
        self.output_mapping = nn.Linear(dim//2, dim)
        self.act = nn.ReLU(inplace=False)
        
    def forward(self, feat):
        feat = self.input_mapping(feat)
        feat = self.act(feat)
        # feat = self.middle_mapping(feat)
        # feat = self.act(feat)
        feat = self.output_mapping(feat) 
        return feat
        

class MyTrOCRDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = TrOCRAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if config.is_decoder:
            self.encoder_attn = TrOCRAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                kdim=config.cross_attention_hidden_size,
                vdim=config.cross_attention_hidden_size,
                dropout=config.attention_dropout,
                is_decoder=True,
                is_cross_attention=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.encoder_adaptor = FinetuneAdaptor(self.embed_dim)
        self.decoder_adaptor = FinetuneAdaptor(self.embed_dim)
        self.linear_layer_adaptor = FinetuneAdaptor(self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        layer_head_mask = None,
        cross_attn_layer_head_mask = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = True,
    ):

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # go through the adaptor layer 
        hidden_states = self.encoder_adaptor(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # go through the adaptor layer 
            hidden_states = self.decoder_adaptor(hidden_states)            
            
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states) 
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # go through the adaptor layer 
        hidden_states = self.linear_layer_adaptor(hidden_states)            
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class MyVisionEncoderDecoderAdaptorModel(VisionEncoderDecoderModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
        >>> decoder_tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")
        >>> model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> img = Image.open(requests.get(url, stream=True).raw)
        >>> pixel_values = image_processor(images=img, return_tensors="pt").pixel_values  # Batch size 1

        >>> output_ids = model.generate(
        ...     pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True
        ... ).sequences

        >>> preds = decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        >>> preds = [pred.strip() for pred in preds]

        >>> assert preds == ["a cat laying on top of a couch next to another cat"]
        ```"""

        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            raise not NotImplementedError
        # At the moment fast initialization is not supported for composite models
        # if kwargs.get("_fast_init", False):
        #     logger.warning(
        #         "Fast initialization is currently not supported for VisionEncoderDecoderModel. "
        #         "Falling back to slow initialization..."
        #     )
        kwargs["_fast_init"] = False

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        with torch.no_grad():
            temp = deepcopy(model.decoder.model.decoder.layers)
            model.decoder.model.decoder.layers = nn.ModuleList([MyTrOCRDecoderLayer(my_config) for _ in range(my_config.decoder_layers)])
        return model

    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> # training
        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> text = "hello world"
        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(pixel_values=pixel_values, labels=labels)
        >>> loss = outputs.loss

        >>> # inference (generation)
        >>> generated_ids = model.generate(pixel_values)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        


def getAdaptorModel(use_config=False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model = MyVisionEncoderDecoderAdaptorModel.from_pretrained("microsoft/trocr-small-handwritten")
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
