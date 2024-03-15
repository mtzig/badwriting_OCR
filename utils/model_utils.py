import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
from datasets import load_metric
from transformers import VisionEncoderDecoderModel


cer_metric = load_metric("cer")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

def getModel(device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

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

def get_dataloaders(dataset_type='t', test_size=0.2, batch_size=4, root=''):

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

        train_dataset = IAMDataset(root_dir=f'{root}/data/alex_writing/',
                                df=train_df,
                                processor=processor)
        eval_dataset = IAMDataset(root_dir=f'{root}/data/alex_writing/',
                                df=test_df,
                                processor=processor)
    elif dataset_type=='iam':
        # if we are using the iam dataset, need to use the optinoal root param to point to where it is
        df = pd.read_csv(root+'/iam_data.csv') # root should be iam_path
        train_df, test_df = train_test_split(df, test_size=test_size)
        # we reset the indices to start from zero
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = IAMDataset(root_dir=root+'/sentences',
                                df=train_df,
                                processor=processor)
        eval_dataset = IAMDataset(root_dir=root+'/sentences', # sentences or lines ?
                                df=test_df,
                                processor=processor)
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
