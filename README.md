# Recognizing Illegible Handwriting
Final Project for CS 766

Alex Clinton, Zhuoming Liu, Khoi Nguyen, Thomas Zeng

## Dataset
The Alex and Thomas datasets in the `data/alex_writing/` and `data/thomas_writing/`
For IAM dataset, to download the IAM dataset view `readme.md ` in the `data/IAM/sentences` directory.

## Prepare the enviroment
The Jupyter notebooks have included all the installation commands of the needed environment, please run the cells step by step to set up the environment.

## The Demo
In here we provide a demo for showing the final OCR results: https://colab.research.google.com/drive/1g81RcersBBUUttUm174K5HDO2s8h07IA?usp=sharing


## The Checkpoints
We provide all different checkpoints in google drive: https://drive.google.com/drive/folders/18SMtQZ_nisR8LqpdL2gLEC9h2VkCOXSe?usp=sharing


## The important files in this repo
Now we list some important Jupyter notebooks and what we used them for:
- `MAML.ipynb`: produces checkpoint of TrOCR trained on MAML.
- `Fine_tune_TrOCR_dual_decoder.ipynb`: finetunes TrOCR model using dual decoder method.
- `Fine_tune_TrOCR_adaptor.ipynb`: finetunes TrOCR model using adaptor method.
- `Fine_tune_TrOCR_from_MAML.ipynb`: finetunes TrOCR model using MAML checkpoint as starting point.
- `Fine_tune_TrOCR_partial_params.ipynb`: finetunes TrOCR model with different model parameters.
