# Task : Training a character-level GPT on the works of Shakespeare data from scratch

#  Contributors :

## Gosula Sunandini 
github repository : https://github.com/sunandhini96
## Katipally Vigneshwar Reddy
github repository : https://github.com/katipallyvig8899



# Files Required:
1. data : shakesphere data
2. model.py : model code
3. train.py : training the model code
4. config : configguration file of shakesphere data
   
## Data preparation:

-> download single (1MB) file and turn it from raw text into one large stream of integers(Tokenize the text into characters. In character-level models, each character becomes a token and data encoding ) 

` python data/shakespeare_char/prepare.py `

This creates a train.bin and val.bin in that data directory.

## Train the model:

To train the model run the following command:

` python train.py config/train_shakespeare_char.py `

## Sample data generation:

To get sample data run the following command:

`  python sample.py --out_dir=out-shakespeare-char `

# Hugging face application link :  

https://huggingface.co/spaces/Gosula/Nano_gpt_Shakespeare_data

