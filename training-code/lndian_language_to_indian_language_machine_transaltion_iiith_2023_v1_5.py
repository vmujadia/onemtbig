# -*- coding: utf-8 -*-
"""Lndian Language to Indian Language Machine Transaltion-IIITH-2023-v1.5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jHXH6lBQZTkptC3MMb1O880draZL3dfy

# ** Hybrid Neural Machine Translation for HimangiY **
#### Vandan Mujadia, Dipti Misra Sharma
#### LTRC, IIIT-Hyderabad, Hyderabad

This demonstrates how to train a sequence-to-sequence (seq2seq) model for Kannada-to-Hindi translation **roughly** based on [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1706.03762) (Vaswani, Ashish et al).

## An Example to Understand sequence to Sequence processing using Transformar Network.

<img src="https://www.tensorflow.org/images/tutorials/transformer/apply_the_transformer_to_machine_translation.gif" alt="Applying the Transformer to machine translation">

Source: [Google AI Blog](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)

## Applying the Transformer to machine translation.


<table>
<tr>
  <td>
   <img width=400 src="https://miro.medium.com/max/720/1*57LYNxwBGcCFFhkOCSnJ3g.png"/>
  </td>
</tr>
<tr>
  <th colspan=1>This tutorial: An encoder/decoder connected by self attention neural network.</th>
<tr>
</table>

# Tools that we are using here

*   Library : Opennmt
*   Library : pytorch based neural network implemtation
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install -U pip
!!git clone https://github.com/OpenNMT/OpenNMT-py
! ls
# %cd OpenNMT-py
!git checkout 1.2.0
!pip3 install torchtext==0.4.0 torch==1.11.0

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/

"""# Check GPU"""

!nvidia-smi

"""# Tokenizer Tool"""

!pip install git+https://github.com/vmujadia/tokenizer.git --upgrade

"""# To Clean and Filter Parallel Corpora"""

!git clone https://github.com/moses-smt/mosesdecoder.git

"""# To tackle vocabulary issue : Subword algorithm"""

!git clone https://github.com/rsennrich/subword-nmt.git

!ls mosesdecoder/scripts/training/clean-corpus-n.perl

"""# For this; Training Corpora

##  Kannada - Hindi
## (small courpus MIT+CDAC-B developed)
"""

! wget -O train.src https://ssmt.iiit.ac.in/uploads/data_mining/kannada-hindi_combined_all.hi
! wget -O train.tgt https://ssmt.iiit.ac.in/uploads/data_mining/kannada-hindi_combined_all.kn
! wget -O valid.src https://ssmt.iiit.ac.in/uploads/data_mining/flores200-dev.hi
! wget -O valid.tgt https://ssmt.iiit.ac.in/uploads/data_mining/flores200-dev.kn

"""# Data Numbers"""

print ('Data Stats')
! wc -l train.*
! wc -l valid.*

"""# Tokenize the text"""

from ilstokenizer import tokenizer
import codecs

def to_tokenize_and_lower(input_path, output_path):
  outfile = open(output_path, 'w')
  for line in codecs.open(input_path):
    line = line.strip()
    line = tokenizer.tokenize(line).lower()
    #print (line)
    outfile.write(line+'\n')
  outfile.close()

to_tokenize_and_lower('train.src','train.src.tkn')
to_tokenize_and_lower('train.tgt','train.tgt.tkn')

to_tokenize_and_lower('valid.src','valid.src.tkn')
to_tokenize_and_lower('valid.tgt','valid.tgt.tkn')

! cat train.src.tkn > train.all.tkn
! cat train.tgt.tkn >> train.all.tkn

"""# Data Cleaning"""

! perl mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 2.5 train src.tkn tgt.tkn train_filtered 1 250

print ('Data Stats')
! wc -l train*
! wc -l valid*

print ('Data Stats')
! wc -l train*
! wc -l valid*

"""# Train subword model,
## Experiment with no of subword merge operation
"""

!python subword-nmt/subword_nmt/learn_bpe.py -s 5000 < train.all.tkn > train.codes

"""# How do subword codes look"""

! head -n 10 train.codes

"""# Apply Subword to the corpus"""

!python subword-nmt/subword_nmt/apply_bpe.py -c train.codes < train.src > train.kn
!python subword-nmt/subword_nmt/apply_bpe.py -c train.codes < train.tgt > train.hi

!python subword-nmt/subword_nmt/apply_bpe.py -c train.codes < valid.src > valid.kn
!python subword-nmt/subword_nmt/apply_bpe.py -c train.codes < valid.tgt > valid.hi

"""# Training Corpus now"""

! head -n 10 train.kn

! head -n 10 train.hi

"""
# Starting  NMT Training
## Preprocessing stage ; create dictionaries, make corpora ready for parallel processing
"""

!pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 configargparse

!python OpenNMT-py/preprocess.py \
	    -train_src train.kn \
	    -train_tgt train.hi \
	    -valid_src valid.kn \
	    -valid_tgt valid.hi \
	    -save_data processed -share_vocab -overwrite

ls data-bin/trial

"""# Training
## Parameters to fix for your corpora and language pair



```
    --encoder-embed-dim	128 --encoder-ffn-embed-dim	128 \
    --encoder-layers	2 --encoder-attention-heads	2 \
    --decoder-embed-dim	128 --decoder-ffn-embed-dim	128 \
    --decoder-layers	2 --decoder-attention-heads	2 \
    --dropout 0.3 --weight-decay 0.0 \
    --max-update 4000 \
    --keep-last-epochs	10 \
```



---


"""

! python OpenNMT-py/train.py -data processed -save_model model.pt \
		-layers 6 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -transformer_ff 2048 -heads 8  \
		-encoder_type transformer -decoder_type transformer -position_encoding \
		-train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
		-batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
		-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
		-max_grad_norm 0 -param_init 0  -param_init_glorot \
		-label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
		-world_size 1 -gpu_ranks 0