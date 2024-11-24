# -*- coding: utf-8 -*-
rouge score table
"""

!pip install pandas matplotlib tabulate datasets transformers rouge-score kaggle kagglehub

import kagglehub

path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
print("Path to dataset files:", path)

import os
import pandas as pd

directory_path = "/root/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2/cnn_dailymail"
files = os.listdir(directory_path)
print(files)

train_df = pd.read_csv(os.path.join(directory_path, 'train.csv'))

print("Training Data:")
print(train_df.head())

from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    GPTNeoForCausalLM
)


models = {
    'T5': (T5ForConditionalGeneration.from_pretrained('t5-small'), T5Tokenizer.from_pretrained('t5-small')),
    'BART': (BartForConditionalGeneration.from_pretrained('facebook/bart-base'), BartTokenizer.from_pretrained('facebook/bart-base')),
    'Pegasus': (PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum'), PegasusTokenizer.from_pretrained('google/pegasus-xsum')),
    'GPT-2': (GPT2LMHeadModel.from_pretrained('gpt2'), GPT2Tokenizer.from_pretrained('gpt2')),
    'GPT-Neo': (GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B'), GPT2Tokenizer.from_pretrained('gpt2')),
}

!pip install rouge-score



!pip install pandas matplotlib tabulate datasets transformers rouge-score kaggle kagglehub

import kagglehub
import os
import pandas as pd
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
from rouge_score import rouge_scorer

path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
print("Path to dataset files:", path)


directory_path = "/root/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2/cnn_dailymail"
files = os.listdir(directory_path)
print("Files in dataset directory:", files)

train_df = pd.read_csv(os.path.join(directory_path, 'train.csv'))


print("Training Data:")
print(train_df.head())


models = {
    'T5': (T5ForConditionalGeneration.from_pretrained('t5-small'), T5Tokenizer.from_pretrained('t5-small')),
    'BART': (BartForConditionalGeneration.from_pretrained('facebook/bart-base'), BartTokenizer.from_pretrained('facebook/bart-base')),
    'Pegasus': (PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum'), PegasusTokenizer.from_pretrained('google/pegasus-xsum')),
    'Primera': (AutoModelForSeq2SeqLM.from_pretrained('allenai/primera'), AutoTokenizer.from_pretrained('allenai/primera')),
    'LongT5': (AutoModelForSeq2SeqLM.from_pretrained('google/long-t5-tglobal-base'), AutoTokenizer.from_pretrained('google/long-t5-tglobal-base'))
}

def calculate_rouge(model_name, model, tokenizer, df, num_samples=10):
    """Calculates ROUGE scores for a given model.
    Args:
        model_name: The name of the model (e.g., 'T5', 'BART').
        model: The model object.
        tokenizer: The tokenizer object.
        df: The DataFrame containing the articles and summaries.
        num_samples: The number of samples to evaluate.
    Returns:
        A dictionary containing the average ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for i in range(num_samples):
        article = df['article'][i]
        reference = df['highlights'][i]

        inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        summary_ids = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


        scores = scorer.score(reference, summary)
        for metric in rouge_scores.keys():
            rouge_scores[metric].append(scores[metric].fmeasure)


    avg_rouge_scores = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    return avg_rouge_scores

for model_name, (model, tokenizer) in models.items():
    print(f"Calculating ROUGE scores for {model_name}...")
    rouge_scores = calculate_rouge(model_name, model, tokenizer, train_df)
    print(f"{model_name} ROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.4f}")

