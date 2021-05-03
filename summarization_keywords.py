!pip install transformers
!pip install torch
!pip install sentencepiece
!pip install sentence-transformers
!pip install tika

%%writefile summarize.py
from tika import parser
import sys
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = torch.device('cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def summarize(text,max_len): ## Maximum length of summarize text
  preprocess_text = text[:20000].strip().replace("\n","")
  preprocess_text = preprocess_text.strip().replace(",","")  ## remove commas
  tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt").to(device)

  # summmarize 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=30,
                                      max_length=max_len,
                                      early_stopping=True)

  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary

## Keyword extraction  
model_keywords = SentenceTransformer('distilbert-base-nli-mean-tokens')

def keywords(text, top_n, stop_words):
  keypharases = []
  for i in [(1,1), (2,2)]:
    count_uni = CountVectorizer(text, ngram_range=i, stop_words=stop_words)
    count_uni.fit([text])
    candidates = count_uni.get_feature_names()
    doc_embedding_uni = model_keywords.encode([text])
    candidate_embeddings_uni = model_keywords.encode(candidates)
    distance_uni = cosine_similarity(doc_embedding_uni, candidate_embeddings_uni)
    keywords_uni = [candidates[index] for index in distance_uni.argsort()[0][-10:]]
    keypharases.append(keywords_uni)

  listToStr = ' '.join([str(elem) for elem in keypharases])

  return listToStr

def main(text, max_len, top_n, stop_words, path_output_file):
  summarize_1 = summarize(text,max_len)
  keywords_1 = keywords(text, top_n, stop_words)
  f = open(path_output_file, "x")
  f = open(path_output_file, "a")
  f.write("The summary of the file is : " +summarize_1)
  f.write("\n\n\n These are the keywords which describes the file :"+keywords_1)
  f.close()

#if __name__ == main:
#  main()
