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

text = '''You’ve no doubt heard of food for thought, food for love, food for strength,
health food, healing food, soul food, brain food, and the like. For as long as
people have inhabited this planet, edibles have been imbued with all sorts
of attributes beyond satisfying hunger and sustaining life. And in many
cases, popular notions about the powers of various foods and beverages

have been documented by modern scientific investigations that have dem-
onstrated, for example, the soothing qualities of chicken soup for sufferers

of the common cold, and the antibiotic properties of garlic.
Then there are the newer discoveries not rooted in folklore, among
them the protection against cancer afforded by vegetables and fruits rich in
the carotenoid pigments and the cancer-blockers found in members of the
cabbage family; the cholesterol-lowering ability of apples, barley, beans,
garlic, and oats; the heart-saving qualities of fish and alcohol (in moderate
amounts), and the antidiabetic properties of foods rich in dietary fiber.
But while thinking of food as preventive or cure, it is important not

to lose sight of its basic values: to provide needed nutrients and a pleasur-
able eating experience while satisfying hunger and thirst.

In The New Complete Book of Food Carol Ann Rinzler has put it all
together, providing a handy, illuminating guide for all who shop, cook, and
eat. It is a “must have” for all those who want to get the very most out of
the foods they eat, as well as avoid some inevitable dietary and culinary
pitfalls. Ms. Rinzler tells you how to derive the maximum nutritive value
from the foods you buy and ingest, with handy tips on how to select,
store, prepare, and in some cases serve foods to preserve their inherent
worth and avoid their risks. For example, in preparing bean sprouts, you’ll
be cautioned to eat them within a few days of purchase and to cook them
minimally to get the most food value from this vitamin C-rich food. You’ll
appreciate the importance of variety and moderation in your diet when

you discover that broccoli, which possesses two cancer-preventing proper-
ties, also can inhibit thyroid hormone if consumed in excess.

You will also recognize that not all wholesome foods are good for
all folks. Sometimes a health condition will render a food unsuitable for
you. For example, beans might be restricted for those with gout and certain
greens may be limited for those who must stick to a low-sodium diet. Then
too, there are possible interactions—both adverse and advantageous—
between certain foods and nutrients or medications. For example, citrus
fruits are recommended accompaniments for iron-rich vegetables and meats

The New Complete Book of Food
since the vitamin C in the fruits enhances the absorption of iron. Those taking anticoagulant
medication are advised to avoid excessive amounts of green leafy vegetables since the vitamin
K in these foods may reduce the effectiveness of the drug.
You’ll learn what happens to foods when they are cooked at home or processed in
factories. Want to avoid olive-drab green vegetables? Steam them quickly or, better yet, cook
them in the microwave with a tiny bit of water to bypass the discoloring action of acids
on the green pigment chlorophyll. You’ll also get the full story on methods of preserving
milk—from freezing and drying to evaporating and ultrapasteurizing—that should relieve
any anxieties you may have about the safety and healthfulness of processed milk.
In short, this is a book no self-respecting eater should be without. It can serve as a
lifetime reference for all interested in a safe and wholesome diet.
'''
