## Text-summarization

# Installations:
  !pip install transformers
  !pip install torch
  !pip install sentencepiece
  !pip install sentence-transformers
  !pip install tika


# Call main function from summarize
  summarize.main(text=text, max_len=200, top_n=5, stop_words='english')


# Parameters :
  text = string data
  max_len = length of summary needed
  top_n = number of keywords needed
  stop_words = 'english'
