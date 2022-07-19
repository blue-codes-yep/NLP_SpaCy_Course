import spacy
from spacy import displacy
import numpy as np

nlp = spacy.blank('en')
nlp = spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_web_md")

with open("data/wiki_us.txt", "r") as f:
    text = f.read()

doc = nlp2(text)

# Adding certain types of pipes to the already established pipelines.
nlp.add_pipe("sentencizer")

nlp.analyze_pipes()

# Analyze what is included in the pipelines.
print(nlp.analyze_pipes())

# Displays a cool little graph/entites showing how the words, connect/meanings throughout the sentence.
'''
text = "Hi, my name is Blue, and this is pretty cool I like seeing cool stuff like this!"
doc2 = nlp(text)

displacy.serve(doc, style="ent")

'''


# Displays similarty matches, with the word vector model in spaCy.
'''
your_word = "person"

ms = nlp.vocab.vectors.most_similar(
    np.asarray([nlp.vocab.vectors[nlp.vocab.strings[your_word]]]), n=10)
words = [nlp.vocab.strings[w] for w in ms[0][0]]
distances = ms[2]
'''

# Compare the similarity between documents, i.e. could use a for loop to compare documents to see which documents are most similar.
'''
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")

print(doc1, "<->", doc2, doc1.similarity(doc2))

# We can also calculate the similarity between two given words. Similarity of tokens and spans
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))

'''
