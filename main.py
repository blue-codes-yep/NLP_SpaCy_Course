import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

with open("data/wiki_us.txt", "r") as f:
    text = f.read()

doc = nlp(text)

# Displays a cool little graph showing how the words, connect throughout the sentence.
'''
text = "Hi, my name is Blue, and this is pretty cool I like seeing cool stuff like this!"
doc2 = nlp(text)


'''

# for ent in doc.ents:
#    print(ent.text, ent.label_)

displacy.serve(doc, style="ent")
