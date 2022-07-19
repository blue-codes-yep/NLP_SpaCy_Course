import spacy
from spacy import displacy
import pandas as pd

# Working with spaCy to do Fincial NER, along with Pandas to read the csv/tsv files.

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("data/stocks.tsv", sep="\t")
df2 = pd.read_csv("data/indexes.tsv", sep="\t")
df3 = pd.read_csv("data/stock_exchanges.tsv", sep="\t")


symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()
print(symbols[:10])
indexes = df2.IndexName.tolist()
index_symbols = df2.IndexSymbol.tolist()
exchanges = df3.ISOMIC.tolist()+df3["Google Prefix"].tolist()
descriptions = df3.Description.tolist()

# two was also showing as a company, which was a "false postive".
stops = ["two"]
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
patterns = []
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# List of Entities and Patterns
for symbol in symbols:
    patterns.append({"label": "STOCK", "pattern": symbol})
    for l in letters:
        patterns.append({"label": "STOCK", "pattern": symbol+f".{l}"})


for company in companies:
    if company not in stops:
        patterns.append({"label": "COMPANY", "pattern": company})
        words = company.split()
        # Resolution for certain companies not being noticed with spaces.
        if len(words) > 1:
            new = " ".join(words[:2])
            patterns.append({"label": "COMPANY", "pattern": new})

for index in indexes:
    patterns.append({"label": "INDEX", "pattern": index})
    versions = []
    words = index.split()
    caps = []
    for word in words:
        word = word.lower().capitalize()
        caps.append(word)
    versions.append(" ".join(caps))
    versions.append(words[0])
    versions.append(caps[0])
    versions.append(" ".join(caps[:2]))
    versions.append(" ".join(words[:2]))
    for version in versions:
        if version != "NYSE":
            patterns.append({"label": "INDEX", "pattern": version})

for symbol in index_symbols:
    patterns.append({"label": "INDEX", "pattern": symbol})

for d in descriptions:
    patterns.append({"label": "STOCK_EXCHANGE", "pattern": d})
for e in exchanges:
    patterns.append({"label": "STOCK_EXCHANGE", "pattern": e})


ruler.add_patterns(patterns)


print(len(patterns))

# source: https://www.reuters.com/business/futures-rise-after-biden-xi-call-oil-bounce-2021-09-10/
text = '''
Sept 10 (Reuters) - Wall Street's main indexes were subdued on Friday as signs of higher inflation and a drop in Apple shares following an unfavorable court ruling offset expectations of an easing in U.S.-China tensions.

Data earlier in the day showed U.S. producer prices rose solidly in August, leading to the biggest annual gain in nearly 11 years and indicating that high inflation was likely to persist as the pandemic pressures supply chains. read more .

"Today's data on wholesale prices should be eye-opening for the Federal Reserve, as inflation pressures still don't appear to be easing and will likely continue to be felt by the consumer in the coming months," said Charlie Ripley, senior investment strategist for Allianz Investment Management.

Apple Inc (AAPL.O) fell 2.7% following a U.S. court ruling in "Fortnite" creator Epic Games' antitrust lawsuit that stroke down some of the iPhone maker's restrictions on how developers can collect payments in apps.


Sponsored by Advertising Partner
Sponsored Video
Watch to learn more
Report ad
Apple shares were set for their worst single-day fall since May this year, weighing on the Nasdaq (.IXIC) and the S&P 500 technology sub-index (.SPLRCT), which fell 0.1%.

Sentiment also took a hit from Cleveland Federal Reserve Bank President Loretta Mester's comments that she would still like the central bank to begin tapering asset purchases this year despite the weak August jobs report. read more

Investors have paid keen attention to the labor market and data hinting towards higher inflation recently for hints on a timeline for the Federal Reserve to begin tapering its massive bond-buying program.

The S&P 500 has risen around 19% so far this year on support from dovish central bank policies and re-opening optimism, but concerns over rising coronavirus infections and accelerating inflation have lately stalled its advance.


Report ad
The three main U.S. indexes got some support on Friday from news of a phone call between U.S. President Joe Biden and Chinese leader Xi Jinping that was taken as a positive sign which could bring a thaw in ties between the world's two most important trading partners.

At 1:01 p.m. ET, the Dow Jones Industrial Average (.DJI) was up 12.24 points, or 0.04%, at 34,891.62, the S&P 500 (.SPX) was up 2.83 points, or 0.06%, at 4,496.11, and the Nasdaq Composite (.IXIC) was up 12.85 points, or 0.08%, at 15,261.11.

Six of the eleven S&P 500 sub-indexes gained, with energy (.SPNY), materials (.SPLRCM) and consumer discretionary stocks (.SPLRCD) rising the most.

U.S.-listed Chinese e-commerce companies Alibaba and JD.com , music streaming company Tencent Music (TME.N) and electric car maker Nio Inc (NIO.N) all gained between 0.7% and 1.4%


Report ad
Grocer Kroger Co (KR.N) dropped 7.1% after it said global supply chain disruptions, freight costs, discounts and wastage would hit its profit margins.

Advancing issues outnumbered decliners by a 1.12-to-1 ratio on the NYSE and by a 1.02-to-1 ratio on the Nasdaq.

The S&P index recorded 14 new 52-week highs and three new lows, while the Nasdaq recorded 49 new highs and 38 new lows.
'''
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

'''
# Using regex inside of spaCy.
text = "Paul Newman was an American actor, but Paul Hollywood is a British TV Host. The name Paul is quite common."


nlp = spacy.blank("en")
doc = nlp(text)

# If you want to use regex across multiple tokens, implement it like this.


@Language.component("paul_ner")
def paul_ner(doc):
    # Find any instance of Paul that is a capital letter, and followed by a word-break.
    pattern = r"Paul [A-Z]\w+"

    originial_ents = list(doc.ents)
    # Multi word token entity list.
    mwt_ents = []
    for match in re.finditer(pattern, doc.text):
        start, end = match.span()
        # Doing this because regex gives your character spans, because the doc object works on a token level.
        span = doc.char_span(start, end)
        if span is not None:
            # Append the span, and text from the regex match object to our empty list.
            mwt_ents.append((span.start, span.end, span.text))

    # Inject them into the doc object.
    for ent in mwt_ents:
        start, end, name = ent
        per_ent = Span(doc, start, end, label="PERSON")
        originial_ents.append(per_ent)

    doc.ents = originial_ents
    return (doc)


nlp2 = spacy.blank("en")
nlp2.add_pipe("paul_ner")
doc2 = nlp2(text)
print(doc2.ents)
'''


'''

# Return objects, that match with our pattern.
matches = re.finditer(pattern, text)

# Loop over our matches to print out the objects.
for match in matches:
    print(match)
'''


'''
# Removing certain entities with a custom pipe.


@Language.component("remove_GPE")
def remove_gpe(doc):
    orginial_ents = list(doc.ents)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            orginial_ents.remove(ent)
    doc.ents = orginial_ents
    return (doc)


nlp.add_pipe("remove_GPE")
nlp.analyze_pipes()

doc = nlp("Britain is a place. John is a doctor.")

for ent in doc.ents:
    print(ent.text, ent.label_)
'''


'''
# Working with matchers too get quotes from Alice, in the first few chapters of alice in the wonderland. 


nlp = spacy.load("en_core_web_sm")

with open("data/alice.json", "r") as f:
    data = json.load(f)

A lemma is a dictionary form, or citation form of a set of words. In English, for example, break, breaks, broke, broken and breaking are forms of the same lexeme,
with break as the lemma by which they are indexed.

speak_lemmas = ["think", "say"]
text = data[0][2][0].replace("`", "'")
matcher = Matcher(nlp.vocab)
# Pattern formats : https://spacy.io/usage/rule-based-matching
pattern1 = [{'ORTH': "'"}, {'IS_ALPHA': True, "OP": "+"}, {'IS_PUNCT': True, "OP": "*"}, {'ORTH': "'"}, {"POS": "VERB", "LEMMA": {"IN": speak_lemmas}},
            {"POS": "PROPN", "OP": "+"}, {'ORTH': "'"}, {'IS_ALPHA': True, "OP": "+"}, {'IS_PUNCT': True, "OP": "*"}, {'ORTH': "'"}]
pattern2 = [{'ORTH': "'"}, {'IS_ALPHA': True, "OP": "+"}, {'IS_PUNCT': True, "OP": "*"},
            {'ORTH': "'"}, {"POS": "VERB", "LEMMA": {"IN": speak_lemmas}}, {"POS": "PROPN", "OP": "+"}]
pattern3 = [{"POS": "PROPN", "OP": "+"}, {"POS": "VERB", "LEMMA": {"IN": speak_lemmas}},
            {'ORTH': "'"}, {'IS_ALPHA': True, "OP": "+"}, {'IS_PUNCT': True, "OP": "*"}, {'ORTH': "'"}]
matcher.add("PROPER_NOUNS", [pattern1, pattern2, pattern3], greedy='LONGEST')

doc = nlp(text)
matches = matcher(doc)
matches.sort(key=lambda x: x[1])
print(len(matches))
# Looping over the first few chapters of alice in the wonderland, to find quotes from the "think", and "say" lemmas.
for text in data[0][2]:
    text = text.replace("`", "'")
    doc = nlp(text)
    matches = matcher(doc)
    matches.sort(key=lambda x: x[1])
    print(len(matches))
    for match in matches[:10]:
        print(match, doc[match[1]:match[2]])
'''


'''
nlp = spacy.load("en_core_web_sm")

with open("data/wiki_mlk.txt", "r") as f:
    text = f.read()

matcher = Matcher(nlp.vocab)

# Look for Proper Nouns, that occurs (OP +), or more times, as well as part of speech that is a verb.
pattern = [{"POS": "PROPN", "OP": "+"}, {"POS": "VERB"}]
matcher.add("PROPER_NOUN", [pattern], greedy="LONGEST")
doc = nlp(text)
matches = matcher(doc)

# lambda function to sort the matches from the start of the document.
matches.sort(key=lambda x: x[1])
print(len(matches))
for match in matches[:10]:
    print(match, doc[match[1]:match[2]])
'''


'''
Basic way of using matcher, to find email addresses.


matcher = Matcher(nlp.vocab)
pattern = [{
    "LIKE_EMAIL": True
}]
matcher.add("EMAIL_ADDRESS", [pattern])

doc = nlp("This is an email address: johndoe@gmail.com")
matches = matcher(doc)
print(nlp.vocab[matches[0][0]].text)
'''


'''
Adding entity_ruler pipelines, use entity_rulers whenever the things you are extracting are important to have label that cooresponds with it.
Use a matcher whenever something that is a structure within the text to help you extract information. 
Use regex, whenever it's a complciated pattern that isn't dependent on specific parts of speech. 

nlp = spacy.load("en_core_web_sm")
text = "Joe Rogan, and Theo Von are both comedians, and have a podcast. The Joe Rogan Experience, and This Past Weekend are the names of the podcasts."

doc = nlp(text)

ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "ORG", "pattern": "The Joe Rogan Experience"},
    {"label": "ORG", "pattern": "This Past Weekend"}
]

ruler.add_patterns(patterns)

doc2 = nlp(text)
for ent in doc2.ents:
    print(ent.text, ent.label_)
    
Output before adding ner entity_ruler pipeline: 
Joe Rogan PERSON
Theo Von PERSON
The Joe Rogan PERSON
This Past Weekend DATE
--------- 
Output after adding pipeline: 
Joe Rogan PERSON
Theo Von PERSON
The Joe Rogan Experience ORG
This Past Weekend ORG
'''


'''
# Read from document with python.
with open("data/wiki_us.txt", "r") as f:
    text = f.read()


#Assign document that was read from to nlp2 which is then able to be used by spaCy.
doc = nlp2(text)

# Adding certain types of pipes to the already established pipelines.
nlp.add_pipe("sentencizer")

# Analyze what is included in the pipelines.
nlp.analyze_pipes()

'''


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
