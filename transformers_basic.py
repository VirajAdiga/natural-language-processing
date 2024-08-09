from transformers import pipeline


# Sentiment analysis
classifier = pipeline("sentiment-analysis")
sentiments = classifier(
    [
        "I am not feeling great now",
        "Food was good last Sunday"
    ]
)
print(sentiments)

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
classification = classifier("We have a good revenue forecast", candidate_labels=["education", "business", "sports"])
print(classification)

# Text generation using a specific model
generator = pipeline("text-generation", model="distilgpt2")
result = generator("We are not machines!", max_length=30, num_return_sequences=2)
print(result)

# Named entity recognition
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
named_entities = ner("Santosh works at Google, Warsaw")
print(named_entities)

# Text summarization
summarizer = pipeline("summarization")
summary = summarizer(
    "Cricket is a globally popular sport that originated in England and has since captivated millions of fans "
    "worldwide. It is a bat-and-ball game played between two teams of eleven players each on a circular or "
    "oval-shaped field. The game is governed by a set of rules known as the Laws of Cricket, maintained by the "
    "Marylebone Cricket Club (MCC). The objective of the game is to score more runs than the opposing team. A match "
    "is divided into innings, where one team bats and attempts to score runs, while the other team bowls and fields, "
    "aiming to restrict the runs and dismiss the batsmen. The primary tools of the game are the cricket bat, "
    "a flat wooden blade, and the cricket ball, a hard, leather-covered sphere. Cricket is played in various formats, "
    "including Test matches, One Day Internationals (ODIs), and Twenty20 (T20) games. Test matches are the longest "
    "format, lasting up to five days, while ODIs and T20s are limited-overs formats, with each team facing a set "
    "number of overs, 50 and 20 respectively. The sport is renowned for its rich history, strategic depth, "
    "and the unique blend of individual brilliance and team coordination. Iconic tournaments like the Cricket World "
    "Cup and the Indian Premier League (IPL) have further elevated the game's global appeal, making cricket not just "
    "a sport, but a cultural phenomenon.",
    max_length=50
)
print(summary)
