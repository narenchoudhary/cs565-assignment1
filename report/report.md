# CS565: Assignment-1

## Basic pre-processing: Segmentation, N-Gram Analysis, Collocation

### Part 1: Getting Started

**Q1** 
Download any freely available corpus or take any accessible corpus from your chosen tool. Explore available sentence segmenter from the tool and use that for sentence segmentation.

**Tool**: Python3 NLTK library (Version 3.2.2)  
**Corpus**: Macbeth by Shakespeare (Available in ``gutenberg`` collection of  NLTK library
**Sentence Segmenter**: NLTK includes implementation of Punkt sentence segmentation [Kiss & Strunk, 2006](http://www.mitpressjournals.org/doi/abs/10.1162/coli.2006.32.4.485#.WJCTQVcnPCI)  
**Sentence Segmentation**: Total number of sentences in corpus is 1907.

---

**Q2**
Create a dictionary after detecting words from all sentences.

**Assumption**: *word* here means any sequence of one or more characters.

Total number of words in corpus is 22211.
Total number of unique words in corpus is 4094.
---

**Q3**
Find all possible unigrams. For each unigram, calculate its frequency in the given corpus. Plot the frequency distribution.

**Assumption**: . A single character token (Eg. `,`, `.`, `\`, `:`) is not considered unigram calculation in the answer.

Total number of unigrams in corpus is 11740.

20 most common unigrams in the frequency distribution are:

| unigram  | count |
|------|-----|
| the  | 530 |
| and  | 375 |
| of   | 315 |
| to   | 310 |
| 'd   | 192 |
| is   | 192 |
| you  | 181 |
| not  | 175 |
| in   | 172 |
| And  | 170 |
| my   | 170 |
| that | 156 |
| it   | 138 |
| Macb | 137 |
| with | 134 |
| his  | 129 |
| 's   | 128 |
| be   | 124 |
| The  | 118 |
| haue | 117 |  

**Frequency Plot**:  

![frequency plot](plots/plot1_3.png)

---

**Q4**
Find all possible bigrams and calculate their frequencies. Plot the frequency distribution.

**Assumption**: . A single character token (Eg. `,`, `.`, `\`, `:`) is not considered unigram calculation in the answer.

Total number of bigrams in corpus is 17439.

20 most common bigrams in the frequency distribution are:

| bigram  | count |
|------|-----|
| ('of', 'the') | 24 |
| ('to', 'the') | 24 |
| ("'t", 'is') | 21 |
| ('my', 'Lord') | 20 |
| ('in', 'the') | 20 |
| ('can', 'not') | 18 |
| ('Thane', 'of') | 18 |
| ('ha', "'s") | 17 |
| ('and', 'the') | 17 |
| ('to', 'be') | 15 |
| ('Exeunt', 'Scena') | 15 |
| ('Enter', 'Macbeth') | 15 |
| ('no', 'more') | 15 |
| ('of', 'my') | 15 |
| ('do', "'s") | 15 |
| ('not', 'be') | 14 |
| ('of', 'Cawdor') | 14 |
| ('in', 'his') | 13 |
| ("'d", 'with') | 12 |
| ('from', 'the') | 12 |

**Frequency Plot**:
![frequency plot](plots/plot1_4.png)

---

**Q5**
Similarly find all trigrams possible and calculate their frequencies. Plot the frequency distribution.

Total number of trigrams in corpus is 17438.

20 most common trigrams in the frequency distribution are:

| trigram  | count |
|------|-----|
| ('Thane', 'of', 'Cawdor') | 13 |
| ('my', 'good', 'Lord') | 8 |
| ('Knock', 'Knock', 'Knock') | 5 |
| ('Who', "'s", 'there') | 5 |
| ('the', 'Thane', 'of') | 5 |
| ('What', "'s", 'the') | 5 |
| ('my', 'Lord', 'Macb') | 4 |
| ('can', 'not', 'be') | 4 |
| ('good', 'Lord', 'Macb') | 4 |
| ('Enter', 'Macbeth', 'Macb') | 4 |
| ('Exeunt', 'Scena', 'Secunda') | 4 |
| ('This', 'is', 'the') | 4 |
| ('to', 'Night', 'Lady') | 3 |
| ('And', 'to', 'be') | 3 |
| ('Exeunt', 'Scena', 'Quarta') | 3 |
| ('the', 'three', 'Witches') | 3 |
| ('Rosse', 'and', 'Angus') | 3 |
| ('trouble', 'Fire', 'burne') | 3 |
| ('bed', 'to', 'bed') | 3 |
| ('you', 'haue', 'done') | 3 |

**Frequency Plot**:
![frequency plot](plots/plot1_5.png)

---

**Q6**
Each group is expected to explore two tools (e.g. NLTK and Apache OpenNLP).


### Part 2: Few Basic Questions

**Q1**
How many (most frequent) words are required for 90% coverage of the selected corpus?

2322 unigrams are required to cover 90.0 % of total corpus.

---

**Q2**
How many (most frequent) bigrams are required for 80% coverage of the corpus?

11068 bigrams are required to cover 80.0 % of total corpus.

---

**Q3**

How many (most frequent) trigrams are required for 70% coverage of the corpus?

11913 trigrams are required to cover 70.0 % of total corpus.

---

**Q4**

Repeat the above after performing lemmatization.

The NLTK Lemmatization method is based on [WordNet](http://wordnet.princeton.edu/)’s built-in morphy function.
Following are the results of Q.1, Q.2, Q.3 on lemmatized text.

2235 unigrams are required to cover 90.0 % of total corpus.
11035 bigrams are required to cover 80.0 % of total corpus.
11911 trigrams are required to cover 70.0 % of total corpus.

---

**Q5**

Compare the statistics of the two cases, with and without lemmatization.

As expected, after lemmatization less tokens were needed to cover same percentage of corpus as compared to number of tokens required before lemmatization.

This makes sense becuase lemmatization reduces inflectional forms by reducing the word to it's base form.

---

**Q6**

Summarize the results in the report.

#### Count of n-grams:

| n-gram |count|
|---|---|
|unigram|17440|
|bigram|17439|
|trigram|17438|

#### Before lemmatization vs After lemmatization:

|n-gram|before lemmatization|after lemmatization|
|---|---|---|
|unigram (90% coverage)|2322|2235|
|bigram (80% coverage)|11068|11035|
|trigram (70% coverage)|11913|11911|

---

### Part 3 : Writing some of your basic codes and comparing with results obtained using tools

**Q1**

Repeat section 2 after implementing discussed heuristics in the class for sentence segmenter and
word tokenizer. If you want to improvise on the discussed heuristics, you can do that but you
should describe your heuristics in the report. Also summarize your findings by comparing the
results obtained using your heuristics and tools.

---

**Q2**

Implement Pearson’s Chi-Square Test for finding all bigram (contiguous) collocations in your chosen
corpus. Do not use libraries. Discuss if you have made any interesting observation.


Pearson's Chi-square test returns following 20 collocations as best candidates:  

Knock Knock  
Exeunt Scena  
three Witches  
Scena Secunda  
ha 's  
't is  
no more  
at once  
worthy Thane  
mine owne  
my Lord  
thou art  
your Highnesse  
'T is  
can not  
Enter Macbeth  
Giue me  
good Lord  
Thane of  
do 's  


---

