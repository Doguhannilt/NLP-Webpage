import streamlit as st
import pandas as pd
from PIL import Image
import pygame

st.title("NLP Introduction")
st.info("The webpage is for beginners.")
# Initialize pygame
pygame.init()

# Function to play music
def play_music():
    pygame.mixer.music.load("music.mp3")
    pygame.mixer.music.play()

# Function to stop music
def stop_music():
    pygame.mixer.music.stop()

# İki sütun oluştur
col1, col2 = st.columns(2)

# İlk sütun için "Play Music" butonu
with col1:
    if st.button("Play Music"):
        play_music()

# İkinci sütun için "Stop Music" butonu
with col2:
    if st.button("Stop Music"):
        stop_music()
st.subheader("What is NLP")

st.write("""
NLP is an interdisciplinary field that is commonly used in text-based projects. The field of natural language processing began in the 1940s, after World War II. At this time, people recognized the importance of translation from one language to another and hoped to create a machine that could do this sort of translation automatically.

In this documentation, you may be able to understand the core concepts of Natural Language Processing.""")


image_1 = Image.open('Images/1.png')
st.image(image_1, caption='NLP Introduction')

st.subheader("Common NLP Tasks and Applications")
lst = ['Translation: One of the basic application in NLP', 'Summarization: Summarizing a long "corpus" into a few paragraphs', "Question Answering: Today's chatbots are good examples of Question-Answering.",'Speech Recognition: Getting logical information from your speech', 'Classification: Classify the "things" by using Artifical Intelligence']
for i in lst:
    st.markdown("- " + i)

st.subheader("History of NLP")
st.write("The history of NLP (Natural Language Processing) and the development of chatbots like GPT (Generative Pre-trained Transformer) is quite fascinating. Here's a brief overview:")

history_of_nlp = [
"**Early NLP Research:** NLP has its roots in the 1950s and 1960s with early research focusing on rule-based systems for language translation and information retrieval."
"**Statistical NLP:** In the 1990s, statistical methods gained prominence in NLP. Researchers started using machine learning techniques to process and understand natural language. This era saw the emergence of probabilistic models like Hidden Markov Models.",
"**Deep Learning Revolution:** Around 2010, deep learning techniques, particularly neural networks, began to dominate NLP. This marked a significant shift in the field's progress.",
"**GPT-1 (2018):** OpenAI introduced the first version of the Generative Pre-trained Transformer, GPT-1. It demonstrated impressive language generation capabilities but had limitations.",
"**GPT-2 (2019):** GPT-2 made headlines due to its remarkable text generation capabilities. OpenAI initially withheld the full model due to concerns about potential misuse but later made it available to the public.",
"**GPT-3 (2020):** GPT-3, the model I'm based on, is one of the most prominent developments in NLP. It's a massive neural network with 175 billion parameters and can perform various natural language understanding and generation tasks. It's been used in a wide range of applications, including chatbots, content generation, and more.",
"**GPT-4 (TBA):** As of my last knowledge update in September 2021, there was no official information about GPT-4, but it's likely that advancements in NLP have continued, and newer models may have been developed.",
"**Applications:** NLP and chatbots like GPT have found applications in customer support, content generation, language translation, medical diagnosis, and more. They've also raised ethical and privacy concerns, leading to discussions about responsible AI development and usage."]
for i in history_of_nlp:
    st.markdown("- " + i)
st.divider()
st.subheader("NLP libraries")
nlp_libraries = [
    {
        'Name': 'NLTK',
        'Main Features': [
            'Comprehensive NLP library for Python.',
            'Includes a wide range of text processing libraries and modules.',
            'Supports tokenization, stemming, lemmatization, part-of-speech tagging, and more.',
            'Suitable for educational purposes and small-scale NLP tasks.',
        ],
        'Use Cases': [
            'Text preprocessing and cleaning.',
            'Teaching and learning NLP concepts.',
            'Small-scale NLP projects.',
        ],
    },
    {
        'Name': 'spaCy',
        'Main Features': [
            'Fast and efficient NLP library designed for production use.',
            'Provides pre-trained models for various languages.',
            'Supports tokenization, part-of-speech tagging, named entity recognition, and more.',
            'Rich ecosystem of extensions and custom components.',
        ],
        'Use Cases': [
            'Large-scale text processing in production environments.',
            'Information extraction, named entity recognition, and text classification.',
            'Building custom NLP pipelines.',
        ],
    },
    {
        'Name': 'Gensim',
        'Main Features': [
            'Focused on topic modeling and document similarity analysis.',
            'Implements Word2Vec, Doc2Vec, and other embedding algorithms.',
            'Useful for training word embeddings and working with large text corpora.',
            'Notable for its simplicity and scalability.',
        ],
        'Use Cases': [
            'Topic modeling, document clustering, and similarity analysis.',
            'Building word embeddings for various NLP tasks.',
            'Scalable and distributed text processing.',
        ],
    },
    {
        'Name': 'fastText',
        'Main Features': [
            'Library developed by Facebook AI for text classification and word embeddings.',
            'Supports fast and efficient text classification with pre-trained models.',
            'Enables training custom word embeddings with subword information.',
            'Suitable for multilingual and large-scale text classification tasks.',
        ],
        'Use Cases': [
            'Text classification, sentiment analysis, and text categorization.',
            'Building word embeddings that consider subword information.',
            'Multilingual and cross-lingual NLP tasks.',
        ],
    },
]

# Convert the list of dictionaries to a DataFrame
library = pd.DataFrame(nlp_libraries)

# Join the lists in 'Main Features' and 'Use Cases' into single strings
library['Main Features'] = library['Main Features'].apply(lambda x: "\n".join(x))
library['Use Cases'] = library['Use Cases'].apply(lambda x: "\n".join(x))
st.table(library)

st.divider()
#P2

st.header("Tokenization")
st.subheader("What is Tokenization")
image_2 = Image.open('Images/2.jpg')
st.image(image_2, caption='Tokenization is get used to create small units from a whole piece',width = 400)
st.code("AI is not able to understand our language if it is not numerical. So, make AI's job easier, we divide the sentences or words into smaller units.")
st.write("Tokens are the basic units of language that machin learning models can understand the whole process.")
st.write("Tokenization is a crucial step in NLP and helps convert human language into a format that compurter can work with.")
st.write("Tokenization can vary in complexity, depending on the language and the specific NLP task. For example, in English, tokenization is often done at the word level, but it can also involve splitting words into smaller units (subword tokenization), especially for languages with complex morphology or for tasks like machine translation.")
st.divider()
st.subheader("Tokenizing English")
st.write("Let's say we have a sentence and we want to make tokenization")
st.write("Sentence: **How dare you to do that? Don't you see I am wearing my clothes**")
st.write("After tokenization:")
lst = ["How","dare","you","to","do","that?","Don't","you","see","I","am","wearing","my","clothes"]
st.write(lst)
st.write("As you see, we created a bunch of units from a sentence")
st.divider()
st.subheader("Tokenizing Other Languages")
st.write("What if we have a word such as **没有迪约森**, can we use tokenization technique that seperate the sentences by using spaces between words?")
st.markdown("So, in some cases, we should use other tokenization techniques. Here are the examples of the other tokenization techniques")
tokenization_lst = ["N-gram Tokenization: This method generates n-grams, contiguous sequences of n items (words or characters) from the text.","Regular Expression Tokenization: This technique uses regular expressions to define custom rules for tokenization.","Penn Treebank Tokenization: This method is based on the Penn Treebank Project's tokenization rules, which consider various factors like contractions, hyphenated words, and special characters to tokenize text.","Byte Pair Encoding (BPE) Tokenization: This method tokenizes text based on the frequency of character pairs, merging the most frequent pairs iteratively."]

for i in tokenization_lst:
    st.markdown("- " + i)
    
#P3
st.divider()
st.header("Basic Preprocessing Techniques")
st.write("To use sentences for the model, we should prepare our corpus by applying **Case Folding**, **Stop Word Removing**, **Stemming**, **Lemmatization**")
st.info("Vocabulary: The set of all unique tokens in a corpus")
st.divider()

st.subheader("Case Folding")
st.write("Case folding is a process applied to a sequence of characters, in which those identified as non-uppercase are replaced by their uppercase equivalents.")
st.write("To understand more properly, let's make a good example by using a sentence from my one of favorite  characters in Naruto:")
st.write("**Sentence:** The power of the Uchiha clan is unmatched, and we will bring about a new era. What if they are loser, we are going to destroy all of them.")
st.write("**After Case Folding:** the power of the uchiha clan is unmatched, and we will bring about a new era. what if they are loser, we are going to destroy all of them. ")
st.divider()

st.subheader("Stop Word Removal")
st.write("It is one of the most commonly used preprocessing steps across different NLP applications. The idea is simply removing the words that occur commonly across all the documents in the corpus. Typically, articles and pronouns are generally classified as stop words.")
st.info("Most of time, the libraries of Python are going to give you packaged Stop Word Removal list but you can create your own custom Stop Word Removal list")
removal_list = ["I","me","my","myself","we","our","ours","ourselves","you","your","yours"]
st.write("Let's create a custom Stop Word Removal list and apply it into a sentence")
st.write("Our custom Stop Word list:")
st.write(removal_list)
st.write(" ")
st.write("**Sentence:** 'You and I are the best friends! I can't do anything by myself.'")
st.write("After applying our custom stop word list:" )
st.write("**Stopword_Sentence:** 'and are the best friends! can't do anything by.'")
st.write("As you see, we removed some words from the sentence that are in our list")
st.divider()

st.subheader("Stemming and Lemmatization")
st.write("Stemming and Lemmatization are another preprocessing techniques that we use for our sentences. In general, the main goal of both of them are is to identify the roots of words but there is a few difference between both")
st.write("The major difference is Stemming is a process that stems or removes last few characters from a word, often leading to incorrect meanings and spelling. Lemmatization considers the context and converts the word to its meaningful base form, which is called Lemma.")

stemming = [
    "Stemming is a text normalization technique that aims to remove suffixes (and sometimes prefixes) from words to get to the root or base form. The resulting form may not always be a valid word.",
    'It is a rule-based approach that chops off common word endings, so words like "running" and "runner" would both be reduced to "run."',
    "Stemming is faster than lemmatization because it uses simple rules and heuristics to process words.",
]

lemm = [
    "Lemmatization is a more advanced and linguistically informed approach. It reduces words to their base or dictionary form, known as the lemma.",
    "The resulting form is usually a valid word that can be found in a dictionary.",
    'Lemmatization takes into account the context of the word in a sentence and applies morphological analysis to transform words. For example, "running" would be lemmatized to "run," and "better" would be lemmatized to "good."',
]

# Create a DataFrame
df = pd.DataFrame({'Stemming': stemming, 'Lemmatization': lemm})

# Display the DataFrame
st.table(df)
st.divider()

st.header("Cosine Similarity")
st.info("We should learn Cosine Similarity because BOW and TF-IDF uses Cosine Similarity")
st.write("Cosine similarity is a mathematical metric used to measure the similarity between two non-zero vectors in a multi-dimensional space. It's often applied in various fields, including natural language processing, information retrieval, and machine learning.")

st.header("Bag Of Words")
st.info("A bag of words is a basic and simple technique in natural language processing (NLP) and text analysis. It's a way of representing text data as a collection of individual words, without considering the order or structure of the words in the text.")

st.write('In a "bag of words" model:')

bag_of_words_list = ["Each unique word in a document is treated as a separate entity.","The frequency of each word is counted within the document.","The resulting data is typically represented in a matrix or dictionary format","The values in this matrix or dictionary often represent word frequencies"]
st.write(bag_of_words_list)
st.write("This approach is called a bag because it's like throwing all the words from a document into a bag and then examining their frequencies without considering their order or structure. While it's a simple and computationally efficient method, it loses information about word order and can't capture the meaning or context of words in a document.")

bag_of_words_code = '''from sklearn.feature_extraction.text import CountVectorizer

# Sample list of documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer on the documents and transform them into a bag of words representation
X = vectorizer.fit_transform(documents)

# Get the feature names (unique words)
feature_names = vectorizer.get_feature_names_out()

# Convert the bag of words matrix to a dense array and print it
bag_of_words = X.toarray()
print("Bag of Words Representation:")
print(bag_of_words)

# Print the feature names (unique words)
print("\nFeature Names (Unique Words):")
print(feature_names)
'''

st.code(bag_of_words_code,language='python')

st.divider()
st.header("TF-IDF")
st.info("TF-IDF stands for Term Frequency-Inverse Document Frequency, and it is a numerical statistic that is often used in natural language processing and information retrieval to measure the importance of a word in a document relative to a collection of documents (corpus). It's commonly used for tasks such as document retrieval, text mining, and information retrieval.")
st.write("The TF-IDF score of a term in a document is calculated based on:")
tf_idf_list = ["TF (Term Frequency): Measures how often a term appears in a document.", "IDF: Measures the importance of a term in the entire collection of documents."]
st.write(tf_idf_list)

tf_idf_code = '''from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get the feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a dictionary to store TF-IDF scores for each term in each document
tfidf_scores = {}

# Iterate over each document and extract TF-IDF scores
for i, doc in enumerate(documents):
    terms = doc.split()
    for j, term in enumerate(terms):
        tfidf_score = tfidf_matrix[i, feature_names.index(term)]
        if term not in tfidf_scores:
            tfidf_scores[term] = []
        tfidf_scores[term].append((i, tfidf_score))

# Print TF-IDF scores for each term in each document
for term, scores in tfidf_scores.items():
    print(f"Term: {term}")
    for doc_id, score in scores:
        print(f"  Document {doc_id + 1}: TF-IDF = {score:.4f}")
'''

st.code(tf_idf_code, language='python')
st.divider()

st.header("What are Word Embeddings for text?")

st.write("Word embeddings are a type of mathematical representation used in natural language processing (NLP) and machine learning to convert words or phrases into numerical vectors. These vectors are designed to capture the semantic meaning of words and their relationships within a given language.")
st.write("Word embeddings are a crucial component in various NLP tasks because they enable computers to understand and work with textual data in a more meaningful way. Some popular techniques for creating word embeddings include Word2Vec, GloVe (Global Vectors for Word Representation), and fastText. These techniques use large corpora of text data to learn vector representations for words based on their context in sentences or documents.")
st.write("Word embeddings have several advantages in NLP applications:")

word_embedding_list = [
'Semantic Similarity: Words with similar meanings have similar vector representations. This property allows algorithms to understand the semantic relationships between words. For example, in a well-trained word embedding model, the vectors for "king" and "queen" will be closer in space compared to the vectors for "king" and "car."',
'Analogies: Word embeddings can be used to solve word analogies. For instance, with word embeddings, you can find that "king - man + woman" is closest to "queen" in vector space.',
"Dimension Reduction: Word embeddings typically have a lower dimensionality compared to one-hot encodings, making them more computationally efficient for many NLP tasks.",
"Improved NLP Models: Word embeddings are often used as input features for various NLP models, such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), to improve their performance in tasks like text classification, sentiment analysis, machine translation, and more."
]

df = pd.DataFrame({'Advantages': word_embedding_list})
st.table(df)

st.subheader("Word Embeddings Techniques")

word2vec = [
"Word2Vec is a word embedding model introduced by Mikolov et al. at Google in 2013. It's based on the idea that words with similar meanings should have similar vector representations. Word2Vec offers two main training algorithms: Continuous Bag of Words (CBOW) and Skip-gram."
]
Glove = [
"GloVe is another word embedding technique developed by Stanford researchers in 2014. Unlike Word2Vec, which uses a predictive model, GloVe is based on a count-based model. It builds a word-context co-occurrence matrix and factorizes it to obtain word vectors."
]

sub_word_embedding = pd.DataFrame({'Word2Vec': word2vec, 'Glove':Glove})

st.table(sub_word_embedding)
st.subheader("Subcategories of Word2vec")

subcategories_of_word2vec = [
"CBOW predicts a target word based on the context (surrounding words). It takes a context window of words and tries to predict the central word. The word vectors learned by CBOW are often good at capturing syntactic information.",
"Skip-gram, on the other hand, does the reverse. It predicts the context words given a central word. Skip-gram is generally better at capturing semantic relationships between words."
]

st.table({'Subcategories Of Word2Vec': subcategories_of_word2vec})
st.subheader("Comparison of Word2Vec and Glove")

Comparison = [
    "Training Approach: Word2Vec uses a predictive model, while GloVe uses a count-based approach.",
    "Training Efficiency: Word2Vec is typically faster to train and more scalable for large datasets. GloVe may require more resources.",
    "Vector Quality: Both Word2Vec and GloVe produce high-quality word vectors, but they may excel in different semantic and syntactic aspects. Skip-gram (Word2Vec) tends to capture semantic relationships better, while CBOW (Word2Vec) is good at capturing syntactic information. GloVe often strikes a balance between the two.",
    "Pre-trained Models: Pre-trained Word2Vec and GloVe embeddings are available and widely used in NLP tasks, saving time and resources in model training."
]

comparison = pd.DataFrame({'Comparison of Word2vec and Glove':Comparison})

st.table(comparison)

word_embedding_code = '''# Install gensim if you haven't already
# pip install gensim

from gensim.models import Word2Vec

# Sample corpus
corpus = [
    "Word embeddings are useful in NLP",
    "Python is a popular programming language",
    "NLP models are trained on large text data",
    "Embedding vectors represent words as numerical values"
]

# Tokenize the corpus
tokenized_corpus = [sentence.split() for sentence in corpus]

# Create and train the Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)

# Find the vector for a specific word
word_vector = model.wv['word']
print("Vector representation of 'word':\n", word_vector)

# Find similar words
similar_words = model.wv.most_similar('NLP', topn=3)
print("Words similar to 'NLP':\n", similar_words)
'''

st.code(word_embedding_code, language= 'python')
st.divider()
st.header("Transformers in NLP")
st.info("We will not go into the subject in great detail but I am going to teach you basic terms and concepts in Transformers")
st.write("Transformers are a class of deep learning models designed to handle sequential data, with a particular emphasis on NLP tasks. They were introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017. Transformers use a mechanism called self-attention to process input data in parallel, making them highly efficient for capturing long-range dependencies in sequences.")

st.subheader("Why do we use Transformers?")
why_transformers = [
"Parallelization: Transformers can process input sequences in parallel rather than sequentially, leading to faster training and inference times.",
"Attention Mechanism: The self-attention mechanism in Transformers allows them to capture relationships between words or tokens in a sequence, making them highly effective for understanding context and semantic meaning.",
"Scalability: Transformers can scale to handle both short and extremely long sequences, making them suitable for a wide range of NLP tasks.",
"State-of-the-Art Performance: Transformers have achieved state-of-the-art performance on various NLP benchmarks and have become the backbone of many language models like BERT, GPT, and more."
]
why_transformers_df = pd.DataFrame({'Why Transformers':why_transformers})
st.table(why_transformers_df)

st.subheader("Advantages of Transformers")
adv_transformers = [
"Efficient Parallel Processing: Transformers can process sequences in parallel, making them faster than traditional sequential models."
"Long-Range Dependencies: The self-attention mechanism allows them to capture relationships between distant words, which is crucial for understanding context.",
"Versatility: Transformers can be fine-tuned for a wide range of NLP tasks, including text classification, translation, summarization, and more.",
"Pretrained Models: Pretrained transformer-based models have been trained on massive amounts of text data, providing a valuable starting point for many NLP tasks.",
"Transfer Learning: Transfer learning with pretrained Transformers allows for better performance with smaller amounts of task-specific data.",
]
adv_transformers_df = pd.DataFrame({'Advantages': adv_transformers})
st.table(adv_transformers_df)

st.subheader("How to use Transformers")
how_transformers = [
"Pretrained Models: You can leverage pretrained transformer-based models (e.g., BERT, GPT-3) for various NLP tasks. Fine-tuning these models on your specific dataset is a common approach."
"Building Custom Models: You can also build custom Transformer architectures tailored to your specific task using deep learning frameworks like TensorFlow or PyTorch.",
"Hugging Face Transformers Library: The Hugging Face Transformers library is a popular resource for working with pretrained transformer models. It provides prebuilt models, training pipelines, and tools to make it easier to use Transformers for NLP tasks."
]
how_transformers_pd = pd.DataFrame({'How to use Transformers': how_transformers})
st.table(how_transformers_pd)

image_3 = Image.open('Images/3.png')
st.image(image_3, caption='Transformers Architecture')

st.write("""
**1-) Input Representation:**

The input to the Transformer consists of a sequence of tokens (words or subword pieces). Each token is typically represented as a vector through an embedding layer.
These embeddings capture the semantic meaning of each token and are often pretrained on large text corpora.

**2-) Positional Encoding:**

Transformers do not have inherent positional information, so positional encoding is added to the token embeddings.
Positional encoding is a set of vectors added to the input embeddings to indicate the position of each token in the sequence.
It enables the model to distinguish between tokens based on their position.

**3-) Multi-Head Self-Attention:**

The core innovation of the Transformer is the self-attention mechanism.
Self-attention computes weighted relationships between all tokens in the sequence, capturing dependencies and relationships between tokens at different positions.
It does this by assigning a weight to each token based on its similarity to other tokens.
The mechanism is applied multiple times in parallel using multiple attention "heads" to capture different types of relationships.

**4-) Position-wise Feed-Forward Networks:**

After self-attention, the output passes through position-wise feed-forward neural networks for each position in the sequence.
These networks consist of fully connected layers with activation functions (typically ReLU).
They introduce non-linearity and allow the model to learn complex relationships between tokens.

**5-) Layer Normalization and Residual Connections:**

Layer normalization is applied after each sub-layer (self-attention and feed-forward) to stabilize training.
Residual connections (skip connections) are used to preserve information from the input to the output of each sub-layer, helping with gradient flow.

** 6-) Encoder-Decoder Architecture (Optional):**

While the original Transformer paper introduced a model for sequence-to-sequence tasks, a common variant is to use separate encoder and decoder stacks.
The encoder processes the input sequence, and the decoder generates the output sequence.

**7-) Decoder Self-Attention (For Sequence-to-Sequence Models):**

In sequence-to-sequence tasks, the decoder also uses self-attention, but with a twist.
The decoder's self-attention mechanism is masked to ensure it only attends to previous positions, preventing information leakage from future positions.

**8-) Output Layer:**

The output from the decoder (or encoder-decoder stack) is typically fed through a linear layer followed by a softmax activation function for classification tasks or a linear layer for regression tasks.

**Training and Optimization:**

Transformers are trained using gradient-based optimization algorithms like Adam or SGD.
Training often involves minimizing a loss function specific to the task (e.g., cross-entropy for classification or mean squared error for regression).
Pretrained models are often fine-tuned on task-specific data to improve performance.

**9-) Inference:**

During inference, the model can generate sequences autoregressively (e.g., for text generation) or make predictions directly (e.g., for classification).""")

st.divider()
st.header("What is BERT")

st.info("BERT, which stands for Bidirectional Encoder Representations from Transformers, is a natural language processing (NLP) model developed by Google in 2018. It's a type of transformer-based neural network architecture that has significantly improved the way computers understand and generate human language.")
bert_list = [
"Bidirectional Understanding: BERT is designed to understand the context of a word by considering the surrounding words on both sides. This bidirectional approach helps it capture the nuances of language better than previous models."
"Pretraining and Fine-tuning: BERT is pretrained on a massive corpus of text, learning to predict missing words in sentences. After pretraining, it can be fine-tuned for specific NLP tasks like text classification, named entity recognition, and question-answering.",
"State-of-the-Art Performance: BERT achieved state-of-the-art results on a wide range of NLP tasks, surpassing the performance of previous models. It has been the basis for many subsequent NLP models and architectures.",
"Transformer Architecture: BERT is built on the transformer architecture, which relies on self-attention mechanisms to process sequences of data. Transformers have become the backbone of many NLP models due to their effectiveness."
]
bert_list_df = pd.DataFrame({'Bert':bert_list})
st.table(bert_list_df)


st.divider()
st.header("What is HuggingFace")
st.info("Hugging Face is a company and open-source platform that specializes in natural language processing (NLP) and deep learning. They are best known for their contributions to the development of NLP models and libraries, making it easier for researchers and developers to work with state-of-the-art NLP technologies.")
huggingface_list = [
"Transformers Library: Hugging Face is perhaps most famous for their Transformers library, which provides pre-trained models for a wide range of NLP tasks, such as text classification, translation, question answering, and more. These models are based on transformer architectures like BERT, GPT, RoBERTa, and others."
"Open Source: Much of Hugging Face's work is open-source, meaning that their models and code are freely available for anyone to use, modify, and build upon. This has led to widespread adoption and innovation in the NLP community.",
"Hugging Face Hub: They have created the Hugging Face Hub, a platform for sharing and downloading pre-trained models and datasets. This makes it easy for developers and researchers to access and use NLP resources.",
"Community and Collaboration: Hugging Face actively engages with the NLP community and encourages collaboration. They host competitions, workshops, and events to promote research and development in the field.",
"APIs and Tools: Hugging Face provides APIs and tools that allow developers to integrate their NLP models into applications, making it easier to build chatbots, sentiment analysis tools, language translation services, and more.",
"Research and Innovation: The company is involved in cutting-edge NLP research and often releases new models and techniques that push the boundaries of what's possible in natural language understanding."
]
huggingface_list_df = pd.DataFrame({'HugginFace':huggingface_list})
st.table(huggingface_list_df)

st.divider()
st.header("RNN (Recurrent Neural Networks)")
st.write("Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed for processing sequences of data, such as time series, text, and speech. Unlike traditional feedforward neural networks, RNNs have connections that loop back on themselves, allowing them to maintain a hidden state that captures information about previous inputs in the sequence. ")

st.subheader("What is RNN")
st.info("""An RNN is a type of neural network architecture that introduces the concept of recurrent connections. 
These recurrent connections allow information to persist and be passed from one step of the sequence to the next, enabling the network to capture temporal dependencies.
The key component of an RNN is its hidden state, which is updated at each time step and encodes information about previous inputs in the sequence.""")

st.subheader("Why do we use RNNs?")
why_rnn = [
"Natural Language Processing (NLP): RNNs are widely used for tasks like language modeling, text generation, sentiment analysis, and machine translation."
"Speech Recognition: RNNs can be applied to convert audio signals into text.",
"Time Series Analysis: RNNs are used for tasks like stock price prediction, weather forecasting, and anomaly detection.",
"Video Analysis: RNNs can process video sequences for tasks like action recognition and object tracking.",
"Generative Modeling: RNNs can generate sequences, such as music, text, and images.",
"RNNs are particularly well-suited for tasks that involve sequences of varying lengths because they can handle inputs of different lengths thanks to their recurrent nature."
]
why_rnn_df = pd.DataFrame({'Why do we use Rnns':why_rnn})
st.table(why_rnn_df)

st.subheader("Advantages Of RNNs?")
adv_rnn = [
"Sequential Data Processing: RNNs excel at capturing dependencies in sequential data. They can model the context and relationships between elements in a sequence."
"Flexibility: RNNs are flexible and can be used for a wide range of tasks, from short-text classification to long-term time series forecasting.",
"Stateful Learning: RNNs maintain an internal hidden state that allows them to remember information from past time steps. This stateful learning is essential for tasks where context matters.",
"Online Learning: RNNs can be trained incrementally as new data becomes available, making them suitable for online learning scenarios.",
"Generative Capabilities: RNNs can generate new sequences of data, making them useful for creative applications like text generation and music composition."
]
adv_rnn_df = pd.DataFrame({'Advantages of RNNs':adv_rnn})
st.table(adv_rnn_df)


image_4 = Image.open('Images/4.png')
st.image(image_4, caption='RNN')
st.write("In summary, Recurrent Neural Networks (RNNs) are a class of neural networks designed for processing sequential data. They are essential for various tasks where understanding the temporal context and relationships in data is crucial. However, they also come with challenges, such as the vanishing gradient problem, which has led to the development of more advanced RNN variants like LSTMs and GRUs to address these limitations.")
st.divider()
st.header("Create NLP Pipeline For Text Classification Using Tensorflow")

st.subheader("Libraries")
libraries = '''import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
'''

preprocessing = '''
# Sample data for text classification
texts = ["This is a positive sentence.", "I don't like negative sentences.", "Neutral sentences are okay."]
labels = [1, 0, 2]  # 1 for positive, 0 for negative, 2 for neutral

# Step 1: Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenization and lowercase
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing to all texts
preprocessed_texts = [preprocess_text(text) for text in texts]
'''

word_embedding = '''
# Step 2: Word Embedding (using pre-trained GloVe embeddings)
embedding_dim = 100  # You can choose the embedding dimension based on your pre-trained word embeddings
max_sequence_length = 10  # You can choose the sequence length based on your dataset

# Load pre-trained GloVe embeddings
glove_embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_embeddings_index[word] = coefs

# Create a tokenizer and convert text to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_texts)
sequences = tokenizer.texts_to_sequences(preprocessed_texts)

# Pad sequences to a fixed length
sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Create an embedding matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = glove_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
'''


rnn_model = '''
# Step 3: Build RNN model
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(SimpleRNN(32))  # You can choose the number of units based on your task
model.add(Dense(3, activation='softmax'))  # Output layer with 3 classes (positive, negative, neutral)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
'''

st.code(libraries, language='python')
st.subheader('Preprocessing')
st.code(preprocessing, language='python')
st.subheader('Word Embedding')
st.code(word_embedding, language = 'python')
st.subheader('RNN Model')
st.code(rnn_model, language = 'python')
st.info("Test Loss: 0.3521, Test Accuracy: 0.8685")

