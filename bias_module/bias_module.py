import os
import csv
import gzip
import shutil
import gdown
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel, BertForMaskedLM, pipeline, BertTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM # Import AutoTokenizer and AutoModelForSeq2SeqLM
import spacy
import re
import language_tool_python
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

class BiasDetector:
    def __init__(self):
        self.word_vectors = None
        self.male_words = ["he", "him", "his", "himself", "Man","Gentleman","Boy","Male ","Father","Dad","Daddy","Papa","Pa","Son","Brother","Husband","Uncle","Grandfather","Grandpa","Granddad","Nephew","Patriarch","Godfather","Stepfather","Stepbrother","Brother-in-law","Father-in-law","Foster father","Mister (Mr.)","Sir","Lord","His Lordship","Master","Esquire","Actor","Host","Waiter","Steward","Governor","Seamster","Poet","Emperor","Sorcerer","Priest","Prophet ","King","Prince","Duke","Baron","Marquis","Emperor","Count","Viscount","Tsar / Czar","Sultan","Pharaoh","Monk","Abbot","Priest","Deacon","God","Titan","Warlock","Wizard","Bachelor","Groom","Fiancé","Widower","Beau","Gentleman","Lad","Laddie","Chap","Bloke","Fellow","Cavalier","Squire","Dandy"]
        self.female_words = ["she", "her", "hers", "herself","Woman","Lady","Girl","Female ","Mother","Mom","Mum","Mommy","Mummy","Mama","Daughter","Sister","Wife","Aunt","Grandmother","Grandma","Granny","Nan","Nana","Niece","Matriarch","Godmother","Stepmother","Stepsister","Sister-in-law","Mother-in-law","Foster mother","Wet nurse","Midwife","Madam","Miss","Mrs.","Ms.","Dame","Lady","Ladyship","Queen Mother","Actress","Hostess","Waitress","Stewardess","Governess","Seamstress","Songstress","Countess","Baroness","Poetess","Empress","Sorceress","Priestess","Prophetess","Seductress","Temptress","Queen","Princess","Duchess","Baroness","Marchioness","Empress","Countess","Viscountess","Heiress","Tsarina","Sultana ","Nun","Abbess","Priestess","Goddess","Valkyrie","Muse","Siren ","Bachelorette","Belle","Bride","Fiancée","Spinster","Diva","Socialite","Housewife","Maiden","Damsel","Lass","Lassie","Doyenne","Debutante"]
        self.model_file = "GoogleNews-vectors-negative300.bin.gz"  # Define model_file here
        self.download_model() # Download the model
        self.extract_model(self.model_file, "GoogleNews-vectors-negative300.bin") # Extract the model
        self.load_model() # Load the model
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


    def download_model(self):
        """Download the Word2Vec model from Google Drive."""
        shareable_link_google = "https://drive.google.com/file/d/1qIZ76gmniBA11mXfylA4TpNrQ2gHfFgE/view?usp=sharing"
        google_file_id = shareable_link_google.split("/d/")[1].split("/view")[0]
        google_download_url = f"https://drive.google.com/uc?id={google_file_id}"
        if not os.path.exists(self.model_file):
          gdown.download(google_download_url, self.model_file, quiet=False)

    def extract_model(self, gz_file_path, extracted_file_path):
        """Extract the .gz file."""
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(extracted_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def load_model(self):
        """Load the Word2Vec model."""
        self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    def preprocess_sentence(self, sentence):
        """Preprocess the sentence by converting it to lowercase and filtering out unknown words."""
        return [word.lower() for word in sentence.split() if word.lower() in self.word_vectors]

    def get_average_vector(self, words):
        """Get the average vector for a list of words."""
        vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word_vectors.vector_size)
        
    def sentiment_analysis(self, sentence):
      """ Do sentiment analysis of the sentence to determine if the sentence is biased or not along with WEAT score"""
      result = self.sentiment_pipeline(sentence)
      return result[0]['label'], result[0]['score']

    def get_word_vector(self, word):
      """Retrieve the word embedding vector or return a zero vector if not found."""
      if word in self.word_vectors:
          return self.word_vectors[word]  # Returns the 300-dimensional vector
      else:
          return np.zeros(300)  # Returns a zero vector if word is missing

    def weat_score(self,X, A, B):
      """
      Compute the WEAT effect size for a set of target words X against attribute sets A and B.

      X: Word embeddings for the target words (sentence words)
      A: Word embeddings for attribute set A (male words)
      B: Word embeddings for attribute set B (female words)
      """
      mean_A = np.mean([1 - cosine(x, a) for x in X for a in A])  # Similarity with A
      mean_B = np.mean([1 - cosine(x, b) for x in X for b in B])  # Similarity with B

      std_dev = np.std([1 - cosine(x, a) for x in X for a in A] +
                      [1 - cosine(x, b) for x in X for b in B])  # Std deviation

      return (mean_A - mean_B) / std_dev if std_dev != 0 else 0  # Effect size

    def compute_bias_weat(self, sentence):
      """Compute gender bias using the WEAT score."""
      words = self.preprocess_sentence(sentence)
      sentiment_result, sentiment_score = self.sentiment_analysis(sentence)
      print("Sentiment analysis : ", sentiment_result)
      if sentiment_result == "neutral":
        score = sentiment_score * 0.1
        return {'Sentence': [sentence], 'label': ["Non-biased"], 'score': [score], 'sentiment_analysis': [sentiment_result]}
      elif sentiment_result == "positive":
        score = sentiment_score * 0.1
        return {'Sentence': [sentence], 'label': ["Non-Biased"], 'score': [score], 'sentiment_analysis': [sentiment_result]}
      else:
        sentence_vectors = [self.get_word_vector(word) for word in words if word in self.word_vectors]

        male_vectors = [self.get_word_vector(word) for word in self.male_words if word in self.word_vectors]
        female_vectors = [self.get_word_vector(word) for word in self.female_words if word in self.word_vectors]

        if not sentence_vectors or not male_vectors or not female_vectors:
          return {'Sentence': [sentence], 'label': ["Insufficient Data"], 'score': [0]}

        weat = self.weat_score(sentence_vectors, male_vectors, female_vectors)

        bias_label = "Biased" if abs(weat) > 0.2 else "Non-biased"  # Threshold for bias detection

        return {'Sentence': [sentence], 'label': [bias_label], 'score': [weat], 'sentiment_analysis': [sentiment_result]}

    def compute_bias(self, sentence):
        """Compute the gender bias for a sentence."""
        words = self.preprocess_sentence(sentence)
        sentence_vector = self.get_average_vector(words)
        male_vector = self.get_average_vector(self.male_words)
        female_vector = self.get_average_vector(self.female_words)

        male_similarity = cosine_similarity([sentence_vector], [male_vector])[0][0]
        female_similarity = cosine_similarity([sentence_vector], [female_vector])[0][0]

        bias = male_similarity - female_similarity
        if bias > 0.1 or bias < -0.1:
            bias_result = {'Sentence': [sentence], 'label': ["Biased"], 'score': [bias]}
        else:
            bias_result = {'Sentence': [sentence], 'label': ["Non-biased"], 'score': [bias]}
        return bias_result


class BiasFilter:
  PRONOUNS = {
        "he": "they", "she": "they",
        "him": "them", "her": "them",
        "his": "their", "hers": "theirs",
        "himself": "themselves", "herself": "themselves",
        "s/he": "they", "xe": "they", "ze": "they", "ey": "they"
    }

  GENDERED_NOUNS = {
        "husband": "spouse", "wife": "spouse",
        "father": "parent", "mother": "parent",
        "son": "child", "daughter": "child",
        "brother": "sibling", "sister": "sibling",
        "uncle": "relative", "aunt": "relative",
        "nephew": "relative", "niece": "relative",
        "grandfather": "grandparent", "grandmother": "grandparent",
        "fiancé": "partner", "fiancée": "partner",
        "boy": "child", "girl": "child",
        "stepfather": "stepparent", "stepmother": "stepparent",
        "stepson": "stepchild", "stepdaughter": "stepchild",
        "godfather": "godparent", "godmother": "godparent",
        "boyfriend": "partner", "girlfriend": "partner",
        "bride": "married partner", "groom": "married partner",
        "fireman": "firefighter", "policeman": "police officer",
        "mailman": "mail carrier", "salesman": "salesperson",
        "businessman": "businessperson", "chairman": "chairperson",
        "waiter": "server", "waitress": "server",
        "steward": "flight attendant", "stewardess": "flight attendant",
        "actor": "performer", "actress": "performer",
        "congressman": "legislator", "congresswoman": "legislator",
        "alderman": "council member", "ombudsman": "ombudsperson",
        "postman": "postal worker", "weatherman": "meteorologist",
        "craftsman": "artisan", "foreman": "supervisor",
        "milkman": "milk deliverer", "fisherman": "fisher",
        "draftsman": "drafter", "seamstress": "sewer",
        "laundryman": "laundry worker", "repairman": "repair technician",
        "watchman": "security guard", "middleman": "intermediary",
        "workman": "worker", "spokesman": "spokesperson",
        "forefather": "ancestor", "maid": "housekeeper",
        "housewife": "homemaker", "househusband": "homemaker",
        "bondsman": "guarantor", "clergyman": "clergy",
        "policewoman": "police officer", "handyman": "maintenance worker",
        "showman": "entertainer", "cameraman": "camera operator",
        "prince": "royal", "princess": "royal",
        "king": "monarch", "queen": "monarch",
        "duke": "noble", "duchess": "noble",
        "emperor": "ruler", "empress": "ruler",
        "lord": "noble", "lady": "noble",
        "sultan": "ruler", "sultana": "ruler",
        "serviceman": "service member", "servicemen": "service members",
        "servicelady": "service member", "serviceladies": "service members",
        "airman": "aviator", "seaman": "sailor",
        "infantryman": "infantry soldier", "guardsman": "guard",
        "rifleman": "sharpshooter", "midshipman": "naval officer",
        "monk": "clergy", "nun": "clergy",
        "priest": "clergy", "priestess": "clergy",
        "chaplain": "spiritual leader",
        "sportsman": "athlete", "sportswoman": "athlete",
        "batman": "cricket assistant", "batboy": "equipment manager",
        "linesman": "referee", "ballboy": "ball retriever",
        "headmaster": "principal", "headmistress": "principal",
        "councilman": "council member", "councilwoman": "council member",
        "founding fathers": "founders", "manpower": "workforce",
        "mankind": "humanity", "brotherhood": "fellowship",
        "fellow": "peer", "grandmaster": "expert",
        "layman": "non-specialist", "marksman": "sharpshooter",
        "newsman": "journalist", "nobleman": "noble",
        "playboy": "socialite", "showgirl": "performer",
        "strongman": "weightlifter", "tradesman": "trader",
        "workman": "worker", "yachtsman": "sailor",
        "drummer boy": "drummer", "peasant woman": "farmer",
        "gentleman": "person", "lady": "person",
        "grandson": "grandchild", "granddaughter": "grandchild",
        "bachelor": "unmarried person", "spinster": "unmarried person",
        "manhunt": "search", "policemen": "police officers",
        "bridegroom": "married partner", "horseman": "rider",
        "journeyman": "skilled worker", "lumberjack": "logger",
        "midwife": "birth assistant", "tomboy": "energetic child",
        "wise man": "sage", "witch": "magic practitioner",
        "wizard": "magic practitioner", "heir": "successor",
        "heiress": "successor"
    }
  def __init__(self):
    # Load the pre-trained BERT model and tokenizer
    self.tokenizer = BertTokenizerFast.from_pretrained("kdadobe1/bert-bias-detection-retrained")
    self.bert_model = BertForMaskedLM.from_pretrained("kdadobe1/bert-bias-detection-retrained")
    self.t5_tokenizer = AutoTokenizer.from_pretrained("kdadobe1/google_flan_t5_small_retrained")  # Or a larger t5_model
    #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("kdadobe1/google_flan_t5_small_retrained")
    self.tool = language_tool_python.LanguageTool("en-US")
    #self.t5_model.to(self.device)
    self.detector = BiasDetector()

  def complete_statement(self, masked_statement):
    """Predicts the masked word(s) in the input statement."""
    inputs = self.tokenizer(masked_statement, return_tensors="pt")
    with torch.no_grad():
        outputs = self.bert_model(**inputs)
    predictions = outputs.logits
    # Get the predicted token index for the [MASK] token
    mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
    print(predictions)
    predicted_token_id = predictions[0, mask_token_index, :].argmax(axis=-1)
    # Replace the [MASK] token with the predicted word
    predicted_token = self.tokenizer.decode(predicted_token_id)
    completed_statement = masked_statement.replace(self.tokenizer.mask_token, predicted_token)
    return completed_statement, predicted_token
    
    
  def is_biased(self, statement):
    """Uses the dbias classifier to check if the statement is biased."""
    biased_result = self.detector.compute_bias_weat(statement)
    print(biased_result)
    return biased_result['label'][0], biased_result['score'][0], biased_result['sentiment_analysis'][0]

  def rephrase_with_t5(self, sentence):
    """Rephrases the sentence in detoxified, gender-neutral, and proper English format."""

    # Stronger and clearer prompt
    input_text = f"Paraphrase this sentence in fluent, detoxified English and correct the punctuation marks: {sentence}"

    input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids
    #input_ids = input_ids.to(self.device)

    outputs = self.t5_model.generate(
        input_ids,
        max_length=100,
        min_length=10,
        temperature=0.7,
        top_p=0.9,
        num_beams=5
    )

    # Decode and clean up the output
    rephrased_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(rephrased_text)

    # Remove any unwanted repetition of the input prompt
    if rephrased_text.lower().startswith("paraphrase this"):
        rephrased_text = rephrased_text.split(":", 1)[-1].strip()

    rephrased_text = self.tool.correct(rephrased_text)
    return rephrased_text


  def rephrase_with_combined(self,sentence):
      """ Makes sentence gender-neutral using regex, dictionary replacement, and T5 (optional). """
      doc = nlp(sentence[:1000])
      new_sentence = []
      PRONOUN_SET = set(BiasFilter.PRONOUNS.keys())
      GENDERED_NOUN_SET = set(BiasFilter.GENDERED_NOUNS.keys())
      for token in doc:
          word_lower = token.text.lower()
          print("The word is ", word_lower)
          if word_lower in BiasFilter.PRONOUNS:
              print("In pronouns")
              new_sentence.append(BiasFilter.PRONOUNS[word_lower])
              
          if token.pos_ == "NOUN" and word_lower in BiasFilter.GENDERED_NOUNS:
              print("In gender nouns")
              new_sentence.append(BiasFilter.GENDERED_NOUNS[word_lower])
          else:
              new_sentence.append(token.text)
      neutral_sentence = " ".join(new_sentence)
      neutral_sentence_t5 = self.rephrase_with_t5(neutral_sentence)
      neutral_sentence_lang = self.tool.correct(neutral_sentence_t5)
      return neutral_sentence_t5, neutral_sentence_lang

  def process_statement(self, masked_statement):
      """Handles the full workflow: completion, bias detection, and correction."""
      completed_statement, predicted_token = self.complete_statement(masked_statement)
      print(f"Completed Statement: {completed_statement}")
      print(f"Predicted token: {predicted_token}")
      biased, initial_bias_score, sentiment_analysis = self.is_biased(completed_statement)
      if biased == 'Biased':
          print("******The statement is biased*******")
          print("Initial Bias score ", initial_bias_score)
          print("******Applying gender-neutral filter.******")
          gender_neutral_statement_t5, neutral_sentence_lang  = self.rephrase_with_combined(completed_statement)
          final_bias, final_bias_score, final_sentiment_analysis = self.is_biased(gender_neutral_statement_t5)
          return completed_statement, initial_bias_score, sentiment_analysis, gender_neutral_statement_t5, neutral_sentence_lang, final_bias_score
      else:
          print("The statement is not biased.")
          return completed_statement, initial_bias_score, sentiment_analysis, completed_statement, completed_statement,  initial_bias_score

