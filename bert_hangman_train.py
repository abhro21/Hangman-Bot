# -*- coding: utf-8 -*-
"""bert_hangman.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dcgU8rK0ht7Ten6jD0yBecY064uIlpf3

# Trexquant Interview Project (The Hangman Game)

* Copyright Trexquant Investment LP. All Rights Reserved.
* Redistribution of this question without written consent from Trexquant is prohibited

## Instruction:
For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server.

When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
or (2) the user has made six incorrect guesses.

You are required to write a "guess" function that takes current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games. You have the opportunity to practice before you want to start recording your game results.

Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.

You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.

This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.
"""

import json
import requests
import random
import string
import secrets
import time
import re
import collections
try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from tqdm import tqdm
import random

class HangmanDataset(Dataset):
    def __init__(self, words, tokenizer, max_length):
        self.words = words
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        masked_word = self.mask_word(word)
        inputs = self.tokenizer(masked_word, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(word, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"]

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

    def mask_word(self, word):
        mask_prob = 0.15
        tokens = list(word)
        for i in range(len(tokens)):
            if random.random() < mask_prob:
                tokens[i] = '[MASK]'
        return ''.join(tokens)

class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []

        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()

        self.current_dictionary = []
        self.n_word_dictionary = self.build_n_word_dictionary(self.full_dictionary)

        # Load and fine-tune BERT model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = self.fine_tune_bert()
        self.model.eval()  # Set the model to evaluation mode


    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']
        data = {link: 0 for link in links}
        for link in links:
            requests.get(link)
            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s
        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def build_n_word_dictionary(self, word_list):
        n_word_dict = {i: [] for i in range(3, 30)}
        max_length = self.find_max_length(word_list)
        for count in range(3, max_length + 1):
            for word in word_list:
                if len(word) >= count:
                    for i in range(len(word) - count + 1):
                        n_word_dict[count].append(word[i:i + count])
        return n_word_dict

    def find_max_length(self, word_list):
        return max(len(word) for word in word_list)

    def build_dictionary(self, dictionary_file_location):
        with open(dictionary_file_location, "r") as text_file:
            full_dictionary = text_file.read().splitlines()
        return full_dictionary

    def vowel_count(self, clean_word):
        vowels = "aeiou"
        count = sum(1 for char in clean_word if char in vowels)
        return count / len(clean_word) if len(clean_word) > 0 else 0.0

    def func(self, new_dictionary):
        dictx = collections.Counter()
        for words in new_dictionary:
            temp = collections.Counter(words)
            for i in temp:
                temp[i] = 1
                dictx = dictx + temp
        return dictx

    def func2(self, n_word_dictionary, clean_word):
        new_dictionary = []
        l = len(clean_word)
        for dict_word in n_word_dictionary[l]:
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)
        return new_dictionary

    def get_state(self, word):
        return word[::2].replace("_", ".")

    def _original_guess(self, word, n_word_dictionary):
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        current_dictionary = self.current_dictionary
        new_dictionary = []

        for dict_word in current_dictionary:
            if len(dict_word) != len_word:
                continue
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)

        self.current_dictionary = new_dictionary

        c = self.func(new_dictionary)
        sorted_letter_count = c.most_common()

        guess_letter = '!'
        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                if letter in "aeiou" and self.vowel_count(clean_word) > 0.52:
                    self.guessed_letters.append(letter)
                    continue
                guess_letter = letter
                break

        if guess_letter == '!':
            new_dictionary = self.func2(n_word_dictionary, clean_word)
            c = self.func(new_dictionary)
            sorted_letter_count = c.most_common()
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    if letter in "aeiou" and self.vowel_count(clean_word) > 0.52:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break

        if guess_letter == '!':
            x = int(len(clean_word) / 2)
            if x >= 3:
                c = collections.Counter()
                for i in range(len(clean_word) - x + 1):
                    s = clean_word[i:i + x]
                    new_dictionary = self.func2(n_word_dictionary, s)
                    temp = self.func(new_dictionary)
                    c = c + temp
                sorted_letter_count = c.most_common()
                for letter, instance_count in sorted_letter_count:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break

        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    if letter in "aeiou" and self.vowel_count(clean_word) > 0.52:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break

        return guess_letter


    def fine_tune_bert(self):
        print("Fine-tuning BERT model...")
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        # Prepare the dataset
        max_length = max(len(word) for word in self.full_dictionary) + 2  # Add 2 for [CLS] and [SEP] tokens
        dataset = HangmanDataset(self.full_dictionary, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Set up the optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)

        # Fine-tuning loop
        model.train()
        num_epochs = 3
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        print("Fine-tuning complete!")

        # Save the fine-tuned model and tokenizer
        model_save_path = 'fine_tuned_bert_model'
        model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Model saved at {model_save_path}")

        return model

    def load_fine_tuned_bert(self):
        model_save_path = 'fine_tuned_bert_model'
        model = BertForMaskedLM.from_pretrained(model_save_path)
        tokenizer = BertTokenizer.from_pretrained(model_save_path)
        print(f"Model and tokenizer loaded from {model_save_path}")
        return model, tokenizer

    def guess(self, word, n_word_dictionary):
        clean_word = word[::2].replace("_", ".")
        remaining_letters = clean_word.count('.')

        if remaining_letters <= 3:
            return self._bert_guess(clean_word)
        else:
            return self._original_guess(word, n_word_dictionary)

    def _bert_guess(self, clean_word):
        possible_words = [word for word in self.full_dictionary if len(word) == len(clean_word) and re.match(clean_word, word)]

        if not possible_words:
            return self._original_guess(clean_word, self.n_word_dictionary)

        # Prepare input for BERT
        inputs = self.tokenizer(clean_word.replace('.', '[MASK]'), return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()

        for token in top_tokens:
            predicted_token = self.tokenizer.decode([token])
            if predicted_token.isalpha() and predicted_token.lower() not in self.guessed_letters:
                return predicted_token.lower()

        # Fallback to original strategy if BERT doesn't provide a valid guess
        return self._original_guess(clean_word, self.n_word_dictionary)

    def _filter_dictionary(self, word):
        """Filter the dictionary after each guess based on the current word state."""
        clean_word = word[::2].replace("_", ".")
        filtered_dictionary = []

        for dict_word in self.current_dictionary:
            if re.match(clean_word, dict_word):
                filtered_dictionary.append(dict_word)

        # Aggressively filter current dictionary based on new word state
        self.current_dictionary = filtered_dictionary

    def _guess_with_n_grams(self, clean_word):
        """Guess letter based on bi-grams and tri-grams."""
        c = collections.Counter()

        # Iterate through the clean word and match with bi-grams and tri-grams
        for i in range(len(clean_word) - 1):
            partial_word = clean_word[i:i+2]
            for bi_gram, count in self.bi_grams.items():
                if re.match(partial_word, bi_gram) and bi_gram[1] not in self.guessed_letters:
                    c[bi_gram[1]] += count

        for i in range(len(clean_word) - 2):
            partial_word = clean_word[i:i+3]
            for tri_gram, count in self.tri_grams.items():
                if re.match(partial_word, tri_gram) and tri_gram[2] not in self.guessed_letters:
                    c[tri_gram[2]] += count

        # Choose the most common letter in bi-grams and tri-grams if available
        sorted_letter_count = c.most_common()
        for letter, count in sorted_letter_count:
            if letter not in self.guessed_letters:
                return letter

        # If still no guess, return '!'
        return '!'

    def start_game(self, practice=True, verbose=True):
        """Modified start game to initialize bi-grams and tri-grams."""
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        # Build bi-grams and tri-grams for backup strategy
        # self.bi_grams, self.tri_grams = self.build_n_grams(self.full_dictionary)

        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print(f"Successfully started a new game! Game ID: {game_id}. # of tries remaining: {tries_remains}. Word: {word}.")

            while tries_remains > 0:
                guess_letter = self.guess(word, self.n_word_dictionary)
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print(f"Guessing letter: {guess_letter}")

                try:
                    res = self.request("/guess_letter", {"request": "guess_letter", "game_id": game_id, "letter": guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e

                if verbose:
                    print(f"Server response: {res}")

                next_word = res.get('word')
                tries_remains = res.get('tries_remains')

                if res.get('status') == "success":
                    # Filter dictionary after correct guess
                    self._filter_dictionary(next_word)
                    if verbose:
                        print(f"Successfully finished game: {game_id}")
                    return True
                elif res.get('status') == "failed":
                    reason = res.get('msg', "Unknown error")
                    if verbose:
                        print(f"Game {game_id} failed. Reason: {reason}")
                    return False
                else:
                    word = next_word

        else:
            print(f"Error starting a new game: {response}")
        return False

    def my_status(self):
        return self.request("/my_status", {})

    def request(self, endpoint, params):
        if 'hangman' in self.hangman_url:
            url = self.hangman_url + endpoint
        else:
            url = self.hangman_url + endpoint

        if self.access_token:
            params["access_token"] = self.access_token

        headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else None
        res = self.session.get(url, params=params, timeout=self.timeout, headers=headers)

        if res.status_code == 200:
            return res.json()
        else:
            raise HangmanAPIError(f"Failed to perform request to {url}. Status code: {res.status_code}")


class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)

"""# API Usage Examples

## To start a new game:
1. Make sure you have implemented your own "guess" method.
2. Use the access_token that we sent you to create your HangmanAPI object.
3. Start a game by calling "start_game" method.
4. If you wish to test your function without being recorded, set "practice" parameter to 1.
5. Note: You have a rate limit of 20 new games per minute. DO NOT start more than 20 new games within one minute.
"""

# Script to use the HangmanAPI with loaded model
def play_hangman_with_loaded_model(num_games, practice=1):
    # Initialize the HangmanAPI (it will load the model by default)
    api = HangmanAPI(access_token="55d0caa2c23420f098a5efd9b453af", timeout=2000)

    wins = 0
    losses = 0

    for game_num in range(1, num_games + 1):
        print(f"Starting game {game_num}")

        result = api.start_game(practice=practice, verbose=True)

        if result:
            wins += 1
            print(f"Game {game_num}: Won")
        else:
            losses += 1
            print(f"Game {game_num}: Lost")

        print(f"Current stats - Wins: {wins}, Losses: {losses}")
        print(f"Win rate: {wins / game_num:.2%}")

        # Optional: Add a small delay to avoid overwhelming the API
        time.sleep(1)


    print("\nGames complete!")
    print(f"Final stats - Wins: {wins}, Losses: {losses}")
    print(f"Final win rate: {wins / num_games:.2%}")

    api.save_model()


# Play 100 games with the loaded model`
play_hangman_with_loaded_model(num_games=50, practice=1)

"""## Playing practice games:
You can use the command below to play up to 100,000 practice games.
"""