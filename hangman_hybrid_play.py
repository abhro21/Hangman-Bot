import sys
import requests
import time
import collections
import re
from transformers import BertForMaskedLM, BertTokenizer
import torch
import numpy as np
from datetime import date
from collections import defaultdict, Counter
from datetime import datetime

class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        
        print(f"Initiating...", file=sys.stderr)
        
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []

        print(f"Loading Dictionary...", file=sys.stderr)
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.consecutive_wrong_guesses = 0
        
        
        self.current_dictionary = []
        self.n_word_dictionary = self.build_n_word_dictionary(self.full_dictionary)
        
        # Set path to fine-tuned BERT model
        model_dir = 'fine_tuned_bert_model'

        print(f"Loading BERT Model...", file=sys.stderr)
        
        # Load BERT model and tokenizer from Google Drive
        self.bert_model = BertForMaskedLM.from_pretrained(model_dir)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.bert_model.eval()  # Set the model to evaluation mode


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

    def getCounterFromDict(self, new_dictionary):
        dictx = collections.Counter()
        for words in new_dictionary:
            temp = collections.Counter(words)
            for i in temp:
                temp[i] = 1
                dictx = dictx + temp
        return dictx

    def appendWordToDict(self, n_word_dictionary, clean_word):
        new_dictionary = []
        l = len(clean_word)
        for dict_word in n_word_dictionary[l]:
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)
        return new_dictionary

    def get_state(self, word):
        return word[::2].replace("_", ".")

    def guessFirstVowel(self, word, n_word_dictionary, iterationCount):
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        
        guess_letter = 'e'
        
        #print ("First Guess Vowel - Iteration=",iterationCount)
        
        vowel_dictionary = []
        current_dictionary = self.current_dictionary
        #print ("Length of current_dictionary for vowels= ",len(current_dictionary))
        for dict_word in current_dictionary:
            if len(dict_word) != len_word:
                continue
            else:
                vowelWord = re.sub('[bcdfghjklmnpqrstvwxyz]', '', dict_word)
                vowel_dictionary.append(vowelWord)

        c = self.getCounterFromDict(vowel_dictionary)
        sorted_vowel_count = c.most_common(iterationCount)
        
        for letter, instance_count in sorted_vowel_count:
            guess_letter = letter
            #print ("First Guess Vowel: ", guess_letter)
        
        return guess_letter
        
    def guessFirstConsonant(self, word, n_word_dictionary, iterationCount):
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        
        guess_letter = 't'
        
        #print ("First Guess Consonant - Iteration=",iterationCount)
        
        const_dictionary = []
        current_dictionary = self.current_dictionary
        #print ("Length of current_dictionary for consonants = ",len(current_dictionary))
        for dict_word in current_dictionary:
            if len(dict_word) != len_word:
                continue
            else:
                cWord = re.sub('[aeiou]', '', dict_word)
                #print ("cWord: " + cWord)
                const_dictionary.append(cWord)

        c = self.getCounterFromDict(const_dictionary)
        sorted_const_count = c.most_common(iterationCount)
        
        for letter, instance_count in sorted_const_count:
            guess_letter = letter
            #print ("First Guess Consonant: ", guess_letter)
        
        return guess_letter        
        
    def guess(self, word, n_word_dictionary):
        
        # Check if fallback is needed
        #if self.consecutive_wrong_guesses >= 3:
            #return self.fallback_guess()
        
        clean_word = word[::2].replace("_", ".")
        original_guess = self._original_guess(word, n_word_dictionary)
        bert_guess = self._bert_guess(clean_word)
        pattern_guess = self._pattern_based_guess(clean_word)

        # Calculate the revealed ratio
        revealed_ratio = (len(clean_word) - clean_word.count('.')) / len(clean_word)
        
        word_length = len(clean_word)
        if word_length <= 7:
            # For shorter words, prioritize original and pattern-based guesses

            if revealed_ratio > 0.66:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0, 0)
            elif revealed_ratio > 0.4:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0, 0.5)
            else:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0, 0)
        else:
            # For longer words, prioritize BERT guesses
            if revealed_ratio > 0.89:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0.4, 0)
            if revealed_ratio > 0.66:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0.3, 0.1)
            elif revealed_ratio > 0.4:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0, 0)
            else:
                final_guess = self.combine_guesses(original_guess, bert_guess, pattern_guess, 1, 0, 0)

        
        return final_guess

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
        
        c = self.getCounterFromDict(new_dictionary)
        sorted_letter_count = c.most_common()
        
        
        
        if len(clean_word) <= 6:
            vowel_threshold = 0.55
        else:
            vowel_threshold = 0.52

        guess_letter = '!'
        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                if letter in "aeiou" and self.vowel_count(clean_word) > vowel_threshold:
                    self.guessed_letters.append(letter)
                    continue
                guess_letter = letter
                break

        if guess_letter == '!':
            new_dictionary = self.appendWordToDict(n_word_dictionary, clean_word)
            c = self.getCounterFromDict(new_dictionary)
            sorted_letter_count = c.most_common()
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    if letter in "aeiou" and self.vowel_count(clean_word) > vowel_threshold:
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
                    new_dictionary = self.appendWordToDict(n_word_dictionary, s)
                    temp = self.getCounterFromDict(new_dictionary)
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
                    if letter in "aeiou" and self.vowel_count(clean_word) > vowel_threshold:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break
                
        # Return the top 5 guesses instead of just one
        top_guesses = []
        for letter, _ in sorted_letter_count:
            if letter not in self.guessed_letters:
                top_guesses.append(letter)
                if len(top_guesses) == 5:
                    break
        return top_guesses
    
    
    def fallback_guess(self):
        # Get the list of letters already guessed
        guessed_set = set(self.guessed_letters)

        # Iterate through the sorted common letters to find the next one to guess
        for letter, _ in self.full_dictionary_common_letter_sorted:
            if letter not in guessed_set:
                self.consecutive_wrong_guesses = 0  # Reset the counter
                return letter
                
    def handle_guess_response(self, response):
        # This method processes the server response after a guess
        if response.get('status') == 'success':
            self.consecutive_wrong_guesses = 0  # Reset counter on a successful guess
        else:
            self.consecutive_wrong_guesses += 1  # Increment counter on a wrong guess            
    
    def _pattern_based_guess(self, clean_word):
        # Expanded common suffixes and prefixes for better pattern-based guessing
        common_suffixes = [
            'ing', 'ed', 'es', 'ion', 'ive', 'ly', 'ment', 'ness', 'ble',
            'ous', 'ful', 'ant', 'ent', 'al', 'er', 'ist', 'ship', 'rd',
            'ism', 'ity', 'acy', 'ry', 'y', 's', 'er', 'or', 'en', 'ment', 
            'less', 'ness', 'ish', 'like', 'hood', 'dom', 'al', 'nce', 'cy',
            'ence', 'tude', 'ship', 'th', 'ly', 'ic', 'ed', 'ium', 'gy'
        ]
        
        common_prefixes = [
            'un', 're', 'in', 'dis', 'pre', 'mis', 'non', 'im', 'ir', 
            'over', 'under', 'sub', 'inter', 'trans', 'super', 'anti', 
            'ex', 'pro', 'co', 'de', 'anti', 'counter', 'fore', 'self', 
            'bio', 'geo', 'micro', 'macro', 'multi', 'semi', 'sub'
        ]
        
        # Common inflections and roots
        common_roots = ['ject', 'duct', 'scrib', 'script', 'port', 'struct', 
                        'vent', 'fer', 'mit', 'spect', 'tract', 'cede', 
                        'cede', 'duce', 'form', 'sist', 'pose']
        
        # Potential letters based on suffixes, prefixes, and roots
        potential_letters = set()
        
        # Check for common suffixes
        for suffix in common_suffixes:
            if clean_word.endswith('.' * len(suffix)):
                potential_letters.update(set(suffix))
        
        # Check for common prefixes
        for prefix in common_prefixes:
            if clean_word.startswith('.' * len(prefix)):
                potential_letters.update(set(prefix))
        
        # Check for common roots in the middle of the word
        for root in common_roots:
            if '.' * (len(root) - 1) in clean_word:
                potential_letters.update(set(root))

        # Allow guessing letters based on common consonant/vowel combinations
        common_letter_combinations = ['th', 'ch', 'sh', 'ph', 'wh', 'qu', 'ng', 'ck']
        for combo in common_letter_combinations:
            if combo in clean_word:
                potential_letters.update(set(combo))

        # Return potential letters excluding already guessed letters, limited to 5 guesses
        return list(potential_letters - set(self.guessed_letters))[:5]



    def _bert_guess(self, clean_word):
        masked_word = clean_word.replace('.', '[MASK]')
        inputs = self.bert_tokenizer(masked_word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        mask_token_index = torch.where(inputs["input_ids"] == self.bert_tokenizer.mask_token_id)[1]
        mask_token_logits = outputs.logits[0, mask_token_index, :]
        
        # Add positional weighting
        position_weights = torch.linspace(1, 0.5, mask_token_logits.size(-1)).unsqueeze(0)
        weighted_logits = mask_token_logits * position_weights
        top_20_tokens = torch.topk(weighted_logits, 20, dim=1).indices[0].tolist()
        
        top_chars = []
        for token in top_20_tokens:
            char = self.bert_tokenizer.decode([token]).lower()
            if char.isalpha() and len(char) == 1 and char not in self.guessed_letters and char not in top_chars:
                top_chars.append(char)
                if len(top_chars) == 5:
                    break
    
        return top_chars

    def combine_guesses(self, original_guesses, bert_guesses, pattern_guesses, original_weight, bert_weight, pattern_weight):
        combined_scores = {}
        # Improved scoring system
        for i, guess in enumerate(original_guesses):
            combined_scores[guess] = combined_scores.get(guess, 0) + original_weight * (1 / (i + 1))
        
        for i, guess in enumerate(bert_guesses):
            combined_scores[guess] = combined_scores.get(guess, 0) + bert_weight * (1 / (i + 1))
        
        for i, guess in enumerate(pattern_guesses):
            combined_scores[guess] = combined_scores.get(guess, 0) + pattern_weight * (1 / (i + 1))
        
        return max(combined_scores, key=combined_scores.get)

    def start_game(self, game_num, practice=True, verbose=False):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            prev_tries_remains = tries_remains
            vowelIterCount = 1;
            consIterCount = 0;
            
            if verbose:
                print(f"Successfully started a new game! Game ID: {game_id}. # of tries remaining: {tries_remains}. Word: {word}.")
            
            clean_word = word[::2].replace("_", ".")
            
            while tries_remains > 0:
                
                if len(clean_word) <= 7:
                    if vowelIterCount != -1:
                        guess_letter = self.guessFirstVowel(word, self.n_word_dictionary, vowelIterCount)
                        #print ("Vowel Guessed = " + guess_letter)                
                    elif consIterCount != -1:
                        guess_letter = self.guessFirstConsonant(word, self.n_word_dictionary, consIterCount)  
                        #print ("Consonant Guessed = " + guess_letter)                    
                    else:
                        guess_letter = self.guess(word, self.n_word_dictionary)
                else:
                    guess_letter = self.guess(word, self.n_word_dictionary)

                
                if guess_letter in self.guessed_letters:
                    alternative_guesses = [g for g in self.guess(word, self.n_word_dictionary) if g not in self.guessed_letters]
                    guess_letter = alternative_guesses[0] if alternative_guesses else random.choice(string.ascii_lowercase)
                
                if verbose:
                    print(f"Guessing letter: {guess_letter}")
                
                self.guessed_letters.append(guess_letter)

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
                                
                if vowelIterCount != -1:
                    if (prev_tries_remains != tries_remains and vowelIterCount == 1): #failure for guessFirstVowel()
                        vowelIterCount = vowelIterCount + 1 # enable calling guessFirstVowel again
                    else:
                        vowelIterCount = -1 # Set to a value such that guessFirstVowel() does not get called.

                if vowelIterCount == -1 and consIterCount != -1: #First Vowel  done,
                    if prev_tries_remains == tries_remains and consIterCount == 0: #first consonant not yet started or matched
                        consIterCount = 1
                    elif prev_tries_remains == tries_remains and consIterCount == 1: #first consonant matched
                        consIterCount = -1 
                    elif prev_tries_remains != tries_remains: #First consonant failed
                        if consIterCount == 2: #Max 2 tries
                            consIterCount = -1
                        else:
                            consIterCount = consIterCount + 1 # enable calling guessFirstConsonant again
                    else:
                        consIterCount = -1 # First consonant is done. Set to a value such that guessFirstConsonant() does not get called.
                                
                prev_tries_remains = tries_remains 
                
                if res.get('status') == "success":
                    if verbose:
                        print(f"Successfully finished game: {game_id}")
                    
                    print (f'Game {game_num}: Won: ',  len(clean_word))
                    print (f'Game {game_num}: Won: Length of Word = ',  len(clean_word), file=sys.stderr)
                    
                    return 0
                elif res.get('status') == "failed":
                    #reason = res.get('reason', "Unknown error")
                    reason = res.get('reason')
                    if reason and reason == "game terminated by server":
                        print (f'Game {game_num}: Terminated by Server', file=sys.stderr)
                        return -1
                    else:
                        print (f'Game {game_num}: Lost: ',  len(clean_word))
                        print (f'Game {game_num}: Lost: Length of Word = ',  len(clean_word), file=sys.stderr)
                        return 1
                    #if verbose:
                    #    print(f"Game {game_id} failed. Reason: {reason}")
                    #return 0
                else:
                    word = next_word

        else:
            print(f"Error starting a new game: {response}")
        return -1

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




# Script to use the HangmanAPI with loaded model
# Script to use the HangmanAPI with loaded model
def play_hangman_with_loaded_model(num_games, practice):
    
    print("Start Time = ", datetime.now(), file=sys.stderr)
    
    # Initialize the HangmanAPI (it will load the model by default)
    api = HangmanAPI(access_token="941fb10466d619f54bca7319df0a2b", timeout=5000)

    wins = 0
    losses = 0

    print("Initializations Complete, Time = ", datetime.now(), file=sys.stderr)
    
    for game_num in range(1, num_games + 1):
        print(f"Starting game {game_num}", file=sys.stderr)

        result = api.start_game(game_num, practice=practice)

        if result==0:
            wins += 1
            #print(f"Game {game_num}: Won")
        elif result==1:
            losses += 1
            #print(f"Game {game_num}: Lost")

        print(f"Current stats - Wins: {wins}, Losses: {losses}, Win rate: {wins / game_num:.2%}", file=sys.stderr)

        # Optional: Add a small delay to avoid overwhelming the API
        time.sleep(0.5)


    print("\nGames complete!", file=sys.stderr)
    print(f"Final stats - Wins: {wins}, Losses: {losses}, Win rate: {wins / num_games:.2%}", file=sys.stderr)

    print("End Time = ", datetime.now(), file=sys.stderr)
    
# Play 100 games with the loaded model`
play_hangman_with_loaded_model(num_games=50, practice=0)
