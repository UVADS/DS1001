import numpy as np
from collections import defaultdict
import math

# Load word list
def load_words():
    """
    Load words from wordlist.txt file
    """
    try:
        with open('code/wordlist.txt', 'r') as file:
            return [word.strip().lower() for word in file if len(word.strip()) == 5]
    except FileNotFoundError:
        print("Warning: wordlist.txt not found. Using default word list.")
        return [
            "stare", "crane", "trace", "slate", "crate",
            "about", "above", "abuse", "actor", "acute"
        ]

def get_pattern(guess, answer):
    """
    Returns the Wordle pattern as a string of 0,1,2
    0 = grey (letter not in word)
    1 = yellow (letter in word, wrong position)
    2 = green (letter in word, correct position)
    """
    pattern = ['0'] * 5
    letter_counts = defaultdict(int)
    
    # Count letters in answer
    for letter in answer:
        letter_counts[letter] += 1
    
    # First pass: mark correct positions (green)
    for i in range(5):
        if guess[i] == answer[i]:
            pattern[i] = '2'
            letter_counts[guess[i]] -= 1
    
    # Second pass: mark wrong positions (yellow)
    for i in range(5):
        if pattern[i] == '0' and guess[i] in answer and letter_counts[guess[i]] > 0:
            pattern[i] = '1'
            letter_counts[guess[i]] -= 1
    
    return ''.join(pattern)

def calculate_entropy(guess, possible_answers):
    """
    Calculate the expected information gain (entropy) for a guess
    """
    pattern_counts = defaultdict(int)
    total = len(possible_answers)
    
    # Count how many times each pattern appears
    for answer in possible_answers:
        pattern = get_pattern(guess, answer)
        pattern_counts[pattern] += 1
    
    # Calculate entropy using the formula: -Σ p(x) * log2(p(x))
    entropy = 0
    for count in pattern_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    
    return entropy

def get_best_guess(possible_words, possible_answers):
    """
    Returns the word that maximizes expected information gain
    """
    best_entropy = -1
    best_word = None
    
    for word in possible_words:
        entropy = calculate_entropy(word, possible_answers)
        if entropy > best_entropy:
            best_entropy = entropy
            best_word = word
    
    return best_word, best_entropy

def filter_words(words, guess, pattern):
    """
    Filter words based on the feedback pattern received
    """
    return [word for word in words if get_pattern(guess, word) == pattern]

def main():
    all_words = load_words()
    possible_answers = all_words.copy()
    
    while len(possible_answers) > 1:
        best_guess, entropy = get_best_guess(all_words, possible_answers)
        print(f"\nBest guess: {best_guess} (Information gain: {entropy:.2f} bits)")
        print(f"Possible answers remaining: {len(possible_answers)}")
        
        # Get feedback from user
        pattern = input("Enter pattern (0=grey, 1=yellow, 2=green): ")
        if pattern == "22222":  # Correct guess
            print("Solved!")
            break
            
        # Filter possible answers based on feedback
        possible_answers = filter_words(possible_answers, best_guess, pattern)
        
    if len(possible_answers) == 1:
        print(f"\nThe answer must be: {possible_answers[0]}")
    elif len(possible_answers) == 0:
        print("\nNo valid words remain. There might be an error in the pattern input.")

if __name__ == "__main__":
    main()
