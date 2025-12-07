# IPA Phonemizer: https://github.com/bootphon/phonemizer

import os
import string

_pad = "$"
_unknown = "<unk>"
_punctuation = (
    ";:,.!?¡¿—…\"«»“” "
    "،ˌˈ"
    "()" "[]" '{' '}'
)
_phonemes = (
    ".abdfhijklmnqrstuwz"
    "ðħɣɹʃʒʔʕ"
    "θχːˤ̪"
    "ɪɡɛəpɑʊɒɔeʌɜ"
    "ŋɐvoɾxɯʋʉʰ"
    "c-1yæɕɟɨɬɭɲɳɻʀʂʐ"
    "ʲ̩̃çʎʑɖʈɫ"
)

symbols = [_pad] + list(_punctuation) + list(_phonemes) + [_unknown]

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        unknowns = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                unknowns.append(char)
                indexes.append(self.word_index_dictionary['<unk>']) # unknown token

            with open("unknown_characters.txt", "a", encoding="utf-8") as f:
                for unk in set(unknowns):
                    f.write(unk + "\n")
        return indexes