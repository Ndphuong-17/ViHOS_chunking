import re
import numpy as np
# import more_itertools as mit
from itertools import groupby
from operator import itemgetter
import pandas as pd
from ast import literal_eval

def unicode(text):
    # Define the replacement mappings
    replacements = {
        "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
        "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
        "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
        "Ủy": "Uỷ"
    }
    
    # Define a function to apply the replacements to a single text
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
    
def dupplicate_punctuation(text, pos=[], spans=[]):
    if not pos:
        pos = list(range(len(text)))

    def replace(text, pattern, replacement, pos):
        matches = [0]  # Initialize a list to track the positions of matches.

        # Nested function to handle each regex match.
        def capture_and_replace(match, ret):
            matches.extend([match.start() + 1, match.end()])  # Store the start+1 and end positions of the match.
            return ret  # Return the replacement text for the match.

        # Get the length of the original text.
        l = len(text)

        # Use `re.sub` to find all occurrences of the pattern and replace them.
        # `capture_and_replace` is used as a callback to record match positions.
        text = re.sub(pattern, lambda match: capture_and_replace(match, replacement), text, flags=re.IGNORECASE)

        # Add the length of the modified text to the matches.
        matches.append(l)

        # Split the matches list into pairs of start and end positions.
        slices = np.array_split(matches, int(len(matches) / 2))

        # Adjust the `pos` list according to the changes made in the text.
        res = []
        for s in slices:
            res += pos[s[0]:s[1]]  # Extend `res` with the corresponding slice of `pos`.

        # Ensure the length of the updated `text` matches the updated `pos` list.
        assert len(text) == len(res)

        return text, res  # Return the updated text and the adjusted `pos` list.

    # Collapse duplicated punctuations 
    punc = ',. !?\"\''

    # Perform the replacement for each punctuation character.
    for c in punc:
        pat = f'([{c}]+)'
        text, pos = replace(text, pat, c, pos)

    # Adjust spans according to the new positions.
    new_spans = []
    for start, end in spans:
        new_start = pos.index(start) if start in pos else -1
        new_end = pos.index(end - 1) + 1 if (end - 1) in pos else -1
        if new_start != -1 and new_end != -1:
            new_spans.append([new_start, new_end])

    # Ensure that the length of `text` matches the updated `pos`.
    assert len(text) == len(pos)
    
    return text, pos, new_spans



def find_ranges(span):
    # Group consecutive numbers and create ranges.
    # ex: [0, 1, 3, 20, 21] --> [(0, 1), (3,3), (20, 21)]
    return [(group[0], group[-1]) for _, g in groupby(enumerate(span), lambda x: x[0] - x[1])
            for group in [list(map(itemgetter(1), g))]]

def load_data(path):
    tsd = pd.read_csv(path)
    tsd['spans'] = tsd['spans'].apply(literal_eval)

    data = []
    for _, row in tsd.iterrows():
        text, span = row['text'], row['spans']
        segments = find_ranges(span) if span else []
        temp = [[seg[0], seg[-1]] if len(seg) > 1 else [seg[0]] for seg in segments]
        text_spans = [text[seg[0]:seg[-1] + 1] for seg in segments]
        
        data.append({'text': text, 'spans': temp, 'text_spans': text_spans})
    
    return data

def sentence_based_chunking(text, pos, spans):
    # Split the text into sentences using basic punctuation marks.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunked_data = []
    start = 0

    for sentence in sentences:
        # Calculate the end position of the sentence in the original text.
        end = start + len(sentence)
        
        # Find spans that fall within the current sentence range and adjust them.
        adjusted_spans = []
        for span in spans:
            if span[0] >= start and span[1] <= end:
                # Adjust spans to be relative to the start of the current sentence.
                adjusted_spans.append([span[0] - start, span[1] - start])

        # Determine the tag based on whether there are any spans in this sentence.
        tag = 1 if adjusted_spans else 0

        # Append the chunked sentence with its spans and tag.
        chunked_data.append({
            'chunk': sentence,
            'Tag': tag,
            'spans': adjusted_spans
        })

        # Update start position for the next sentence.
        start = end + 1  # +1 to account for space after splitting.

    return chunked_data
