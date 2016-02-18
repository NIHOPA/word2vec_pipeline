import pattern.en

def sentence_tokenizer(raw):
    '''
    Uses pattern.en to split input text into a list of word tokens.
    '''
        
    raw_tokens = pattern.en.parse(raw,chunks=False,tags=False)
    raw_sentences  = raw_tokens.split()
    
    # Each token is now a list of elements, we only need the first one
    sentences = [[w[0] for w in s] for s in raw_sentences]
    return sentences

def word_tokenizer(raw):
    '''
    Uses pattern.en to split input text into a list of word tokens.
    '''
    sentences = sentence_tokenizer(raw)

    # Return a list of word tokens
    tokens = [w for s in sentences for w in s]
    
    return tokens