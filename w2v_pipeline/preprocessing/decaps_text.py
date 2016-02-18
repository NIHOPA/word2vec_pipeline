from tokenizers import sentence_tokenizer

class decaps_text(object):

    def diffn(self,s1,s2):
        return len([a for a,b in zip(s1,s2) if a!=b])
    
    def __init__(self):
        pass

    def modify_word(self,org):
        
        lower = org.lower()
        
        if self.diffn(org,lower) > 1:
            return org
        else:
            return lower
                                        
    def __call__(self,doc):

        sentences = sentence_tokenizer(doc)

        doc2 = []

        for sent in sentences:
            
            sent = [self.modify_word(w) for w in sent]
            doc2.append(' '.join(sent))

        doc2 = '\n'.join(doc2)

        return doc2
