import pattern.en


class pos_tokenizer(object):
    
    def __init__(self, POS_blacklist):
        '''
        Uses pattern.en to remove POS terms on the blacklist
        '''

        self.parse = lambda x:pattern.en.parse(x,chunks=False,tags=True)

        # connectors = conjunction,determiner,infinitival to,
        #              interjection,predeterminer
        # w_word = which, what, who, whose, when, where & there ...

        POS = {
            "connector"   : ["CC","IN","DT","TO","UH","PDT"],
            "cardinal"    : ["CD","LS"],
            "adjective"   : ["JJ","JJR","JJS"],
            "noun"        : ["NN","NNS","NNP","NNPS"],
            "pronoun"     : ["PRP","PRP$"],
            "adverb"      : ["RB","RBR","RBS","RP"],
            "symbol"      : ["SYM",'$',],
            "punctuation" : [".",",",":",')','('],
            "modal_verb"  : ["MD"],
            "verb"        : ["VB","VBZ","VBP","VBD","VBG","VBN"],
            "w_word"      : ["WDT","WP","WP$","WRB","EX"],
            "unknown"     : ["FW", "``"],
        }

        self.filtered_POS = POS_blacklist

        '''
                set(("connector", 
                     "cardinal",
                     "pronoun",
                     "adverb",
                     "symbol",
                     "verb",
                     "punctuation",
                     "modal_verb",
                     "w_word",))
        '''

        self.POS_map = {}
        for pos,L in POS.items():
            for y in L: self.POS_map[y]=pos
                                        
    def __call__(self,doc,force_lemma=True):
        tokens = self.parse(doc)
        doc2 = []
        for sentence in tokens.split():
            sent2 = []
            for word,tag in sentence:

                if "PHRASE_" in word:
                    sent2.append(word)
                    continue

                tag = tag.split('|')[0].split('-')[0].split("&")[0]
                
                try:
                    pos = self.POS_map[tag]
                except:
                    print "UNKNOWN POS *{}*".format(tag)
                    pos = "unknown"

                if pos in self.filtered_POS:
                    continue

                org_word = word

                word = pattern.en.singularize(word,pos)

                if pos == "verb" or force_lemma:
                    lem = pattern.en.lemma(word,parse=False)
                    if lem is not None: word = lem

                sent2.append(word)
            doc2.append(' '.join(sent2))

        return '\n'.join(doc2)
