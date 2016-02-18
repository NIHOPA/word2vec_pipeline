import pyparsing as pypar

class remove_parenthesis(object):
    def __init__(self):
        nest = pypar.nestedExpr
        g = pypar.Forward()
        nestedParens   = nest('(', ')')
        nestedBrackets = nest('[', ']')
        nestedCurlies  = nest('{', '}')
        nest_grammar = nestedParens|nestedBrackets|nestedCurlies
        
        parens = "(){}[]"
        letters = ''.join([x for x in pypar.printables
                    if x not in parens])
        word = pypar.Word(letters)

        g = pypar.OneOrMore(word | nest_grammar)
        self.grammar = g

                                        
    def __call__(self,line):
        try:
            tokens = self.grammar.parseString(line)
        except (pypar.ParseException, RuntimeError):
            # On fail simply remove all parens
            line = line.replace('(','')
            line = line.replace(')','')
            line = line.replace('[','')
            line = line.replace(']','')
            line = line.replace('{','')
            line = line.replace('}','')
            tokens = line.split()

        # Remove nested parens
        tokens = [x for x in tokens if type(x) in [str,unicode]]
        text = ' '.join(tokens)
        return text

