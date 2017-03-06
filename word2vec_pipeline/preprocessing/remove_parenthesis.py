import pyparsing as pypar
import pattern.en


class remove_parenthesis(object):

    def __init__(self):
        nest = pypar.nestedExpr
        g = pypar.Forward()
        nestedParens = nest('(', ')')
        nestedBrackets = nest('[', ']')
        nestedCurlies = nest('{', '}')
        nest_grammar = nestedParens | nestedBrackets | nestedCurlies

        parens = "(){}[]"
        letters = ''.join([x for x in pypar.printables
                           if x not in parens])
        word = pypar.Word(letters)

        g = pypar.OneOrMore(word | nest_grammar)
        self.grammar = g

        # self.parse = lambda x:pattern.en.parse(x,chunks=False,tags=False)
        self.parse = lambda x: pattern.en.tokenize(
            x)  # ,chunks=False,tags=False)

    def __call__(self, text):

        sentences = self.parse(text)

        doc_out = []
        for sent in sentences:

            # Count the number of left and right parens
            LP = sum(1 for a in sent if a == '(')
            RP = sum(1 for a in sent if a == ')')

            # If the count of the left paren doesn't match the right ignore
            FLAG_valid = (LP == RP)

            try:
                tokens = self.grammar.parseString(sent)
            except (pypar.ParseException, RuntimeError):
                FLAG_valid = False

            if not FLAG_valid:
                # On fail simply remove all parens
                sent = sent.replace('(', '')
                sent = sent.replace(')', '')
                sent = sent.replace('[', '')
                sent = sent.replace(']', '')
                sent = sent.replace('{', '')
                sent = sent.replace('}', '')
                tokens = sent.split()

            # Remove nested parens
            tokens = [x for x in tokens if type(x) in [str, unicode]]
            text = ' '.join(tokens)

            doc_out.append(text)

        return '\n'.join(doc_out)
