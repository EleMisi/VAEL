from problog.logic import Term, Constant
from problog.logic import Var, AnnotatedDisjunction


def create_facts(sequence_len, n_digits=10):
    """
    Return the list of ADs necessary to describe an image with 'sequence_len' digits.
    'n_facts' specifies how many digits we are considering (i.e. n_facts = 2 means that the images can contain only 0 or 1)
    """

    ad = []  # Empty list to store the ADs
    for i in range(sequence_len):
        pos = i + 1
        annot_disj = ""  # Empty string to store the current AD facts

        # Build the AD
        digit = Term('digit')
        X = Var('X')
        facts = [digit(X, Constant(pos), Constant(y), p='p_' + str(pos) + str(y)) for y in range(n_digits)]
        annot_disj += str(AnnotatedDisjunction(facts, None)) + '.'

        ad.append(annot_disj)

    return ad


def define_ProbLog_model(facts, rules, label, digit_query=None, mode='query'):
    """Build the ProbLog model using teh given facts, rules, evidence and query."""
    model = ""  # Empty program

    # Insert annotated disjuctions
    for i in range(len(facts)):
        model += "\n\n% Digit in position " + str(i + 1) + "\n\n"
        model += facts[i]

    # Insert rules
    model += "\n\n% Rules\n"
    model += rules

    # Insert digit query
    if digit_query:
        model += "\n\n% Digit Query\n"
        model += "query(" + digit_query + ")."

    # Insert addition query
    if mode == 'query':
        model += "\n\n% Addition Query\n"
        model += "query(addition(img," + str(label) + "))."

    elif mode == 'evidence':
        model += "\n\n% Addition Evidence\n"
        model += "evidence(addition(img," + str(label) + "))."

    return model
