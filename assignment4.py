import argparse
import nltk
from nltk.tree import Tree
from model.recognizer import recognize
from model.parser import parse, count

GRAMMAR_PATH = './data/atis-grammar-cnf.cfg'


def main():
    parser = argparse.ArgumentParser(
        description='CKY algorithm'
    )

    parser.add_argument(
        '--structural', dest='structural',
        help='Derive sentences with structural ambiguity',
        action='store_true'
    )

    parser.add_argument(
        '--recognizer', dest='recognizer',
        help='Execute CKY for word recognition',
        action='store_true'
    )

    parser.add_argument(
        '--parser', dest='parser',
        help='Execute CKY for parsing',
        action='store_true'
    )

    parser.add_argument(
        '--count', dest='count',
        help='Compute number of parse trees from chart without \
              actually computing the trees (Extra Credit)',
        action='store_true'
    )

    args = parser.parse_args()

    # load the grammar
    grammar = nltk.data.load(GRAMMAR_PATH)
    # load the raw sentences
    s = nltk.data.load("grammars/large_grammars/atis_sentences.txt", "auto")
    # extract the test sentences
    t = nltk.parse.util.extract_test_sentences(s)

    if args.structural:
        # Devise at least two sentences that exhibit structural ambiguity
        # Print the syntactic trees
        tree1 = Tree.fromstring('(S (NP (Time)) (VP (flies (PP (like (NP (an arrow)))))))')
        tree2 = Tree.fromstring('(S (NP (Time flies)) (PP (like (NP (an arrow)))))')

        tree3 = Tree.fromstring('(S (NP (I)) (VP (saw (NP (the man) (PP (with (NP (the telescope))))))))')
        tree4 = Tree.fromstring('(S (NP (I)) (VP (saw (NP (the man))) (PP (with (NP (the telescope))))))')

        print("Structural Ambiguity Example 1.1:")
        tree1.draw()
        print("Structural Ambiguity Example 1.2:")
        tree2.draw()
        print("\nStructural Ambiguity Example 2.1:")
        tree3.draw()
        print("\nStructural Ambiguity Example 2.2:")
        tree4.draw()

    elif args.recognizer:
        # Implement the CKY algorithm in model/recognizer.py and use it as a recognizer
        # Provide a list of grammatical and ungrammatical sentences and test the recognizer

        grammatical_sentences = ["what is the cheapest ticket from memphis to miami .",
                                 "please book a one way coach fare from chicago to indianapolis on united flight two ninety two next wednesday .",
                                 "how much does flight number a nineteen cost from new york to los angeles on monday morning .",
                                 "what kind of aircraft is american 's flight fifteen oh one that departs at six fifty nine p.m .",
                                 "how much does first class on that flight cost and how much does coach on that flight cost .",
                                 "please list the flights leaving newark stopping over in seattle for approximately five hours and then on to san francisco .",
                                 "i 'd like the cheapest round trip ticket from minneapolis to san diego arriving in san diego before seven p.m .",
                                 "what is the cheapest first class ticket available flying from new york to miami one way but also round trip .",
                                 "please give me the number of flights that depart from lester b pearson international airport on april twenty fifth .",
                                 "what is the cheapest one way flight from phoenix to san diego that arrives in the morning on thursday june second .",
                                 "i 'd like to fly from indianapolis to houston on t w a and the plane should arrive around eleven a.m ."]
        ungrammatical_sentences = [
            "prices .",
            "what is e w r .",
            "i 'd like an afternoon flight .",
            "i 'd like to leave on a saturday .",
            "list u s air flights to dallas to boston .",
            "i 'd like to fly out of chicago for less than a hundred fifty dollars round trip .",
            "show me flights from detroit to san diego on tuesday may third .",
            "show me flights from chicago to kansas city leaving around seven p.m. thursday .",
            "i would like to leave on thursday morning may fifth before six a.m .",
            "i need a first class round trip airfare from detroit to saint petersburg .",
            "how far is the airport from new york 's la guardia to downtown ."]

        print("Testing Grammatical Sentences:")

        for sent in grammatical_sentences:
            result = recognize(grammar, nltk.word_tokenize(sent))
            print(f"{sent}: {result}")

        print("\nTesting Ungrammatical Sentences:")
        for sent in ungrammatical_sentences:
            result = recognize(grammar, nltk.word_tokenize(sent))
            print(f"{sent}: {result}")
    elif args.parser:
        # Extend CKY recognizer into a parser in model/parser.py
        # Provide the list of ATIS test sentences with tab-separated numbers of parse trees
        # Choose an ATIS test sentence with a number of parses p such that 1 < p < 5 and provide pictures of its parses

        chosen_sent = t[1][0]
        print("Sentence: ", chosen_sent)
        print("S#\t\t Predicted\t\tLabeled")
        for idx, sents in enumerate(t):
            tree = parse(grammar, sents[0])
            print("{}\t {}\t \t{}".format(idx, len(tree), sents[1]))

        # Choose an ATIS test sentence with a number of parses p such that 1 < p < 5
        selected_tree = parse(grammar, chosen_sent)

        print("\nSelected Sentence:")
        print(" ".join(chosen_sent))
        print("\nNumber of Parses: {}".format(len(selected_tree)))
        print("\nParse Trees:")
        for tree in selected_tree:
            print(tree)
            tree.draw()

    elif args.count:
        # Compute the number of parse trees for an entry A ∈ Ch(i, k) from the chart with backpointers
        # Use model/parser.py to compute the number of parse trees without actually computing the parse tree

        print("S#\t\t Predicted\t\tLabeled")
        for idx, sents in enumerate(t):
            num_tree = count(grammar, sents[0])
            print("{}\t {}\t \t{}".format(idx, num_tree, sents[1]))


if __name__ == "__main__":
    main()
