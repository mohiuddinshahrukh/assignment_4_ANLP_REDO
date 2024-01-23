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

        grammatical = ["what is the cheapest ticket from memphis to miami ."]
        ungrammatical = ["how much does flight number a nineteen cost from new york to los angeles on monday morning ."]

        for sents in grammatical:
            val = recognize(grammar, nltk.word_tokenize(sents))
            print('\nGrammatical sentence: ')
            if val:
                print("{} is in the language of CFG.".format(sents))
            else:
                print("{} is not in the language of CFG.".format(sents))

        for sents in ungrammatical:
            val = recognize(grammar, nltk.word_tokenize(sents))
            print('\nUngrammatical sentence: ')
            if val:
                print("{} is in the language of CFG.".format(sents))
            else:
                print("{} is not in the language of CFG.".format(sents))

    elif args.parser:
        # Extend CKY recognizer into a parser in model/parser.py
        # Provide the list of ATIS test sentences with tab-separated numbers of parse trees
        # Choose an ATIS test sentence with a number of parses p such that 1 < p < 5 and provide pictures of its parses

        selected_sentence = t[1][0]
        print("selected_sentence: ", selected_sentence)
        print("ID\t Predicted_Tree\tLabeled_Tree")
        for idx, sents in enumerate(t):
            tree = parse(grammar, sents[0])
            print("{}\t {}\t \t{}".format(idx, len(tree), sents[1]))

        # Choose an ATIS test sentence with a number of parses p such that 1 < p < 5
        selected_tree = parse(grammar, selected_sentence)

        print("\nSelected Sentence:")
        print(" ".join(selected_sentence))
        print("\nNumber of Parses: {}".format(len(selected_tree)))
        print("\nParse Trees:")
        for tree in selected_tree:
            print(tree)
            tree.draw()

    elif args.count:
        # Compute the number of parse trees for an entry A ∈ Ch(i, k) from the chart with backpointers
        # Use model/parser.py to compute the number of parse trees without actually computing the parse tree

        print("ID\t Predicted_Tree\tLabeled_Tree")
        for idx, sents in enumerate(t):
            num_tree = count(grammar, sents[0])
            print("{}\t {}\t \t{}".format(idx, num_tree, sents[1]))


if __name__ == "__main__":
    main()
