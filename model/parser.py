import nltk
from typing import Set, List


def parse(grammar: nltk.grammar.CFG, sentence: List[str]) -> Set[nltk.ImmutableTree]:
    n = len(sentence)
    chart = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    back_pointers = [[{} for _ in range(n + 1)] for _ in range(n + 1)]

    for i, word in enumerate(sentence):
        # Find non-terminal symbols for each word
        chart[i][i + 1].update(prod.lhs() for prod in grammar.productions(rhs=word))

    for width in range(2, n + 1):
        for start in range(n - width + 1):
            end = start + width
            for mid in range(start + 1, end):
                for prod in grammar.productions():
                    if len(prod.rhs()) == 2 and prod.rhs()[0] in chart[start][mid] and prod.rhs()[1] in chart[mid][end]:
                        chart[start][end].add(prod.lhs())
                        back_pointers[start][end].setdefault(prod.lhs(), []).append((mid, prod))

    def build_trees(start, end, symbol):
        if start + 1 == end:
            # Leaf node
            return {nltk.ImmutableTree(symbol, [sentence[start]])}
        else:
            trees = set()
            for mid, prod in back_pointers[start][end].get(symbol, []):
                left_trees = build_trees(start, mid, prod.rhs()[0])
                right_trees = build_trees(mid, end, prod.rhs()[1])
                trees.update(
                    nltk.ImmutableTree(prod.lhs(), [left, right]) for left in left_trees for right in right_trees)
            return trees

    return set(tree for symbol in chart[0][n] for tree in build_trees(0, n, symbol))


def count(grammar: nltk.grammar.CFG, sentence: List[str]) -> int:
    n = len(sentence)
    chart = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(n):
        # Find non-terminal symbols for each word
        productions = grammar.productions(rhs=sentence[i])
        for prod in productions:
            chart[i][i + 1] += 1

    for width in range(2, n + 1):
        for start in range(n - width + 1):
            end = start + width
            for mid in range(start + 1, end):
                for prod in grammar.productions():
                    if len(prod.rhs()) == 2:
                        chart[start][end] += chart[start][mid] * chart[mid][end]

    return chart[0][n]
