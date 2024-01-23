import nltk
from typing import List


def recognize(grammar: nltk.grammar.CFG, sentence: List[str]) -> bool:
    n = len(sentence)
    chart = [[set() for _ in range(n + 1)] for _ in range(n + 1)]

    # Initialize the chart with non-terminal symbols for individual words
    for i, word in enumerate(sentence):
        productions = grammar.productions(rhs=word)
        chart[i][i + 1].update(prod.lhs() for prod in productions)

    # Fill in the chart using dynamic programming
    for width in range(2, n + 1):
        for start in range(n - width + 1):
            end = start + width
            for mid in range(start + 1, end):
                for prod in grammar.productions():
                    if len(prod.rhs()) == 2 and prod.rhs()[0] in chart[start][mid] and prod.rhs()[1] in chart[mid][end]:
                        chart[start][end].add(prod.lhs())

    # The start symbol of the grammar should be in the cell (0, n)
    return grammar.start() in chart[0][n]
