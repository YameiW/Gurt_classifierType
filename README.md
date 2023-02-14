# Measure words are measurably different from sortal classifiers
Code and data corresponding to "Measure words are measurably different from classifiers" – GURT/SyntaxFest conference 2023.

Code available in python notebook.

## Pickle files description
There are three .pkl files with extracted nominal phrases, contextual embeddings of classifiers after dimentional reduction, and extracted nouns from all three corpora used in the study (Leipzig Corpora Collection).

## Python files
Three python files:
1. computation of the dimensionally reduced embeddings for each classifier based on contextual word embeddings extracted from a Chinese BERT model ('bert-base-chinese'),
2. functions used in the analysis,
3. results of manual validation.

## References:
- [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Chinese)
- [A Chinese BERT model](https://huggingface.co/bert-base-chinese)
