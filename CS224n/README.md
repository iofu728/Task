# CS224n: Natural Language Processing with Deep Learning

**`(Stanford / Winter 2020) version`**

_Instructor_: **Christopher Manning**

20 Lectures + 5 Assignments + 1 Final Project

lecture website: https://web.stanford.edu/class/cs224n/

> This repository holds handwritten notes, formula derivations, and thoughts, about the course.

## Lectures

| Public Date | Study Date | Description                                                              | Other                                                                                       |
| ----------- | ---------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| Jan 7       | Feb 18     | Introduction and Word Vectors<br>[[slide]][1] [[note]][10001]            | derivations <br> CBOW, SG, NegativeSample,<br> Hierarchical Softmax <br> in [[note]][10001] |
| Jan 9       | Feb 19     | Word Vectors 2 and Word Senses<br>[[slide]][2] [[note]][10002]           | Glove combine co-ocurence Matix and window base                                             |
| Jan 14      | Feb 20     | Word Win Classification, Neural Networks<br>[[slide]][3] [[note]][10003] | FP, BP, nn-linear func, optimization                                                        |
| Jan 16      | Feb 20     | Matrix Calculus and Backpropagation<br>[[slide]][4]                      | Gradient(BP)                                                                                |
| Jan 21      | Feb 21     | Linguistic Structure: Dependency Parsing<br>[[slide]][5] [[note]][10005] | Dependency Parsing which nested is common                                                   |
| Jan 23      | Feb 22     | The probability of a sentence? RNN & LM<br>[[slide]][6] [[note]][10006]  | RNN, Gated Structure                                                                        |
| Jan 28      | Feb 22     | Vanishing Gradients and Fancy RNNs<br>[[slide]][7]                       | Gated, stack, bidirectional                                                                 |
| Jan 30      | Feb 24     | Machine Translation, Seq2Seq & Attention<br>[[slide]][8] [[note]][10008] | MT, Seq2Seq                                                                                 |
| Feb 4       | Feb 26     | Practical Tips for Final Projects<br>[[slide]][9] [[note]][10009]        | Research ideas                                                                              |
| Feb 6       | Feb 27     | QA, Transformer architectures<br>[[slide]][10] [[note]][10010]           | QA, Attention, start + end                                                                  |
| Feb 11      | Feb 28     | ConvNets for NLP<br>[[slide]][11] [[note]][10011]                        | CNN                                                                                         |
| Feb 13      | Feb 29     | Information from parts of words (Subword Models)<br>[[slide]][12]        | Sub-word, BPE, Character-level, morphemes                                                   |
| Feb 18      | Mar 1      | Contextual Word Representations: BERT <br>[[slide]][13]                  | Pre-BERT & BERT & Post-BERT                                                                 |

## Assignments

**`The submission latex template in`** [**[Assignment2]**][21002]

| Due Date | Submission Date | HW                                                                                        | Topic                                   |
| -------- | --------------- | ----------------------------------------------------------------------------------------- | --------------------------------------- |
| Jan 14   | Feb 19          | Assignment 1 (6%): Introduction to word vectors<br>[[HW]][20001]                          | Co-Occurrence, SVD, GenSim              |
| Jan 21   | Feb 27          | Assignment 2 (12%): Derivatives and implementation of word2vec algorithm<br>[[HW]][20002] | naive-softmax, neg-sample, SG, Gradient |

## Course Material

| lecture | Read Date | Name                                                                                                                             | Other                       |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 1       | -         | Word2Vec Tutorial - The Skip-Gram Model                                                                                          | -                           |
|         |           | Efficient Estimation of Word Representations in Vector Space (original word2vec paper)                                           | -                           |
|         |           | Distributed Representations of Words and Phrases and their Compositionality (negative sampling paper)                            | -                           |
| 2       |           | GloVe: Global Vectors for Word Representation (original GloVe paper)                                                             | -                           |
|         |           | Improving Distributional Similarity with Lessons Learned from Word Embeddings                                                    | -                           |
|         |           | Evaluation methods for unsupervised word embeddings                                                                              | -                           |
|         |           | A Latent Variable Model Approach to PMI-based Word Embeddings                                                                    | -                           |
|         |           | Linear Algebraic Structure of Word Senses, with Applications to Polysemy                                                         | -                           |
|         |           | On the Dimensionality of Word Embedding.                                                                                         | -                           |
| 3       |           | Review of differential calculus                                                                                                  | -                           |
|         |           | Natural Language Processing (Almost) from Scratch                                                                                | -                           |
|         | Feb 27    | [Adam: A Method for Stochastic Optimization][30001]                                                                              | momentum + squared gradient |
| 4       |           | Learning Representations by Backpropagating Errors                                                                               | -                           |
|         |           | Derivatives, Backpropagation, and Vectorization                                                                                  | -                           |
|         |           | Yes you should understand backprop                                                                                               | -                           |
| 5       |           | Incrementality in Deterministic Dependency Parsing                                                                               | -                           |
|         |           | A Fast and Accurate Dependency Parser using Neural Networks                                                                      | -                           |
|         |           | Dependency Parsing                                                                                                               | -                           |
|         |           | Globally Normalized Transition-Based Neural Networks                                                                             | -                           |
|         |           | Universal Stanford Dependencies: A cross-linguistic typology                                                                     | -                           |
|         |           | Universal Dependencies website                                                                                                   | -                           |
| 6       |           | N-gram Language Models (textbook chapter)                                                                                        | -                           |
|         |           | The Unreasonable Effectiveness of Recurrent Neural Networks (blog post overview)                                                 | -                           |
|         |           | Sequence Modeling: Recurrent and Recursive Neural Nets (Sections 10.1 and 10.2)                                                  | -                           |
|         |           | On Chomsky and the Two Cultures of Statistical Learning                                                                          | -                           |
| 7       |           | Sequence Modeling: Recurrent and Recursive Neural Nets (Sections 10.3, 10.5, 10.7-10.12)                                         | -                           |
|         |           | Learning long-term dependencies with gradient descent is difficult (one of the original vanishing gradient papers)               | -                           |
|         |           | On the difficulty of training Recurrent Neural Networks (proof of vanishing gradient problem)                                    | -                           |
|         |           | Vanishing Gradients Jupyter Notebook (demo for feedforward networks)                                                             | -                           |
|         |           | Understanding LSTM Networks (blog post overview)                                                                                 | -                           |
| 8       |           | Statistical Machine Translation slides, CS224n 2015 (lectures 2/3/4)                                                             | -                           |
|         |           | Statistical Machine Translation (book by Philipp Koehn)                                                                          | -                           |
|         |           | BLEU (original paper)                                                                                                            | -                           |
|         |           | Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)                                                  | -                           |
|         |           | Sequence Transduction with Recurrent Neural Networks (early seq2seq speech recognition paper)                                    | -                           |
|         |           | Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)                         | -                           |
|         |           | Attention and Augmented Recurrent Neural Networks (blog post overview)                                                           | -                           |
|         |           | Massive Exploration of Neural Machine Translation Architectures (practical advice for hyperparameter choices)                    | -                           |
| 9       |           | Practical Methodology (Deep Learning book chapter)                                                                               | -                           |
| 10      |           | Attention Is All You Need                                                                                                        | -                           |
|         |           | Layer Normalization                                                                                                              | -                           |
|         |           | Image Transformer                                                                                                                | -                           |
|         |           | Music Transformer: Generating music with long-term structure                                                                     | -                           |
| 11      |           | Convolutional Neural Networks for Sentence Classification                                                                        | -                           |
|         |           | Improving neural networks by preventing co-adaptation of feature detectors                                                       | -                           |
|         |           | A Convolutional Neural Network for Modelling Sentences                                                                           | -                           |
| 12      |           | Minh-Thang Luong and Christopher Manning. Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models | -                           |
|         |           | Revisiting Character-Based Neural Machine Translation with Capacity and Compression                                              | -                           |
|         |           | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding                                                 | -                           |

## License

[Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.en)

Copyright (c) 2019-present, gunjianpan(iofu728)

[1]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture01-wordvecs1.pdf
[2]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture02-wordvecs2.pdf
[3]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture03-neuralnets.pdf
[4]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture04-neuralnets.pdf
[5]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture05-dep-parsing.pdf
[6]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture06-rnnlm.pdf
[7]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture07-fancy-rnn.pdf
[8]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture08-nmt.pdf
[9]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture09-final-projects.pdf
[10]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture10-QA.pdf
[11]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture11-convnets.pdf
[12]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/lecture12-subwords.pdf
[13]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/Jacob_Devlin_BERT.pdf
[10001]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes01-wordvecs1.pdf
[10002]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes02-wordvecs2.pdf
[10003]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes03-neuralnets.pdf
[10005]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes04-dependencyparsing.pdf
[10006]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes05-LM_RNN.pdf
[10008]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes06-NMT_seq2seq_attention.pdf
[10009]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/final-project-practical-tips.pdf
[10010]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes07-QA.pdf
[10011]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/notes08-CNN.pdf
[20001]: https://github.com/iofu728/Task/blob/develop/CS224n/assignment1/exploring_word_vectors.ipynb
[20002]: https://github.com/iofu728/Task/blob/develop/CS224n/assignment2/assignment2.pdf
[21002]: https://github.com/iofu728/Task/blob/develop/CS224n/assignment2/assignment2.tex
[30001]: https://github.com/iofu728/Task/blob/develop/CS224n/notes/papers/Adam.pdf
