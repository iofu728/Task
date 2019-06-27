> 🤧Some tasks for competition by gunjianpan

```console
❱ ❱❱ ❱❱❱ ❱❱❱❱ ❱❱❱❱ ❱❱❱❱❱ ❱❱❱❱❱❱ ❱❱❱❱❱❱❱
```

## Navigation

| Name                                                                                                                       | Classify                | Type           | Method                    | Modify |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------- | ------------------------- | ------ |
| [NLP senior Task3: _Chinese traditional Sequence Annotation_](#NLP-senior-Task3-Chinese-traditional-Sequence-Annotation)   | Sequence annotation     | Classify       | BiLSTM-CRF, Bert-CRF      | 190626 |
| [DataMining Project](#DataMining-Project)                                                                                  | Semantic Representation | Classify       | LSTM-ATT                  | 190620 |
| [NLP cbb Task2: _Medicine Corpus processing with Corpus_](#nlp-cbb-task2-medicine-corpus-processing-with-corpus)           | Sequence annotation     | Classify       | BiLSTM-CRF, CRF           | 190601 |
| [NLP senior Task2: _SemEval2013-Task13-WordSenseInduction_](#nlp-senior-task2-semeval2013-task13-wordsenseinduction)       | WordSenseInduction      | Cluster, LM    | BiLM + Clustering/WordNet | 190528 |
| [Semantic Course Task3: _NLPCC 2019 Task2_](#semantic-course-task3-nlpcc-2019-task2-open-domain-semantic-parsing)          | Semantic Parsing        | Generation     | Seq2Seq                   | 190520 |
| [Deecamp 2019 Exam A](#deecamp-2019-exam-a)                                                                                | Basic Knowledge         | Test           | -                         | 190427 |
| [NLP cbb Task1: _Medicine Corpus processing_](#nlp-cbb-task1-medicine-corpus-processing)                                   | Linguistics             | Generate       | Jieba + Dict + Rules      | 190418 |
| [Semantic Task2: _SemEval2017-Task4-SentimentAnalysis_](#semantic-course-task2-semeval2017-task4-sentimentanalysis)        | Sentiment               | Classify       | TextCNN & Bert            | 190414 |
| [NLP senior Task1: _SemEval2018-Task7-RelationClassification_](#nlp-senior-task1-semeval2018-task7-relationclassification) | Semantic Relation       | Multi Classify | TextCNN & LR & LinearSV   | 190411 |
| [Concrete](#concrete)                                                                                                      | Feature                 | Classify       | lightGBM                  | 190328 |
| [Semantic Course Task1](#semantic-course-task1)                                                                            | Word Similarity         | Regression     | word2vec & Bert & WordNet | 190324 |
| [Interview](#interview)                                                                                                    | Feature                 | Classify       | lightGBM                  | 190318 |
| [Elo](#elo)                                                                                                                | Feature                 | Classify       | lightGBM                  | 190304 |
| [Titanic](#titanic)                                                                                                        | Feature                 | Classify       | lightGBM                  | 181220 |

## NLP senior Task3: _Chinese traditional Sequence Annotation_

- **Task**: Chinese traditional Sequence Annotation, CWS & NER
- **Code**: [iofu728/Chinese_T-Sequence-annotation](https://github.com/iofu728/Chinese_T-Sequence-annotation)
- **Final paper**: [iofu728/Chinese_T-Sequence-annotation](https://github.com/iofu728/Chinese_T-Sequence-annotation/blob/master/final_paper/main.pdf)
- **Result**: CWS: **97.78%**/Dev F1; NER: **96.92%**/Dev F1

## DataMining Project

- **Task** Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder
- **Final paper** [dataMining](https://github.com/iofu728/Task/blob/develop/dataMining/dataMining.pdf)

## NLP cbb Task2: _Medicine Corpus processing with Corpus_

- **Task**: Chinese word segmentation, part-of-speech tagging, named-entity recognition
- **Code**: [pku-nlp-forfun/CWS_POS_NER](https://github.com/pku-nlp-forfun/CWS_POS_NER)
- **Final paper**: [pku-nlp-forfun/CWS_POS_NER](https://github.com/pku-nlp-forfun/CWS_POS_NER/blob/master/Report/main.pdf)
- **Result**: CWS: **90.55**/Dev F1, **91.16%**/Test F1; POS: **96.06**/Dev F1, **95.86%**/Test F1(Top 10)

## NLP senior Task2: _SemEval2013-Task13-WordSenseInduction_

- **Task**: [SemEval2013-Task13-WordSenseInduction](https://www.cs.york.ac.uk/semeval-2013/task13.html)

  - Word Sense Induction (WSI) seeks to identify the different senses (or uses) of a target word in a given text in an automatic and fully-unsupervised manner. It is a key-enabling technology that aims to overcome the limitations associated with traditional knowledge-based & supervised Word Sense Disambiguation (WSD) methods, such as:
  - their limited adaptation to new languages and domains
  - the fixed granularity of senses
  - their inability to detect new senses (uses) not present in a given dictionary

- **Code**: [pku-nlp-forfun/SemEval2013-WordSenseInduction](https://github.com/pku-nlp-forfun/SemEval2013-WordSenseInduction)
- **Final paper**: [Multi-fusion on SemEval-2013 Task13: Word Sense Induction](https://github.com/pku-nlp-forfun/SemEval2013-Task13-Paper/blob/master/main.pdf)
- **Result**: **11.06%**/Fuzzy NMI, **57.72%**/Fuzzy B-Cubed, **25.27%**/Average

## Semantic Course Task3: _NLPCC 2019 Task2 Open Domain Semantic Parsing_

- **Task**: [Senantic Parsing](http://tcci.ccf.org.cn/conference/2019/dldoc/taskgline02.pdf)

  - In this year’s NLPCC, we call for the Open Domain Semantic Parsing shared task. The goal of this task is to predict the correct logical form (in lambda-calculus) for each question in the test set, based on a given knowledge graph.
  - In this task, a Multi-perspective Semantic ParSing (or MSParS) dataset will be released, which can be used to evaluate the performance of a semantic parser from different aspects. This dataset includes more than 80,000 human-generated questions, where each question is annotated with entities, the question type and the corresponding logical form. We split MSParS into a train set, a development set and a test set. Both train and development sets will be provided to participating teams, while the test set will NOT. After participating teams submit their output files, we will evaluate their performances.

- **Code**: [semantic/task3](https://github.com/iofu728/Task/tree/develop/semantic/task3/semantic3)
- **Final paper**: [semantic/task3](https://github.com/iofu728/Task/blob/develop/semantic/task3/final_paper/main.pdf)
- - **Result**: BLEU 0.538

## DeeCamp 2019 Exam A

- **Problem**: [DeeCamp 2019 Exam A](https://github.com/iofu728/Task/blob/master/interview/deeCamp.md)

## NLP cbb Task1: _Medicine Corpus processing_

- **Task**: Chinese word segmentation, part-of-speech tagging, named-entity recognition in Rule
- **Code**: [nlpcbb/task1](https://github.com/iofu728/Task/tree/master/nlpcbb/task1)
- **Final paper**: [nlpcbb/task1](https://github.com/iofu728/Task/tree/master/nlpcbb/task1)

## Semantic Course Task2: _SemEval2017-Task4-SentimentAnalysis_

- **Task**: [SemEval2017-Task4-SentimentAnalysis](http://alt.qcri.org/semeval2017/task4/)

  - Subtask A. (rerun): Message Polarity Classification: Given a message, classify whether the message is of positive, negative, or neutral sentiment.

- **Code**: [iofu728/SemEval2017-Task4-SentimentAnalysis](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis)
- **Final paper**: [Bert & TextCNN on SemEval-2017 Task4 subTask A: Sentiment Analysis in twitter](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis/blob/master/paper/final_paper/main.pdf)
- **Result**: **69.39%**/Recall, **69.91%**/Macro-F1, **70.03%**/accuracy-subTaskA(No.1)

## NLP senior Task1: _SemEval2018-Task7-RelationClassification_

- **Task**: [SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers](https://competitions.codalab.org/competitions/17422)

  - One of the emerging trends of natural language technologies is their application to scientific literature. There is a constant increase in the production of scientific papers and experts are faced with an explosion of information that makes it difficult to have an overview of the state of the art in a given domain. Recent works from the semantic web, scientometry and natural language processing communities aimed to improve the access to scientific literature, in particular to answer queries that are currently beyond the capabilities of standard search engines. Examples of such queries include finding all papers that address a given problem in a specific way, or to discover the roots of a certain idea.
  - The NLP tasks that underlie intelligent processing of scientific documents are those of information extraction: identifying concepts and recognizing the semantic relation that holds between them. The current task adresses semantic relation extraction and classification into 6 categories, all of them specific to scientific literature.
  - Information extraction from corpora including relation extraction and classification normally involves a complicated multiple-step process. We provide a framework for evaluating systematically how the single steps affect the ultimate result. Correspondingly, the proposed tasks are split into subtasks that allow to measure the impact of entity annotation quality, the type of entities, and the extraction of relation instances (entity pairs) on relation classification. All of the tasks are centered around the classification of entity pairs in context into six different, non-overlapping categories of semantic relations that are defined in advance. Moreover, one of the subtasks involves the identification of relation instances.

- **Record**: [pku-nlp-forfun/SemEval-2018-RelationClassification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)
- **Code**: [pku-nlp-forfun/SemEval-2018-RelationClassification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)
- **Final paper**: [pku-nlp-forfun/SemEval2018-Task7-Paper](https://github.com/pku-nlp-forfun/SemEval2018-Task7-Paper/blob/master/main.pdf)
- **Result**: Macro_f1: **67.74%**/subTask1.1(No.8), **77.35%**/subTask1.2(No.7)

## Concrete

- **Task**: [Concrete Warning](https://www.datafountain.cn/competitions/336/details)
- **code**: [concrete](https://github.com/iofu728/Task/tree/master/concrete)

## Semantic Course Task1

- **Task**: _Word similarity_

  - Two way to calculate word similarity. one for dictionary, one for corpus.
  - Data: Mturk-771
  - [http://www2.mta.ac.il/~gideon/mturk771.html](http://www2.mta.ac.il/~gideon/mturk771.html)

- **Final result**: [semanticTask1](https://github.com/iofu728/Task/blob/master/semantic/task1/semanticTask1.pdf)
- **code**: [semantic/Task1](https://github.com/iofu728/Task/tree/master/semantic/task1)

## Interview

- **Task**:

  - Description: This is a very simple _binary classification task_, you can design your model in any way you want.
  - Evaluation: AUC, area under the ROC curve
  - Data: The data file are provided in your answer submission folder in SharePoint (the link shown above).
  - train.csv: used to train model.
  - test.csv: used to metric your model's performance.
  - Feedback / Submission:
  - Your AUC score at the test dataSet.
  - A brief description of your model and feature engineering.
  - Your code.

- **Final result**: [Code record](https://github.com/iofu728/Task/blob/master/interview/CodeTaskRecord.pdf)
- **Code**: [interview](https://github.com/iofu728/Task/tree/master/interview)
- **Result**: macro_f1: **84.707%**/Train, **82.429%**/Test

## Elo

- **Task**: [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation/leaderboard)
- **code**: [elo](https://github.com/iofu728/Task/tree/master/elo)

## Titanic

- **Task**: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)
- **code**: [titanic](https://github.com/iofu728/Task/tree/master/titanic)
