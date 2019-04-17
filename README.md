# Task by [`gunjianpan`](https://github.com/iofu728)

## Navigation

| Name                                                                                                                     | Classify                         | Type           | Method                              | Modify Data |
| ------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | -------------- | ----------------------------------- | ----------- |
| [NLP cbb Task1: Medicine Corpus processing](#nlp-cbb-task1-medicine-corpus-processing)                                   | Linguistics                      | Generate       | Jieba + Medicine Dict + Rules       | 2019-04-18  |
| [Semantic Course Task2: SemEval2017-Task4-SentimentAnalysis](#semantic-course-task2-semeval2017-task4-sentimentanalysis) | Sentiment Analysis               | Classify       | TextCNN & Bert                      | 2019-04-14  |
| [NLP senior Task1: SemEval2018-Task7-RelationClassification](#nlp-senior-task1-semeval2018-task7-relationclassification) | Semantic Relation Classification | Multi Classify | TextCNN & LR & LinearSV             | 2019-04-11  |
| [Concrete](#concrete)                                                                                                    | Feature Engineering              | Classify       | lightGBM                            | 2019-03-28  |
| [Semantic Course Task1](#semantic-course-task1)                                                                          | Word Similarity                  | Regression     | word2vec & bert_embedding & wordnet | 2019-03-24  |
| [Interview](#interview)                                                                                                  | Feature Engineering              | Classify       | lightGBM                            | 2019-03-18  |
| [Elo](#elo)                                                                                                              | Feature Engineering              | Classify       | lightGBM                            | 2019-03-04  |
| [Titanic](#titanic)                                                                                                      | Feature Engineering              | Classify       | lightGBM                            | 2018-12-20  |

## NLP cbb Task1: Medicine Corpus processing

- **Task**: Chinese word segmentation, part-of-speech tagging, named-entity recognition
- **Record**: [nlpcbb/task1/README.md](https://github.com/iofu728/Task/tree/master/nlpcbb/task1)
- **Final paper**: [nlpcbb/task1/README.md](https://github.com/iofu728/Task/tree/master/nlpcbb/task1)

## Semantic Course Task2: SemEval2017-Task4-SentimentAnalysis

- **Task**: [SemEval2017-Task4-SentimentAnalysis](http://alt.qcri.org/semeval2017/task4/)
- **Record**: [iofu728/SemEval2017-Task4-SentimentAnalysis](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis)
- **Code**: [iofu728/SemEval2017-Task4-SentimentAnalysis](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis)
- **Final paper**: [Bert & TextCNN on SemEval-2017 Task4 subTask A: Sentiment Analysis in twitter](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis/blob/master/paper/final_paper/main.pdf)
- **Result**: **69.39%**/Recall, **69.91%**/Macro-F1, **70.03%**/accuracy-subTaskA(No.1)

## NLP senior Task1: SemEval2018-Task7-RelationClassification

- **Task**: [SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers](https://competitions.codalab.org/competitions/17422)
- **Record**: [pku-nlp-forfun/SemEval-2018-RelationClassification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)
- **Code**: [pku-nlp-forfun/SemEval-2018-RelationClassification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)
- **Final paper**: [pku-nlp-forfun/SemEval2018-Task7-Paper](https://github.com/pku-nlp-forfun/SemEval2018-Task7-Paper/blob/master/main.pdf)
- **Result**: Macro_f1: **67.74%**/subTask1.1(No.8), **77.35%**/subTask1.2(No.7)

## Concrete

- **Task**: [Concrete Warning](https://www.datafountain.cn/competitions/336/details)
- **code**: [concrete/](https://github.com/iofu728/Task/tree/master/cronete)

## Semantic Course Task1

- **Task**: Word similarity

- Two way to calculate word simarity. one for dictionary, one for corpus.
- Data: Mturk-771
- [http://www2.mta.ac.il/~gideon/mturk771.html](http://www2.mta.ac.il/~gideon/mturk771.html)

- **Final result**: [semanticTask1](https://github.com/iofu728/STask/blob/master/semantic/task1/semanticTask1.pdf)
- **code**: [semantic/Task1/](https://github.com/iofu728/Task/tree/master/semantic/task1)

## Interview

- **Task**:

- Description: This is a very simple binary classification task, you can design your model in any way you want.
- Evaluation: AUC, area under the ROC curve
- Data: The data file are provided in your answer submission folder in SharePoint (the link shown above).
- train.csv: used to train model.
- test.csv: used to metric your model's performance.
- Feedback / Submission:
- Your AUC score at the test dataSet.
- A brief description of your model and feature engineering.
- Your code.

- **Final result**: [Code record](https://github.com/iofu728/STask/blob/master/interview/CodeTaskRecord.pdf)
- **Code**: [titanic/](https://github.com/iofu728/Task/tree/master/interview)
- **Result**: macro_f1: **84.707%**/Train, **82.429%**/Test

## Elo

- **Task**: [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation/leaderboard)
- **code**: [elo/](https://github.com/iofu728/Task/tree/master/elo)

## Titanic

- **Task**: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)
- **code**: [titanic/](https://github.com/iofu728/Task/tree/master/titanic)
