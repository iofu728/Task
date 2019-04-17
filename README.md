> 🤧Some tasks for competition by gunjianpan

## Navigation

| Name                                                                                                                       | Classify          | Type           | Method                    | Modify |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------------- | ------------------------- | ------ |
| [NLP cbb Task1: _Medicine Corpus processing_](#nlp-cbb-task1-medicine-corpus-processing)                                   | Linguistics       | Generate       | Jieba + Dict + Rules      | 190418 |
| [Semantic Task2: _SemEval2017-Task4-SentimentAnalysis_](#semantic-course-task2-semeval2017-task4-sentimentanalysis)        | Sentiment         | Classify       | TextCNN & Bert            | 190414 |
| [NLP senior Task1: _SemEval2018-Task7-RelationClassification_](#nlp-senior-task1-semeval2018-task7-relationclassification) | Semantic Relation | Multi Classify | TextCNN & LR & LinearSV   | 190411 |
| [Concrete](#concrete)                                                                                                      | Feature           | Classify       | lightGBM                  | 190328 |
| [Semantic Course Task1](#semantic-course-task1)                                                                            | Word Similarity   | Regression     | word2vec & Bert & WordNet | 190324 |
| [Interview](#interview)                                                                                                    | Feature           | Classify       | lightGBM                  | 190318 |
| [Elo](#elo)                                                                                                                | Feature           | Classify       | lightGBM                  | 190304 |
| [Titanic](#titanic)                                                                                                        | Feature           | Classify       | lightGBM                  | 181220 |

## NLP cbb Task1: _Medicine Corpus processing_

- **Task**: Chinese word segmentation, part-of-speech tagging, named-entity recognition
- **Code**: [nlpcbb/task1](https://github.com/iofu728/Task/tree/master/nlpcbb/task1)
- **Final paper**: [nlpcbb/task1](https://github.com/iofu728/Task/tree/master/nlpcbb/task1)

## Semantic Course Task2: _SemEval2017-Task4-SentimentAnalysis_

- **Task**: [SemEval2017-Task4-SentimentAnalysis](http://alt.qcri.org/semeval2017/task4/)
- **Code**: [iofu728/SemEval2017-Task4-SentimentAnalysis](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis)
- **Final paper**: [Bert & TextCNN on SemEval-2017 Task4 subTask A: Sentiment Analysis in twitter](https://github.com/iofu728/SemEval2017-Task4-SentimentAnalysis/blob/master/paper/final_paper/main.pdf)
- **Result**: **69.39%**/Recall, **69.91%**/Macro-F1, **70.03%**/accuracy-subTaskA(No.1)

## NLP senior Task1: _SemEval2018-Task7-RelationClassification_

- **Task**: [SemEval-2018 task 7 Semantic Relation Extraction and Classification in Scientific Papers](https://competitions.codalab.org/competitions/17422)
- **Record**: [pku-nlp-forfun/SemEval-2018-RelationClassification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)
- **Code**: [pku-nlp-forfun/SemEval-2018-RelationClassification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)
- **Final paper**: [pku-nlp-forfun/SemEval2018-Task7-Paper](https://github.com/pku-nlp-forfun/SemEval2018-Task7-Paper/blob/master/main.pdf)
- **Result**: Macro_f1: **67.74%**/subTask1.1(No.8), **77.35%**/subTask1.2(No.7)

## Concrete

- **Task**: [Concrete Warning](https://www.datafountain.cn/competitions/336/details)
- **code**: [concrete](https://github.com/iofu728/Task/tree/master/cronete)

## Semantic Course Task1

- **Task**: _Word similarity_

  - Two way to calculate word similarity. one for dictionary, one for corpus.
  - Data: Mturk-771
  - [http://www2.mta.ac.il/~gideon/mturk771.html](http://www2.mta.ac.il/~gideon/mturk771.html)

- **Final result**: [semanticTask1](https://github.com/iofu728/STask/blob/master/semantic/task1/semanticTask1.pdf)
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

- **Final result**: [Code record](https://github.com/iofu728/STask/blob/master/interview/CodeTaskRecord.pdf)
- **Code**: [interview](https://github.com/iofu728/Task/tree/master/interview)
- **Result**: macro_f1: **84.707%**/Train, **82.429%**/Test

## Elo

- **Task**: [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation/leaderboard)
- **code**: [elo](https://github.com/iofu728/Task/tree/master/elo)

## Titanic

- **Task**: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)
- **code**: [titanic](https://github.com/iofu728/Task/tree/master/titanic)
