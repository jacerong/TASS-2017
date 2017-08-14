In order to compile the polarity lexicons into ``finite-state transducers``, the ``foma`` library [https://fomafst.github.io/] should be already installed. The library additionally allows to save the transducers as binary files.

Before getting into details of the compilation process, the lexicons are described below.

| id | Lexicon             | Positive words | Negative words | Total | Reference                                                                                                                                                                    |
|----|---------------------|----------------|----------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | AFINN               | 882            | 1,720          | 2,602 | F. Å. Nielsen (2011), "A new ANEW: Evaluation of a word list for sentiment analysis in microblogs".                                                                          |
| 2  | ElhPolar            | 1,474          | 2,668          | 4,142 | X. S. Urizar and I. S. V. Roncal, "Elhuyar at TASS 2013".                                                                                                                    |
| 3  | iSOL                | 2,509          | 5,624          | 8,133 | M. D. Molina-González and E. Martínez-Cámara and M. T. Martín-Valdivia and J. M. Perea-Ortega (2013), "Semantic orientation for polarity classification in Spanish reviews". |
| 4  | NRC Emotion Lexicon | 1,265          | 2,213          | 3,478 | S. Mohammad and P. Turney (2013), "Crowdsourcing a Word-Emotion Association Lexicon".                                                                                        |
| 5  | StrengthLex         | 477            | 870            | 1,347 | V. Pérez-Rosas and C. Banea and R. Mihalcea (2013), "Learning Sentiment Lexicons in Spanish".                                                                                |

On the other hand, other lexicons were built by joining two, three, four, or five of the above.

| id | Constituent lexicons |
|----|----------------------|
| 6  | 1,2                  |
| 7  | 1,3                  |
| 8  | 1,4                  |
| 9  | 1,5                  |
| 10 | 2,3                  |
| 11 | 2,4                  |
| 12 | 2,5                  |
| 13 | 3,4                  |
| 14 | 3,5                  |
| 15 | 4,5                  |
| 16 | 1,2,3                |
| 17 | 1,2,4                |
| 18 | 1,2,5                |
| 19 | 1,3,4                |
| 20 | 1,3,5                |
| 21 | 1,4,5                |
| 22 | 2,3,4                |
| 23 | 2,3,5                |
| 24 | 2,4,5                |
| 25 | 3,4,5                |
| 26 | 1,2,3,4              |
| 27 | 1,2,3,5              |
| 28 | 1,2,4,5              |
| 29 | 1,3,4,5              |
| 30 | 2,3,4,5              |
| 31 | 1,2,3,4,5            |

Given these points, it is proceeded to compile the lexicons into transducers.

```
$ cd /jacerong/datasets/lexicons/src/
$ /path/to/foma-0.9.18/foma
source compile_transducers.foma
exit
$ mv lexicon-*.bin /jacerong/datasets/lexicons/bin/
```
