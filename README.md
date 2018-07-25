# Deep NLP Reading List
This serves as my own detailed roadmap and reading list/notes for studying Deep Learning and/with NLP. Each section will refer to useful materials that can help, including MOOCs, blog posts, books, lecture notes, papers, and other awesome paper lists and roadmaps.

## Table of Contents
0. [Mathematical Foundations](#mathematical-foundations)
    - [Basics](#basics)
    - [Advanced](#advanced)
1. [Machine Learning](#machine-learning)
2. [Deep Learning](#deep-learning)
3. [Statistical NLP](#statistical-nlp)
4. [Deep Learning for NLP](#deep-learning-for-nlp)
    - [Text Classification](#text-classification)
    - [Word Embeddings](#word-embeddings)
    - [Question Answering](#question-answering)
    - [End-to-End Dialog](#end-to-end-dialog)
    - [Neural Machine Translation](#neural-machine-translation)
    - [Multi-task Learning](#multi-task-learning)
    - [Memory Augmented](#memory-augmented)
    - [Meta Learning](#meta-learning)
    

## Mathematical Foundations
### Basics
[[Back To TOC]](#table-of-contents)

If you are confident in these math subjects, you can just skip this part or simply take a look at some refreshers.

- Linear Algebra
  - Refreshers 
    - Youtube Playlist: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
      - Approxmiately 2 hour long videos with Very Good Visualizations and clear explanations
    - Khan Academy [Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
  - MIT [Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/ax-b-and-the-four-subspaces/the-geometry-of-linear-equations/)
- Multivariable Calculus
  - Refreshers
    - Youtube Playlist: [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
    - [Highlights of Calculus](https://ocw.mit.edu/resources/res-18-005-highlights-of-calculus-spring-2010/): Video lectures by Prof. Gilbert Strang, MIT
    - Khan Academy [Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
  - MIT [Multivariable Calculus](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/)
- Probability and Statistics
  - Refreshers
    - [Deep Learning Book Chapter 3 - Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html)
    - Chapters 1, 2, and 11 of 'Pattern Recognition and Machine Learning' by Bishop (2006)
    - Khan Academy [Probability and Statistics](https://www.khanacademy.org/math/statistics-probability)
  - Harvard [STAT110](https://projects.iq.harvard.edu/stat110)
  - Readings
    - Chapters 2~6 of 'Machine Learning A Probabilistic Perspective' by Murphy (2012)
    - Lecture Notes: ['Probability and Statistics for Data Science'](http://www.cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf)

### Advanced
[[Back To TOC]](#table-of-contents)

The following subjects are some advanced materials that could be useful in understanding many Deep Learning theories and NLP. Particularly relevant ones are bolded.

- Information Theory
  - Refreshers
    - [Deep Learning Book Chapter 3 - Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html)
    - Khan Academy [Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory)
- Statistical Inference
  - CMU [Intermediate Statistics](http://www.stat.cmu.edu/~siva/705/main.html)
- Advanced Probability
- Random Matrix Theory
  - MIT [Eigenvalues of Random Matrices](http://web.mit.edu/18.338/www/index.html)
- Stochastic Processes
  - Coursera [Stochastic Processes](https://www.coursera.org/learn/stochasticprocesses)
  - MIT [Discrete Stocahstic Processes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-262-discrete-stochastic-processes-spring-2011/index.htm)
  - UIUC [Notes on Random Processes](http://hajek.ece.illinois.edu/Papers/randomprocJuly14.pdf)
- Opimization Theory
- Convex Optimization
- Vector Calculus
- Numerical Linear Algebra
  - fast.ai [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra): Focuses on applying what we learned from Linear Algebra to practical Data Science tasks.
- Abstract Algebra
- Real and Complex Analysis
- Theories of Deep Learning
  - Stanford [STATS385](https://stats385.github.io/)


## Machine Learning
[[Back To TOC]](#table-of-contents)

Machine Learning without Deep Learning.

- Refreshers
  - Kyunghyun Cho's [ML w/o DL Lecture Notes](https://github.com/nyu-dl/Intro_to_ML_Lecture_Note)
- Introductory
  - Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) on Coursera
  - Yaser Abu-Mostafa's [Learning From Data](https://work.caltech.edu/telecourse.html)
- Advanced
  - Tom Mitchell's [Machine Learning](http://www.cs.cmu.edu/~ninamf/courses/601sp15)
  - CMU [Intro to ML 10-701](http://www.cs.cmu.edu/~pradeepr/701/)
  - CMU [Advanced Intro to ML](https://www.cs.cmu.edu/~epxing/Class/10715/lecture.html)
- Readings
  - **'Pattern Recognition and Machine Learning' by Bishop (2006)**
  - 'Machine Learning A Probabilistic Perspective' by Murphy (2012)


## Deep Learning
[[Back To TOC]](#table-of-contents)

- Refreshers
  - Chapters 1, 2, 3, and 4 of Kyunghyun Cho's [Natural Language Understanding with Distributed Representation Lecture Notes](https://github.com/nyu-dl/NLP_DL_Lecture_Note)
  - Andrew Ng's Deep Learning courses [deeplearning.ai](https://www.deeplearning.ai/)
- CMU [Introduction to Deep Learning](http://deeplearning.cs.cmu.edu/)
- Stanford [CS231n Convolutional Neural Networks for Computer Vision](http://cs231n.stanford.edu/)
  - [Youtube Playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- Books
  - [Deep Learning Book](http://www.deeplearningbook.org) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
  - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Neilson
- Papers: Full list organized by topics and models can be found in [Deep-Learning-Papers-Reading-Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap), or Columbia's seminar course [Advanced Topics in Deep Learning - Reading List](http://www.advancedtopicsindeeplearning.com/reading-list.html)
  - Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." Nature 521.7553 (2015): 436-444 [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf): A high-level survey paper by the three giants
  - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner.  "Gradient-Based Learning Applied to Document Recognition."  Proceedings of the IEEE, 86(11):2278-2324. 1998 (Seminal Paper: LeNet) [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf): LeNet: Image Classification on Handwritten Digits
  - Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): Big hit of Deep Learning, AlexNet
- Blog Posts
  - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
  - WildML [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

  
## Statistical NLP
[[Back To TOC]](#table-of-contents)

- Refreshers
  - Chapters 6~9.3 of [Goldberg](http://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) book
- Columbia Michael Collins' COMS W4705: [Natural Language Processing](http://www.cs.columbia.edu/~mcollins/cs4705-fall2017/): This course covers a lot of traditional techniques often used in NLP.
  - [Notes on Statistical NLP](http://www.cs.columbia.edu/~mcollins/notes-spring2013.html)
  - Video Lectures on Coursera: Can't find the course anymore, but there are [Youtube videos](https://www.youtube.com/playlist?list=PL0ap34RKaADMjqjdSkWolD-W2VSCyRUQC)
- Chris Manning's [CS 224N/Ling 284 — Natural Language Processing](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml) before merging with Richard Socher's CS224D, covers some missing pieces of Michael Collins' class, along with more real life applications such as Machine Translation.
  - Video Lectures on [Youtube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- Readings
  - 'Foundations of Statistical Natural Language Processing' by Manning and Schütze (1999)
  - [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) drafts by Dan Jurafsky and James Martin

## Deep Learning for NLP
[[Back To TOC]](#table-of-contents)

Here I mainly organize papers I have read or plan to read. Among the ones I read, some accompany notes in a separate .md file linked.

- Overview
  - Goldberg, [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)
  - Kyunghyun Cho's Lecture Notes of [Natural Language Understanding with Distributed Representation Lecture Notes](https://github.com/nyu-dl/NLP_DL_Lecture_Note)
- Books
  - [Goldberg](http://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037)
- Courses
  - Stanford CS224N [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/syllabus.html)
    - The archived version for [2017 Winter Version](https://web.archive.org/web/20171221032823/http://web.stanford.edu/class/cs224n/syllabus.html)
    - [Youtube Playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
  - [Oxford Deep NLP](https://github.com/oxford-cs-deepnlp-2017/lectures)
    - [Youtube Playlist](https://www.youtube.com/playlist?list=PL613dYIGMXoZBtZhbyiBqb0QtgK6oJbpm)(Unofficial)
  - CMU CS 11-747 [Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/schedule.html)
    - Youtube Playlist](https://www.youtube.com/playlist?list=PL8PYTP1V4I8ABXzdqtOpB_eqBlVAz_xPT)

### Text Classification
[[Back To TOC]](#table-of-contents)

- Abusive Language
- Sentiment Analysis
 
### Word Embeddings
[[Back To TOC]](#table-of-contents)

- Language Modeling
- Contextualized Word Embeddings
- Probailistic Word Embeddings
- Interpretable Word Embeddings

### Question Answering
[[Back To TOC]](#table-of-contents)

- SQuAD 1.0 Models

### End-to-End Dialog
[[Back To TOC]](#table-of-contents)

- Goal-oriented
  - Dialog State Tracking
  - Latent Intents
  - Knowledge Base
  - Model Architectures
  - Datasets
  - Using RL
- Chit Chat


### Neural Machine Translation
[[Back To TOC]](#table-of-contents)

- Multi-linguality

### Multi-task Learning
[[Back To TOC]](#table-of-contents)

### Memory Augmented
[[Back To TOC]](#table-of-contents)

- Memory Networks
- Pointer Networks
- Neural Turing Machines

### Meta Learning
[[Back To TOC]](#table-of-contents)

- MAML
