# Deep NLP Roadmap
This serves as my own detailed roadmap for studying Deep Learning and/with NLP. Each section will refer to useful materials that can help, including MOOCs, blog posts, books, lecture notes, papers, and other awesome paper lists and roadmaps.

## Table of Contents
0. [Mathematical Foundations](#mathematical-foundations)
    - [Basics](#basics)
    - [Advanced](#advanced)
1. [Machine Learning](#machine-learning)
2. [Deep Learning](#deep-learning)
3. [Statistical NLP](#statistical-nlp)
4. [Deep NLP](#deep-nlp)
    - [Neural Language Modeling](#neural-language-modeling-with-rnns)
    - [Embeddings](#embeddings)
    - [Text Classification](#text-classification)
    - [Text Generation](#text-generation)
    - [Multi-Lingual, Multi-Task Learning](#multi-lingual-multi-task-learning)
    - [Neural Turing Machines](#neural-turing-machines)
    - [Reinforcement Learning for NLP](#reinforcement-learning-for-nlp)
    

## Mathematical Foundations
### Basics
[[Back To TOC]](#table-of-contents)

If you are confident in these math subjects, you can just skip this part or simply take a look at some refreshers. Otherwise, it is recommended to follow at least the bold-faced materials for each subject.

- Linear Algebra
  - Refreshers 
    - **Youtube Playlist: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)**
      - Approxmiately 2 hour long videos with Very Good Visualizations and clear explanations
    - [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
  - **MIT Linear Algebra Course**: Very intuitive video lectures and materials by Prof. Gilbert Strang, MIT
    - [OCW Scholar Version](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/ax-b-and-the-four-subspaces/the-geometry-of-linear-equations/): Follow this syllabus, and the recitation videos
    - [Original Version](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/): Follow the readings and assignments here
- Multivariable Calculus
  - Refreshers
    - **Youtube Playlist: [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)**
    - **[Highlights of Calculus](https://ocw.mit.edu/resources/res-18-005-highlights-of-calculus-spring-2010/)**: Video lectures by Prof. Gilbert Strang, MIT
    - [Khan Academy Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
  - **MIT Multivariable Calculus Course**
    - [OCW Scholar Version](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/index.htm): Follow this syllabus, and the recitation videos
    - [Original Version](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/): Follow the readings and assignments here
- Probability and Statistics
  - Refreshers
    - [Deep Learning Book Chapter 3 - Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html)
    - Chapters 1, 2, and 11 of 'Pattern Recognition and Machine Learning' by Bishop (2006)
    - [Khan Academy Probability and Statistics](https://www.khanacademy.org/math/statistics-probability)
  - **[Harvard STAT110](https://projects.iq.harvard.edu/stat110)**: The infamous Joe Blitzstein's Introduction to probability and statistics. Despite being an introductory course, it covers quite a lot of the essentials we need to further understand the math in machine learning and deep learning.
  - Readings
    - Chapters 2~6 of 'Machine Learning A Probabilistic Perspective' by Murphy (2012)
    - Lecture Notes: ['Probability and Statistics for Data Science'](http://www.cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf) by Carlos Fernandez-Granda

### Advanced
[[Back To TOC]](#table-of-contents)

The following subjects are some advanced materials that could be useful in understanding many Deep Learning theories and NLP. Particularly relevant ones are bolded.

- **Information Theory**
  - Refreshers
    - [Deep Learning Book Chapter 3 - Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html)
    - [Khan Academy Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory)
- **Statistical Inference**
- **Advanced Probability**
- Opimization
- **Convex Optimization**
- **Vector Calculus**
- Numerical Linear Algebra
  - **[fast.ai Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra)**: Focuses on applying what we learned from Linear Algebra to practical Data Science tasks.
- Abstract Algebra
- Real and Complex Analysis
- Theories of Deep Learning
  - [Stanford STATS385](https://stats385.github.io/)


## Machine Learning
[[Back To TOC]](#table-of-contents)

In this context, Machine Learning refers to the conventional Machine Learning methods used in modeling various tasks before the prevailing advent of deep neural networks. These algorithms are still very often used in the industry due to their performance (in terms of resource consumption) and interpretability. Those who are already familiar with Undergraduate level (Introductory) Machine Learning courses are advised to go through the Graduate level (Advanced) courses. Those who are already very familiar with Machine Learning in general should simply watch the refreshers or skip this part. Advanced topics covered by advanced courses include Probabilistic Graphical Models, EM and clustering, Kernel Methods and SVMs, Reinforcement Learning, etc.

- Refreshers
  - Kyunghyun Cho's [ML w/o DL Lecture Notes](https://github.com/nyu-dl/Intro_to_ML_Lecture_Note)
  - [Deep Learning Book - Numerical Computations](http://www.deeplearningbook.org/contents/numerical.html)
  - [Deep Learning Book - Machine Learning Basics](http://www.deeplearningbook.org/contents/ml.html)
- Introductory
  - Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) on Coursera
  - A real Caltech course, Yaser Abu-Mostafa's [Learning From Data](https://work.caltech.edu/telecourse.html)
- Advanced
  - A real CMU course, Tom Mitchell's [Machine Learning](http://www.cs.cmu.edu/~ninamf/courses/601sp15)
- Readings
  - 'Pattern Recognition and Machine Learning' by Bishop (2006)
  - 'Machine Learning A Probabilistic Perspective' by Murphy (2012)


## Deep Learning
[[Back To TOC]](#table-of-contents)

Deep Learning is a rapidly advancing field so it is hard to keep track of every single good resource out there. Here, I attempt to curate some useful resources. I also highly recommend checking out the full paper reading list provided as a link to guide you through the topics you're interested in.

- Refreshers
  - Chapters 1, 2, 3, and 4 of Kyunghyun Cho's [Natural Language Understanding with Distributed Representation Lecture Notes](https://github.com/nyu-dl/NLP_DL_Lecture_Note)
  - Andrew Ng's Deep Learning courses [deeplearning.ai](https://www.deeplearning.ai/)
- CMU [Introduction to Deep Learning](http://deeplearning.cs.cmu.edu/) by Bhiksha Raj: The syallabus and reading list of this course seems very thorough and covers a lot of grounds. (This course seems to be currently under construction for not upcoming semester, and I couldn't find archived data. However, just following the syllabus itself with other materials seem suitable enough)
- The infamous **Stanford CS231n [Convolutional Neural Networks for Computer Vision](http://cs231n.stanford.edu/)** by Fei Fei Li: This course gives a nice introduction about Deep Learning and its applications to Computer Vision. The assignments are really well structured and gives you a well sense of how Deep Learning works without the TensorFlow magic, using only NumPy and Python.
  - [Youtube Playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- Books
  - **[Deep Learning Book](http://www.deeplearningbook.org)** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
  - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Neilson
- **Papers:** Full list organized by topics and models can be found in [Deep-Learning-Papers-Reading-Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap), or Columbia's seminar course [Advanced Topics in Deep Learning - Reading List](http://www.advancedtopicsindeeplearning.com/reading-list.html)
  - Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." Nature 521.7553 (2015): 436-444 [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf): A high-level survey paper by the three giants
  - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner.  "Gradient-Based Learning Applied to Document Recognition."  Proceedings of the IEEE, 86(11):2278-2324. 1998 (Seminal Paper: LeNet) [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf): LeNet: Image Classification on Handwritten Digits
  - Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): Big hit of Deep Learning, AlexNet
- Blog Posts
  - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
  - WildML [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

  
## Statistical NLP
[[Back To TOC]](#table-of-contents)

I couldn't find many good Statistical NLP courses out there that are at an introductory level. Here are some good resources to study it.

- Refreshers
  - Chapters 6~7 of [Goldberg](http://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037) book
- Columbia Michael Collins' COMS W4705: [Natural Language Processing](http://www.cs.columbia.edu/~mcollins/cs4705-fall2017/): This course covers a lot of traditional techniques often used in NLP.
  - **[Notes on Statistical NLP](http://www.cs.columbia.edu/~mcollins/notes-spring2013.html)**
  - Video Lectures on Coursera: Can't find the course anymore, but there are [Youtube videos](https://www.youtube.com/playlist?list=PL0ap34RKaADMjqjdSkWolD-W2VSCyRUQC)
- Chris Manning's [CS 224N/Ling 284 — Natural Language Processing](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml) before merging with Richard Socher's CS224D, covers some missing pieces of Michael Collins' class, along with more real life applications such as Machine Translation.
  - Video Lectures on [Youtube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- Readings
  - 'Foundations of Statistical Natural Language Processing' by Manning and Schütze (1999)


## Deep NLP
[[Back To TOC]](#table-of-contents)

Now we arrive to Deep NLP. You don't have to take all the materials I have mentioned up till now. In fact, you can even just start here, and tag along until you find some other interesting concepts embedded in these materials that are from above, and study them in parallel: for example, statistical NLP related stuff or Deep Learning architectures and papers. However, it is recommended to at least work in parallel with the above concepts to better understand the applications of Deep Learning to NLP.

- Stanford CS224N [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/syllabus.html)
  - The archived version for [2017 Winter Version](https://web.archive.org/web/20171221032823/http://web.stanford.edu/class/cs224n/syllabus.html)
  - [Youtube Playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
- [Oxford Deep NLP](https://github.com/oxford-cs-deepnlp-2017/lectures)
  - [Youtube Playlist](https://www.youtube.com/playlist?list=PL613dYIGMXoZBtZhbyiBqb0QtgK6oJbpm)(Unofficial)
- CMU CS 11-747 [Neural Networks for NLP](http://phontron.com/class/nn4nlp2017/schedule.html)
  - Youtube Playlist](https://www.youtube.com/playlist?list=PL8PYTP1V4I8ABXzdqtOpB_eqBlVAz_xPT)

### Neural Language Modeling with RNNs
[[Back To TOC]](#table-of-contents)

- Blog Posts
  - Colah [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  
### Embeddings
[[Back To TOC]](#table-of-contents)

- Word Embeddings
- Embedding phrases, sentences, paragraphs, context, and documents
- Cross Lingual

### Text Classification 
[[Back To TOC]](#table-of-contents)

- Neural Networks
- Convolutional Neural Networks
  - Blog Posts
    - WildML [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
- Recurrent Neural Networks
- Recursive Neural Networks

### Text Generation
[[Back To TOC]](#table-of-contents)

- Sequence-to-Sequence Learning
- Attention
- Question Answering
- Neural Machine Translation
- End-to-End Chatbots
- Dialogue Systems

### Multi-Lingual, Multi-task Learning
[[Back To TOC]](#table-of-contents)
- Multi-Lingual Tasks
- Cross-Lingual Tasks
- Multi-Tasks

### Neural Turing Machines
[[Back To TOC]](#table-of-contents)

- Memory Networks
- Pointer Networks

### Reinforcement Learning for NLP
[[Back To TOC]](#table-of-contents)

