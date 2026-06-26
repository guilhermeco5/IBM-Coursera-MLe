# Module 1 — Introduction to AI and Machine Learning

> Exploratory Data Analysis for Machine Learning — IBM / Coursera

→ Deep learning combines two steps of Machine Learning: feature recognition and the classification algorithm.

`AI ⊃ ML ⊃ DL`

## History of AI

### AI Winters

- **1960–1970** — Early algorithms and expert systems
- **1980–1990** — Neural networks and machine learning

### 1950s: Early AI

- **1950** — Alan Turing developed the Turing test, to test a machine's ability to exhibit intelligent behavior.
  - *Will a computer ever be able to sufficiently imitate a human to the point where a suspicious judge cannot tell the difference between human and machine?*
- **1956** — Artificial Intelligence was accepted as a field at the Dartmouth Conference.
- **1957** — Frank Rosenblatt invented the perceptron algorithm. This was the precursor to modern neural networks.
- **1959** — Arthur Samuel published an algorithm for a checkers program using machine learning.

**First AI Winter:**

- **1966** — The ALPAC committee evaluated AI techniques for machine translation and determined there was little yield from the investment (one of the major goals in AI).
- **1969** — Marvin Minsky published a book on the limitations of the perceptron algorithm, which slowed research in neural networks.
- **1973** — The Lighthill report highlighted AI's failure to live up to promises.

### 1980s: AI Boom

**Expert Systems** — systems with programmed rules designed to mimic human experts.

- Ran on mainframe computers with specialized programming languages (e.g. LISP).
- Were the first widely-used AI technology, with two-thirds of "Fortune 500" companies using them at their peak.
- **1986** — The "Backpropagation" algorithm is able to train multi-layer perceptrons, leading to new successes and interest in neural network research.

### Another AI Winter (1980s–1990s)

- Expert systems began to be melded into software suites of general business applications that could run on PCs instead of mainframes.
- Neural networks didn't scale to large problems.
- Interest in AI in business declined.

### 1990s–2000s: Machine Learning

AI solutions had successes in speech recognition, medical diagnosis, robotics, and many other areas.

- AI algorithms were integrated into larger systems and became useful throughout industry.
- The Deep Blue chess system beat world chess champion Garry Kasparov.
- Google's search engine launched using AI technology.
- **2006** — Geoffrey Hinton publishes a paper on unsupervised pre-training that allowed deeper neural networks to be trained. Neural networks are rebranded as deep learning.
- **2009** — The ImageNet database of human-tagged images is presented at the CVPR conference.
- **2010** — Algorithms compete on several visual recognition tasks at the first ImageNet competition.
- **2012** — Deep learning beats the previous benchmark on the ImageNet competition.
- **2013** — Deep learning is used to understand the "conceptual meaning" of words.
- **2014** — Similar breakthroughs appeared in language translation. These led to advancements in Web Search, Document Search, Document Summarization, and Machine Translation. A Stanford team creates a computer vision algorithm that can describe photos.
- **2015** — Deep learning platform TensorFlow is developed.
- **2016** — DeepMind's AlphaGo, developed by Aja Huang, beats Go master Lee Se-dol.
- **2018** — Waymo launches a commercial self-driving-car service in the suburbs of Phoenix.
- **2019** — IBM Project Debater is able to have a full debate with rebuttal against a champion human debater.

## Modern AI (2012–?): Impact

- **Computer vision** → Self-driving cars: object detection
- **Healthcare** → Improved diagnosis
- **Natural Language** → Language translation

### How is this era of AI different?

- Bigger datasets
- Faster computers
- Neural nets
  - → Cutting-edge results in a variety of fields

### Transformative Changes

- **Health** — Enhanced diagnostics, drug discovery, patient care, research, sensory aids
- **Industrial** — Factory automation, predictive maintenance, precision agriculture, field automation
- **Finance** — Algorithmic trading, fraud detection, research, personal finance, risk mitigation
- **Energy** — Oil & gas exploration, smart grid, operational improvement, conservation
- **Government** — Defense, data insights, safety & security, engagement, smarter cities
- **Transport** — Autonomous cars, automated trucking, aerospace, shipping, search & rescue

## Applications

**AI Omnipresence:**

- **Transportation** — Navigation, ride sharing
- **Social Media** — Audience, content
- **Daily Life** — Natural language, object detection

**Latest developments — Computer vision:**

- Deep learning "proven" to work for image classification → 2012
- Models outperform humans on image classification → 2015
- Object detection models beat previous benchmarks → 2016

## Machine Learning Workflow

```
Problem Statement → Data Collection → Data Exploration & Processing
→ Modeling → Validation → Decision Making & Deployment
```

**Vocabulary:**

- **Target** — category or variable that we are trying to predict
- **Features** — properties of data used for prediction
- **Example** — a single data point within the data (one row)
- **Label** — the target value for a single data point

---

## Review

### Introduction to Artificial Intelligence and Machine Learning

- **Artificial Intelligence** is a branch of computer science dealing with the simulation of intelligent behavior in computers. Machines mimic cognitive functions such as learning and problem solving.
- **Machine Learning** is the study of programs that are not explicitly programmed; instead, these algorithms learn patterns from data.
- **Deep Learning** is a subset of machine learning in which multilayered neural networks learn from vast amounts of data.

### History of AI

- AI has experienced cycles of AI winters and AI booms.
- AI solutions include speech recognition, computer vision, assisted medical diagnosis, robotics, and others.
- **Modern AI** — Factors that have contributed to the current state of Machine Learning are: bigger datasets, faster computers, open source packages, and a wide range of neural network architectures.

### Machine Learning Workflow

The machine learning workflow consists of:

```
Problem statement
      ↓
Data collection
      ↓
Data exploration and preprocessing
      ↓
Modeling
      ↓
Validation
      ↓
Decision Making and Deployment
```

### Common data taxonomy in open source ML packages

- **target** — category or value you are trying to predict
- **features** — explanatory variables used for prediction
- **example** — an observation or single data point within the data
- **label** — the value of the target for a single data point
