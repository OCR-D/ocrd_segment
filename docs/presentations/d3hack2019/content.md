layout: true

---

class: title-slide

# `ocrd_segment`

## Data-driven page segmentation

Janek Schleicher, Robert Sachunsky, Kay-Michael Würzner

---

# The task

  1. Segment pages of printed documents
  2. Classify the segments according to their layout-semantic function

---

# The task

<center><img src="img/ex1.png" style="height:600px" /></center>

---

# The data

1. 180,000 annotated pages from the periodical *Die Grenzboten* (1841-1922)
  - Homogeneous layout
  - Completely annotated on page **and** volume level
2. 25,000 annotated pages from XXX volumes of the *German Text Archive* (1506-1899)
  - Heterogeneous layout
  - Incompletely annotated on page **and** volume level
3. 1,000 annotated pages from 230 volumes of the *German Text Archive* (????-????)
  - Heterogeneous layout
  - Completely annotated on page (but not on volume) level

---

# Some questions

Build a data-driven framework with neural networks which performs the task of page segmentation and segment classification.

- Level of representation (pixel, coordinates ...)?
- General architecture (one or multiple classifiers)?
- Network architecture?
- Preprocessing and/or augmentation methods?

---

# The team

- Robert

  is a XXX from Dresden who primarily works on OCR and in particular on OCR post correction.

- Janek

  is a computer scientist from Berlin who has a strong background in machine learning and works on text generation.

- Kay

  is a computational linguist from Dresden who works on OCR and is interested in natural language processing in general.
