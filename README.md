# Movie Recommender System

##  Project Overview

This repository contains a Python-based movie recommendation system that compares **three different recommendation approaches** using the classic MovieLens 100k dataset:

1. **Content-based recommendations** – based on movie genre profiles inferred from user ratings.  
2. **User-to-user collaborative filtering** – based on similarities between users’ ratings and demographic attributes.  
3. **Hybrid recommendations** – combining the above two methods to provide more robust suggestions.

The system also computes **precision metrics** and similarity measures between recommendation lists, showcasing an understanding of both algorithm design and evaluation in recommender systems.

---

##  Dataset

The dataset used in this project is the **MovieLens 100k dataset**, which contains:

- User–movie ratings (`u.data`)
- Movie metadata and genre tags (`u.item`)
- User demographics (`u.user`)

This dataset is commonly used in research and teaching for recommender systems. It includes ratings on a 1–5 scale and allows for training and testing multiple recommendation strategies.

---

##  Requirements

Before running the project, install the required Python packages:

```bash
pip install pandas numpy scikit-learn
