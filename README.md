Problem Statement

Students often struggle to identify which subject their exam-style questions belong to, especially when preparing for competitive exams like IIT-JEE / NEET.
This project aims to build an NLP-based Smart Study Assistant that automatically classifies a given question into:

1.Physics
2.Chemistry
3.Mathematics
4.Biology

The project followed a structured machine learning pipeline. First, exploratory data analysis (EDA) and data cleaning were performed by checking class distribution, analyzing question text length, removing missing values, and creating a cleaned text column for further processing. In the feature engineering stage, TF-IDF vectorization was applied to convert text into numerical features, using up to 8,000 features with both unigrams and bigrams (n-gram range of 1 to 2) to capture contextual information. For model selection, multiple classification algorithms were benchmarked, including Naive Bayes, Logistic Regression, Linear SVM, to identify the most suitable approach for this NLP task. Based on validation performance, Linear Support Vector Classifier (LinearSVC) was selected as the best-performing model and was retrained manually for final use. The trained TF-IDF vectorizer, classification model, and label encoder were then saved using Pickle as tfidf.pkl, best_model.pkl, and label_encoder.pkl respectively, enabling seamless deployment within the Streamlit web application.
This can be extended into a full AI tutor system with topic tagging, difficulty prediction, and answer retrieval.
