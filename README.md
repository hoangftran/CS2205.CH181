# HYBRID METHOD COMBINING HIERARCHICAL TRANSFORMER ENCODERS AND SEQUENCE-TO-SEQUENCE FOR VIETNAMESE SPELLING CORRECTION

# Introduction

Spell correction is critical in Natural Language Processing, especially for languages like Vietnamese with complex tonal and diacritical systems. Misspellings can significantly impact tasks like sentiment analysis or information retrieval by introducing ambiguity. Transformer models, known for their attention mechanism [1], is a prevalent approach for Vietnamese spell correction task [2] [3]. Recent advancements using Hierarchical Transformer Encoders [2] have achieved state-of-the-art performance in Vietnamese spell correction. However, these models primarily rely on Encoder architectures, limiting their ability to handle situations where the corrected output has a different length than the misspelled input. While sequence-to-sequence (seq2seq) models using Transformers [3] have been explored for spell correction, their performance generally falls short of Hierarchical Transformer Encoders models.

This research proposal aims to develop a state-of-the-art Vietnamese spell correction model that overcomes these limitations by leveraging the strengths of both the Hierarchical Transformer Encoders and the Transformer-based architectures. We propose a novel hybrid approach that integrates these architectures within a sequence-to-sequence (seq2seq) framework, enabling the model to effectively handle varying input/output lengths while capturing context at both character and word levels.

Furthermore, we recognize the importance of training on a dataset that reflects the complexities of real-world Vietnamese text. Existing datasets often rely on synthetically generated errors, which may not adequately represent the diverse error types encountered in practice, such as Telex errors, short word replacements, Teencode, and misjoined/separated words. To address this, we propose the creation of a comprehensive Vietnamese spell correction dataset that expands upon the ViLexNorm corpus [4] to incorporate a wider range of real-world error types.

By combining architectural innovations with a focus on realistic training data, our proposed hybrid model aims to push the boundaries of Vietnamese spell correction. We expect this approach to outperform existing models, particularly in scenarios with varying input/output lengths and diverse error types. The successful development of this model has the potential to significantly enhance the accuracy and robustness of Vietnamese NLP tasks, opening up new possibilities for applications in areas such as text processing, information retrieval, and machine translation.

# Objective

The primary objective of this research is to develop a state-of-the-art Vietnamese spell correction model that effectively addresses the limitations of current approaches. We aim to achieve this through the following sub-objectives:

1. Develop a novel hybrid architecture that combines the strengths of Hierarchical Transformer Encoders and Transformer-based models:
   - Design and implement a seq2seq model that integrates Hierarchical Transformer Encoders and leverages both Encoder and Decoder Transformers to capture context at both character and word levels.
   - Evaluate the performance of the hybrid model against existing Vietnamese spell correction approaches, with a focus on scenarios with varying input/output lengths.

2. Create a comprehensive Vietnamese spell correction dataset that reflects real-world error types:
   - Expand upon the ViLexNorm corpus [4] to incorporate a wider range of real-world errors, including Telex errors, short word replacements, Teencode, and misjoined/separated words.
   - Develop a robust data collection and annotation methodology to ensure the quality and diversity of the dataset.
   - Evaluate the effectiveness of the expanded dataset in training the hybrid model compared to models trained on synthetically generated errors.

3. Achieve state-of-the-art performance in Vietnamese spell correction:
   - Train the hybrid model using the comprehensive Vietnamese spell correction dataset.
   - Evaluate the model's performance using standard metrics such as syllable-level precision, recall, and F1 score, as well as accuracy in both detection and correction tasks.
   - Benchmark the hybrid model's performance against existing Vietnamese spell correction approaches, aiming to outperform them in terms of overall accuracy and robustness to diverse error types.
   - Analyze the model's computational efficiency and scalability to assess its feasibility for real-world applications.

By achieving these objectives, we aim to push the boundaries of Vietnamese spell correction and contribute to the advancement of Vietnamese NLP tasks. The successful development of a hybrid model trained on a comprehensive dataset has the potential to significantly enhance the accuracy and robustness of various downstream applications, such as text processing, information retrieval, and machine translation.

# Methodology

This section outlines the research methodology for developing a state-of-the-art Vietnamese spell correction model. The research will be conducted over a period of 9 months, divided into three main stages: Dataset Creation (2 months), Model Development (5 months), and Model Evaluation and Explainability (2 months).

## 1. Dataset Creation (2 months)

### 1.1. Data Collection and Augmentation
- Source Acquisition: Identify and collect Vietnamese text data from various sources, including the ViLexNorm corpus [4], online text corpora (news articles, social media, blogs), and existing Vietnamese spell correction datasets.
- Error Type Analysis: Conduct a thorough analysis of real-world Vietnamese error types, focusing on Telex errors, short word replacements, Teencode, and misjoined/separated words.
- Error Augmentation: Develop a methodology to introduce realistic and diverse error types into the collected text data based on the error type analysis.
- Data Balancing: Ensure a balanced representation of different error types within the dataset to avoid bias towards specific error patterns.

### 1.2. Data Annotation and Quality Control
- Annotation Guidelines: Develop comprehensive guidelines for annotators to identify and correct errors in the augmented text data consistently.
- Annotation Process: Employ a team of Vietnamese linguists and native speakers to annotate the dataset, following the established guidelines.
- Quality Control: Implement a two-stage quality control process, including a pre-annotation qualification test and random sampling of annotated data to ensure accuracy and consistency.

## 2. Model Development (5 months)

### 2.1. Hybrid Architecture Design and Implementation
<!-- - Seq2seq Framework: Design a sequence-to-sequence model that serves as the foundation for the hybrid architecture. -->
- Hierarchical Transformer Encoders: Integrate Hierarchical Transformer Encoders into the seq2seq model to capture context at both character and word levels.
- Hybrid Encoder-Decoder: Implement a hybrid architecture that leverages both Encoder and Decoder Transformers to handle varying input and output lengths effectively.
<!-- - Character and Word Embeddings: Incorporate character-level and word-level embeddings to capture fine-grained and contextual information. -->

### 2.2. Model Training and Optimization
- Training Dataset: Prepare the training dataset by splitting the annotated Vietnamese spell correction dataset into train, validation, and test sets.
- Hyperparameter Tuning: Conduct extensive hyperparameter tuning to optimize the model's performance, considering factors such as learning rate, batch size, and number of layers.
- Training Process: Train the hybrid model using the prepared training dataset, employing techniques such as early stopping and model checkpointing to prevent overfitting and ensure the best model performance.

## 3. Model Evaluation and Explainability (2 months)

### 3.1. Performance Evaluation
- **Evaluation Metrics**: Assess the model's performance using standard metrics such as syllable-level precision, recall, F1 score, and accuracy in both detection and correction tasks.

#### Detection Metrics
1. **Detection Precision (DP)**: Measures the proportion of correctly identified errors out of all identified errors.
   - DP = (# of true detections) / (# of errors detected)

2. **Detection Recall (DR)**: Measures the proportion of correctly identified errors out of all actual errors.
   - DR = (# of true detections) / (# of actual errors)

3. **Detection F1-score (DF)**: Harmonic mean of Detection Precision and Detection Recall.
   - DF = (2 * DP * DR) / (DP + DR)

#### Correction Metrics
1. **Correction Precision (CP)**: Measures the proportion of correctly corrected errors out of all detected errors.
   - CP = (# of true corrections) / (# of errors detected)

2. **Correction Recall (CR)**: Measures the proportion of correctly corrected errors out of all actual errors.
   - CR = (# of true corrections) / (# of actual errors)

3. **Correction F1-score (CF)**: Harmonic mean of Correction Precision and Correction Recall.
   - CF = (2 * CP * CR) / (CP + CR)

#### Overall Accuracy
1. **Accuracy in Detected Errors**: Measures the proportion of correctly corrected errors out of all detected errors.
   - Accuracy (in % detected) = (# of exact corrections) / (# of exact corrections + # of wrong corrections)

2. **Accuracy in Total Errors**: Measures the proportion of correctly corrected errors out of all errors (including undetected ones).
   - Accuracy (in total) = (# of exact corrections) / (# of exact corrections + # of wrong corrections + # of wrong detections)

- **Benchmarking**: Compare the hybrid model's performance against existing Vietnamese spell correction approaches, including the Hierarchical Transformer Encoders [2] and VSEC [3].
- **Error Type Analysis**: Evaluate the model's performance on specific error types, particularly focusing on its ability to handle varying input/output lengths and diverse real-world errors.

### 3.2. Model Explainability and Interpretability
- Attention Analysis: Utilize attention visualization techniques to gain insights into how the model attends to different parts of the input sequence during the correction process.
- Case Studies: Conduct in-depth case studies on specific examples from the test set to understand the model's strengths, weaknesses, and error patterns.
- Computational Efficiency: Analyze the model's computational efficiency and scalability to assess its feasibility for real-world applications, considering factors such as inference time and memory requirements.

# Expected Results

Through the successful implementation of the proposed research methodology, we anticipate the following key results:

## 1. A Comprehensive Vietnamese Spell Correction Dataset
- The creation of a large-scale, diverse, and high-quality Vietnamese spell correction dataset that incorporates a wide range of real-world error types, including Telex errors, short word replacements, Teencode, and misjoined/separated words.
- The dataset will serve as a valuable resource for the Vietnamese NLP community, enabling future research and development in spell correction and related tasks.

## 2. A State-of-the-Art Hybrid Vietnamese Spell Correction Model
- The development of a novel hybrid model that combines the strengths of Hierarchical Transformer Encoders and Transformer-based architectures within a seq2seq framework.
- The hybrid model will demonstrate superior performance compared to existing Vietnamese spell correction approaches, particularly in handling varying input/output lengths and diverse error types.
- We expect the hybrid model to achieve state-of-the-art results on the newly created Vietnamese spell correction dataset, with significant improvements in syllable-level precision, recall, and F1 score.

## 3. Improved Robustness and Generalizability
- The hybrid model, trained on a comprehensive dataset reflecting real-world error types, will exhibit enhanced robustness and generalizability compared to models trained on synthetically generated errors.
- The model will be capable of effectively correcting a wide range of error types, including those not explicitly present in the training data, demonstrating its ability to capture underlying patterns and linguistic knowledge.

## 4. Insights into Model Behavior and Error Patterns
- Through attention analysis and case studies, we will gain valuable insights into the model's decision-making process and its ability to handle specific error types.
- These insights will contribute to a deeper understanding of the challenges in Vietnamese spell correction and guide future research directions.

## 5. Efficiency and Scalability
- The hybrid model will be designed with computational efficiency and scalability in mind, ensuring its feasibility for real-world applications.
- We expect the model to achieve a balance between performance and resource requirements, enabling its deployment in various Vietnamese NLP pipelines.

The successful achievement of these expected results will represent a significant advancement in Vietnamese spell correction. The hybrid model, trained on a comprehensive dataset, will push the boundaries of current approaches and open up new possibilities for downstream applications. The insights gained from this research will also contribute to the broader field of Vietnamese NLP, fostering further research and innovation.

Moreover, the public release of the Vietnamese spell correction dataset and the hybrid model will enable the Vietnamese NLP community to build upon this work and develop more advanced and specialized spell correction systems. The improved accuracy and robustness of Vietnamese spell correction will have far-reaching impacts on various domains, including text processing, information retrieval, machine translation, and user-generated content analysis.

# References

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. Attention Is All You Need. arXiv: 2111.00640
2. Hieu Tran, Cuong V. Dinh, Long Phan, Son T. Nguyen. Hierarchical Transformer Encoders for Vietnamese Spelling Correction. The 34th International Conference on Industrial, Engineering & Other Applications of Applied Intelligent Systems, 2021
3. Dinh-Truong Do, Ha Thanh Nguyen, Thang Ngoc Bui, Dinh Hieu Vo. VSEC: Transformer-based Model for Vietnamese Spelling Correction. arXiv preprint arXiv: 2111.00640, 2021
4. Thanh-Nhi Nguyen, Thanh-Phong Le, Kiet Van Nguyen. ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text. The 18th Conference of the European Chapterof the Association for Computational Linguistics (EACL), 2024
