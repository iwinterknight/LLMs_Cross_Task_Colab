# Cross Task Colaboration in LLMs
This repo contains prototyping code in the form of python notebooks to build, train and evaluate a novel cross-attention layer based architecture that is trained to fuse the knowledge of two related tasks. The tasks here are passage summarization and question answering, in a "question prompted summary generation" setting. Question prompted summary generation aims to generate summaries from the passage, that contain necessary information to answer questions posed to the passage. This work is especially useful in extracting interest points in large scale data retrieval when users have interacted extensively with the data in the form of questions/reviews.

# Architecture
<p align="center">
  <img width="955" alt="cross-attn-training" src="https://github.com/iwinterknight/LLMs_Cross_Task_Colab/assets/37212007/406113f0-fedd-4b5b-9e6b-19327c29a7f2">
</p>
In a nutshell, the cross-task-attention model uses an instruction fine-tuned Flan T5 encoder-decoder model, pretrained on passage summarization task (CNNDailymail) and the encoder blocks of another instruction fine-tuned Flan T5 transformer, pretrained on question answering task (SQuAD2.0). Both the models are trained in a using low rank adaptation(LoRA) using transformer's `peft` library. After removing the softmax layer of the summarizer model's decoder, a cross-attention layer, followed by FFN and layer normalization, is applied between the Summarizer model's decoder representations and the QA encoder representations. The cross attention FFN is essentially a binary classifier to predict whether the summary captures information to answer questions posed to the passage.

The cross-attention architecture is trained on a joint cross-entropy loss for summary generation and the aforementioned binary class prediction, with a weighting factor α, a hyperparameter.

<p align="center">
<img width="323" alt="training_parameters" src="https://github.com/iwinterknight/LLMs_Cross_Task_Colab/assets/37212007/8af36fcf-a5bb-4ea6-937e-03988c9dc869">
</p>

# Evaluation
The base evaluation uses Rouge metric and BERT Score to compare the generated summary's quality against other open sourced summarization models. 
<p align="center">
<img width="334" alt="rouge_scores" src="https://github.com/iwinterknight/LLMs_Cross_Task_Colab/assets/37212007/66edb9e2-4a0b-4500-a50a-54bac7a01984">
</p>
<p align="center">
<img width="317" alt="bert_scores" src="https://github.com/iwinterknight/LLMs_Cross_Task_Colab/assets/37212007/8d00aea0-2dd5-4ed7-969f-c36c71aad077">
</p>
The summary's question answering capability is expressed with the help of an entailment score, ie how well does the generated summary entail answers to questions posed to the passage.
<p align="center">
<img width="326" alt="entailment scores" src="https://github.com/iwinterknight/LLMs_Cross_Task_Colab/assets/37212007/a5fc3971-403f-4ce5-965d-1c9aa3450b25">
</p>
