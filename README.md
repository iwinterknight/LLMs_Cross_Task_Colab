# Cross Task Colaboration in LLMs
This repo contains prototyping code in the form of python notebooks to build, train and evaluate a novel cross-attention layer based architecture that is trained to fuse the knowledge of two related tasks. The tasks here are passage summarization and question answering, in a "question prompted summary generation" setting. Question prompted summary generation aims to generate summaries from the passage, that contain necessary information to answer questions posed to the passage. This work is especially useful in extracting interest points in large scale data retrieval when users have interacted extensively with the data in the form of questions/reviews.

# Architecture
<p align="center">
  <img width="455" alt="cross-attn-training" src="https://github.com/iwinterknight/LLMs_Cross_Task_Colab/assets/37212007/406113f0-fedd-4b5b-9e6b-19327c29a7f2">
</p>
