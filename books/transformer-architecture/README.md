

## 5 Reasons Why Large Language Models (LLMs) Like ChatGPT Use Reinforcement Learning Instead of Supervised Learning for Finetuning

With the huge success of Generative Artificial Intelligence in the past few months, Large Language Models are continuously advancing and improving. These models are contributing to some noteworthy economic and societal transformations. The popular ChatGPT, which OpenAI has developed, is a natural language processing model that allows users to generate meaningful text just like humans. Not only this, it can answer questions, summarize long paragraphs, write codes and emails, etc. Other language models, like Pathways Language Model (PaLM), Chinchilla, etc., have also shown great performances in imitating humans. 

Large Language models use reinforcement learning for fine-tuning. Reinforcement Learning is a feedback-driven Machine learning method based on a reward system. An agent learns to perform in an environment by completing certain tasks and observing the results of those actions. The agent gets positive feedback for every good task and a penalty for each bad action. LLMs like ChatGPT portray exceptional performance, all thanks to Reinforcement Learning.

ChatGPT uses Reinforcement Learning from Human Feedback (RLHF) to fine-tune the model by minimizing the biases. But why not supervised learning? A basic Reinforcement Learning paradigm consists of labels used to train a model. But why can’t these labels be directly used with the Supervised Learning approach? Sebastian Raschka, an AI and ML researcher, shared some reasons in his tweet about why Reinforcement Learning is used in fine-tuning instead of supervised learning. 

1. The first reason for not using Supervised learning is that it only predicts ranks. It doesn’t produce coherent responses; the model just learns to give high scores to responses similar to the training set, even if they are not coherent. On the other hand, RLHF is trained to estimate the quality of the produced response rather than just the ranking score. 

2. Sebastian Raschka shares the idea of reformulating the task as a constrained optimization problem using Supervised learning. The loss function combines the output text loss and the reward score term. This would result in a better quality of the generated response and the ranks. But this approach only works successfully when the objective is to produce question-answer pairs correctly. But cumulative rewards are also necessary to enable coherent conversations between the user and ChatGPT, which SL can’t provide.

3. The third reason for not opting for SL is that it uses cross-entropy to optimize the token level loss. Though at the token level for a text passage, altering individual words in the response may have only a small effect on the overall loss, the complex task of generating coherent conversations can have a complete change of context if a word is negated. Thus, depending on SL cannot be sufficient, and RLHF is necessary for considering the context and coherence of the entire conversation. 

4. Supervised learning can be used to train a model, but it was found that RLHF tends to perform better empirically. A 2022 paper, “Learning to Summarize from Human Feedback,” showed that RLHF performs better than SL. The reason is that RLHF considers the cumulative rewards for coherent conversations, which SL fails to capture due to its token-level loss function.

5. LLMs like InstructGPT and ChatGPT use both Supervised Learning and Reinforcement Learning. The combination of the two is crucial for attaining optimal performance. In these models, the model is first fine-tuned using SL and then further updated using RL. The SL stage allows the model to learn the basic structure and content of the task, while the RLHF stage refines the model’s responses to improved accuracy. 

Content Source: https://www.marktechpost.com/2023/03/05/5-reasons-why-large-language-models-llms-like-chatgpt-use-reinforcement-learning-instead-of-supervised-learning-for-finetuning/
