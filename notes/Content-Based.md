https://modelverse.cs.cmu.edu/

https://github.com/generative-intelligence-lab/modelverse

https://generative-intelligence-lab.github.io/modelverse/

## Abstract

- Content-based model search: given a query and a large set of generative models, find the models that best match the query. 基于内容的模型搜索任务:给定一个查询和一组生成模型，找出与查询最匹配的模型。
- An image, a sketch, a text description, another generative model, or a combination of the above. 图像、草图、文本描述、另一个生成模型或上述模型的组合


## Introduciton

- #### Aim: 

- Find the most relevant deep image generative models that satisfy a user’s input query. 其目的是找到满足用户输入查询的最相关的深度图像生成模型。

- #### Result: 

- Our method presents the first contentbased search algorithm for machine learning models. 提出了第一个基于内容的机器学习模型搜索算法。

- #### Method: 

- we first present a general probabilistic formulation of the model search problem and present a Monte Carlo baseline. To reduce the search time and storage, we “compress” the model’s distribution into pre-computed 1st and 2nd order moments of the deep feature embeddings of the original samples. We then derive closed-form solutions for model retrieval given an input image, text, sketch, or model query. Our final formula can be computed in real-time. 首先提出了模型搜索问题的一般概率公式，并提出了蒙特卡罗基线。为了减少搜索时间和存储，我们将模型的分布“压缩”为原始样本深度特征嵌入预先计算的一阶和二阶矩。然后，我们给出输入图像、文本、草图或模型查询，推导出模型检索的封闭解。我们的最终公式可以实时计算。

## Previous Work

- Deep generative models. 深度生成模型

- Image editing with generative models. 使用生成模型进行图像编辑

- Content-based retrieval. 基于内容的检索

- Transfer learning for generative models. 生成模型的迁移学习

  We show that content-based model search can be used to automatically select pre-trained generators for a new domain, and improve the efficiency of model finetuning. 我们证明了基于内容的模型搜索可以用于自动选择预训练的生成器为一个新的领域，并提高模型微调的效率。

## Methods

- #### Main goal: 

  1. We aim to build a search/retrieval system for deep generative models. When a user specifies an image, sketch, or text query, we would like to retrieve a model that best matches the query. 目标是为深度生成模型建立一个搜索/检索系统。当用户指定图像、草图或文本查询时，我们希望检索与查询最匹配的模型。
  2. we introduce a probabilistic formulation for generative model retrieval. Our formulation is general to different query modalities and various types of generative models, and can be extended to different algorithms. 我们引入了生成式模型检索的概率公式。我们的公式适用于不同的查询模式和各种类型的生成模型，并可以扩展到不同的算法。

- #### Probabilistic Retrieval for Generative Models

  1. We derive our model retrieval formulation based on a Maximum Likelihood Estimation (MLE) objective, and we present our model retrieval algorithms for an image, a text, and a sketch query, respectively. 推导了基于最大似然估计(MLE)目标的模型检索公式，并分别给出了针对图像、文本和草图查询的模型检索算法。
  2. Image-based model retrieval. 基于图像的模型检索：Gaussian Density
  3. Sketch-based model retrieval. 基于草图的模型检索：CLIP
  4. Text-based model retrieval. 基于文本的模型检索：1st + 2nd Moment

- #### Extensions and Applications

  1. Multimodal query. 多模式查询
  1. Finding similar models. 寻找相似的模型：Fréchet Distance
  1. Real image editing. 真实的图像编辑
  1. Few-shot fine-tuning. 少镜头微调

- #### User Interface

  The UI supports searching and sampling from deep generative models in real-time. UI支持从深度生成模型中实时搜索和采样。The user can enter a text prompt, upload an image/sketch, or provide both text and an image/sketch. The interface displays the models that match most closely with the query. Clicking a model takes the user to a new page where they can sample new images from the model. 用户可以输入文本提示、上传图像/草图，或者同时提供文本和图像/草图。该界面显示与查询最匹配的模型。单击一个模型会将用户带到一个新页面，在那里他们可以从模型中提取新的图像。


## Experiments

-  Here we first evaluate our model retrieval methodology over text, image, and sketch modalities and discuss several algorithmic design choices.  我们首先评估我们的模型检索方法在文本、图像和草图形式，并讨论几个算法设计选择。

-  We then show qualitative and quantitative results for the extensions and applications enabled by our model search. 我们展示了模型搜索所支持的扩展和应用的定性和定量结果。

-  评估了使用不同技术训练的 133 个生成模型的集合，包括 GAN、扩散模型、MLP 基于生成模型 CIPS 和自回归模型 VQGAN 。 对于评估，根据生成图像的类型手动为每个模型分配 ground truth 标签，总共有 23 个标签。 示例标签包括“面部”、“动物”、“室内”，其中所有面部模型都将具有“面部”标签。 同样，在 LSUN 的卧室和会议类别等数据集上训练的模型被标记为“室内”。

-  Implementation details. 实现细节：对于基于图像的模型检索和相似模型搜索，测试了三种不同的图像特征：Inception、CLIP和DINO。对于基于文本的模型检索，使用CLIP特性，CLIP学习一个大小不变的特征空间，因为它是通过最大化文本和图像特征之间的余弦相似度来训练的。因此，使用“2规范化的CLIP特性”。

- #### Model Retrieval

  1. Evaluation metrics. 评价指标：

     1. Top-k accuracy, i.e., predicting the ground truth generative model of each query in top k. 即预测top k中每个查询的ground truth生成模型。

     2. Mean Average Precision@k (mAP@k)

        ![image-20221120201116635](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120201116635.png)

        其中Pq(j)是给定查询q的前j个预测的精度，Relq(j)是第j个预测相关的二进制指标。GTq是与查询相对应的相关模型的数量。

  2. Text-based model retrieval. 基于文本的模型检索：

     ![image-20221120201521219](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120201521219.png)

     

  3. Model retrieval via image and sketch queries.

     ![image-20221120202000501](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120202000501.png)

     

  4. Running time and memory.

     ![image-20221120202016985](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120202016985.png)

     

  5. With the 1st Moment method, a user can retrieve models from a 1-million-model collection using text, sketch, or image query in real-time. 使用1st Moment方法，用户可以使用文本、草图或图像实时查询从100万个模型集合中检索模型

  6. Baseline: Index models using model descriptions. 基线:使用模型描述对模型进行索引：

     An alternative approach to content-based model search is to index each model using user-defined descriptions. For each user query, we will find the model with a description that best matches the query. 基于内容的模型搜索的另一种方法是使用用户定义的描述为每个模型建立索引。对于每个用户查询，我们将找到具有与查询最匹配的描述的模型。

     We support the content-based search method in combination with a metadata search. 我们支持基于内容的搜索方法与元数据搜索相结合。

##  Extensions and Applications

- Our work enables users to explore available generative models and find the best models for different use cases. 我们的工作使用户能够探索可用的生成模型，并为不同的用例找到最佳的模型。

  Here we show several use cases, including multi-modal queries, finding similar models, image editing, and few-shot transfer learning. 这里我们将展示几个用例，包括多模态查询、查找相似模型、图像编辑和少镜头迁移学习。
- #### Multimodal User Query
- We demonstrate how leveraging multiple input modalities from the user can retrieve models which are better tailored to user queries. 利用来自用户的多个输入模式来检索更适合用户查询的模型。
- ![image-20221120204048646](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120204048646.png)

- #### Finding Similar Models

- We use the FID between the feature distribution of each generative model as the scoring method for retrieving similar models. We use CLIP , DINO, and Inception networks’ feature space and evaluate Average Precision using ground truth similar models (models with same label). We get an AP of 0.68, 0.68, and 0.66 respectively. Figure 7 shows qualitative examples of similar model retrieval using FID metric in CLIP feature space. 我们使用每个生成模型的特征分布之间的FID作为检索相似模型的评分方法。我们使用CLIP, DINO和Inception网络的特征空间，并使用地面真理相似模型(具有相同标签的模型)评估平均精度。我们得到的AP分别是0.68,0.68和0.66。图7显示了在CLIP特征空间中使用FID度量检索相似模型的定性示例。

- ![image-20221120204331429](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120204331429.png)

- #### Image Reconstruction and Editing

  1. Image inversion. 图像反演：

     ![image-20221120204639097](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120204639097.png)

     

  2. Image editing and interpolation. 图像编辑和插值：

     ![image-20221120205100795](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120205100795.png)

     

- #### Few-Shot Transfer Learning

- vision-aided GANs

- ![image-20221120205328151](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120205328151.png)

## Discussion and Limitations

- We have introduced the problem of content-based retrieval for deep generative image models, whose goal is to help users find, explore, and share new generative models more easily. Interestingly, we have found that scoring based on a probabilistic model works well, and further that applying a Gaussian density or first-moment approximation to the distribution of generated image features produce accurate search results with a minimal memory and time footprint. 

  我们介绍了深度生成图像模型的基于内容的检索问题，其目标是帮助用户更容易地查找、探索和共享新的生成模型。有趣的是，我们发现基于概率模型的评分效果很好，进一步地，将高斯密度或一阶近似应用于生成的图像特征的分布，以最小的内存和时间占用产生精确的搜索结果。

- Our experiments have shown that searches over an indexed collection are useful for finding a good model for image editing and transfer learning. 对索引集合的搜索对于寻找一个好的图像编辑和迁移学习模型是有用的。

- ![image-20221120210237382](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221120210237382.png)

- 局限性：草图查询(例如，鸟草图)将匹配模型与抽象风格。CLIP功能是否应该与草图的形状相匹配，还是与样式和纹理相匹配，是不明确的；对于冲突的多模态查询(大象文本查询+狗的图像)，我们的系统很难检索这两个概念的模型，没有大象模特。

