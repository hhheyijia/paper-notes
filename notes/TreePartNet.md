https://vcc.tech/research/2021/TreePartNet 

https://github.com/marktube/TreePartNet

## TreePartNet: Neural Decomposition of Point Clouds for 3D Tree Reconstruction

## Abstract

**TreePartNet:  A neural network aimed at reconstructing tree geometry from point clouds obtained by scanning real trees.** 是一个 旨在从 扫描获取的真实树的点云模型中 重建树的几何形状 的神经网络。

**Key idea: Learn a natural neural decomposition exploiting the assumption that a tree comprises locally cylindrical shapes.** 利用树包含局部圆柱形状的假设来学习自然神经分解。

**Two-step: **

(1)Two networks are used to detect priors from the point clouds. One detects semantic branching points, and the other network is trained to learn a cylindrical representation of the branches. 使用两个网络从点云中检测先验。 一个网络用来检测语义分支点，另一个网络用来训练学习分支的圆柱形表示。

(2)We apply a neural merging module to reduce the cylindrical representation to a final set of generalized cylinders combined by branches. 应用神经合并模块将圆柱表示简化为一组由分支组合的广义圆柱的集合。

## Introduction

Efficiently and accurately representing, generating, and reconstructing tree geometry is still an open problem due to the complexity and the diversity of branching patterns and the intricate underlying growth mechanisms. 由于分枝模式的复杂性和多样性以及复杂的潜在生长机制，高效、准确地表示、生成和重建树的几何形状仍然是一个悬而未决的问题。

**Paper: An algorithm for the automatic reconstruction of concise geometries of biological trees from point clouds.** 本文提出了一种从点云自动重建生物树简洁几何图形的算法。

Our approach constructs a generalized cylindrical representation based on the core idea of learning a neural decomposition with branching and joint semantics, where joint elements can connect the branches. Furthermore, our approach is based on the assumption that the local shape of branches is naturally cylindrical. Thus, we can partition the input point cloud into clusters that can be approximated by the generalized, parameterized cylinders. 我们的方法基于学习具有分支和联合语义的神经分解的核心思想构建了一个广义圆柱表示，其中联合元素可以连接分支。此外，我们的方法基于分支的局部形状自然为圆柱形的假设。因此，我们可以将输入点云划分为可以由广义参数化圆柱体近似的簇。

![image-20221127203704059](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\main idea.png)

图1. 从(a)原始扫描中有噪声且不完整的非结构化点集合开始，我们提出TreePartNet来寻找分支结构并创建圆柱形分解(b)。树的几何结构由广义圆柱体和光滑分支点(c)表示。最终，可以添加纹理和树叶来增强视觉吸引力(d)。在(e)中，我们从不同的视角展示了渲染版本。

We first exploit a semantic segmentation neural network to predict whether a point is located in a junction region because the joint semantics indicate important topological information. 首先利用语义分割神经网络来预测一个点是否位于连接区域，因为联合语义表示重要的拓扑信息。We then perform a deep, refined clustering with a fixed cluster number to partition the point cloud into an over-complete set of branches. 我们使用固定的簇数执行深度、精细化的聚类，将点云划分为一组过完整的分支。After that, we design a pairwise affinity network to construct a symmetric affinity matrix, where each element predicts the probability that two clusters can be merged. 我们设计了一个两两亲和网络，构造了一个对称亲和矩阵，其中每个元素预测了两个聚类可以合并的概率。The three networks are tightly coupled, and they are trained simultaneously and efficiently. 这三个网络是紧密耦合的，它们被同时有效地训练。In the last step, we fit a generalized cylinder for each cluster to reconstruct the underlying shape. The cylindrical representation then naturally resolves the noise and density non-uniformity, i.e., dense and thick areas due to repeating scanning and misalignment and sparse areas due to occlusions. 最后一步中，我们为每个聚类拟合一个广义圆柱来重建底层形状。圆柱形表示自然地解决了噪声和密度的不均匀性，即重复扫描和不对中造成的密集和粗大区域和遮挡造成的稀疏区域。Finally, we connect the reconstructed branches by considering joint regions to obtain a complete skeletal structure and a polygonal surface mesh. 最后，通过考虑关节区域将重构的分支连接起来，得到完整的骨骼结构和多边形表面网格。

#### Contributions:

(1) a prior-based supervised neural decomposition to learn a cylindrical representation of 3D trees even for incomplete or noisy point sets, 基于先验的监督神经分解，以学习三维树的圆柱形表示，即使是不完整或有噪声的点集;

(2) a new combined reconstruction method for tree structures based on generalized cylinders and branching points, 基于广义圆柱和分支点的树结构组合重建方法; 

(3) a geometry-aware graph clustering method based on a pairwise affinity network, which defines a new module, named Scaled Cosine Distance, inspired by the Transformer. 基于两两亲和网络的几何感知图聚类方法，该方法定义了一个新的模块，命名为缩放余弦距离，灵感来自Transformer。

## Related Work

####   3D Reconstruction of Geometric Tree Models

We are the first to propose a supervised cylindrical representation for 3D tree reconstruction from point clouds. 我们是第一个提出监督圆柱形表示从点云三维树重建。

####  Shape Representations and Decomposition

Learning shapes from 3D data.从三维数据中学习形状

Shape decomposition. 形状分解: GC-based decomposition. But we utilize deep clustering to adaptively decompose tree shapes with a small number of parts that are more robust to noise.但是我们利用深度聚类来自适应分解具有少量部分的树形状，这些部分对噪声更健壮。Our approach can reconstruct GC-based representations of branching structures even for incomplete or noisy point sets. 我们的方法可以重建基于GC的分支结构表示，即使是对于不完整或有噪声的点集。

#### Point set learning

 We borrow from both 3D point feature learning and Transformer. We extend PointNet++ to achieve a novel method that robustly reconstructs 3D branching structures by generalized cylinders and branch joints. 我们借鉴了3D点特征学习和Transformer。我们扩展了pointnet++，实现了一种利用广义圆柱和分支节点稳健重建三维分支结构的新方法。

## PROBLEM STATEMENT AND OVERVIEW

**Goal:  Our goal is to obtain 3D tree reconstruction from an unstructured point cloud.** 我们的目标是从一个非结构化的点云中获得三维树的重建。

The point cloud is assumed to be a scan of a tree surface, and we use both real and synthetic trees in this paper. The input data may be noisy, incomplete, and with non-uniform density distribution. We also assume the normal vectors are not provided. 假设点云是对树表面的扫描，本文同时使用了真实树和合成树。输入数据可能有噪声，不完整，密度分布不均匀。我们还假设没有提供法向量。

#### Core idea:

The core idea of our algorithm is to decompose the point cloud into foliage, non-overlapping sets of branch and junction parts by using deep neural networks. Each part of a branch is then reconstructed as a surface mesh patch. Finally, we merge the branch parts by linking to the critical joint points located in the junction regions. 该算法的核心思想是利用深度神经网络将点云分解为叶子集、不重叠的分支集和连接部分集。然后将分支的每个部分重建为一个表面网格补丁。最后，我们通过连接位于连接区域的关键节点来合并分支部分。

#### Main steps

![image-20221127220332772](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\main steps.png)

图2. 从输入点云P (a)开始，我们首先使用一个语义分割模块来检测连接部分{Ji}（(b)中的红点）。同时，我们的神经网络将输入分解为一组小规模的簇{Ci } (e)，这些簇被自动合并成不重叠的分支{Bi } (d).然后，我们从分割后的分支中提取离散的骨骼部分(e)。通过使用关节骨架节点（红色表示），我们得到了一个完整的骨架（f），它由广义圆柱体表示，并转换为一个表面网格(g)。

给定图2 (a)中的点云，我们首先使用语义分割模块(图2 (b))检测连接部分{Ji}。然后我们将点云分解为一组分支部分。由于树形复杂，且组件的合适数量未知，因此首先执行精细聚类模块，得到数量固定的局部小分支({Ci})的过完整集合(本文为256)，如图2 (c)所示。然后，我们的神经网络通过两两亲和模块自适应合并这些局部分支，得到数量较少但更紧凑的分支{Bi}(图2 (d))。最后，每个分支都表示为一个广义圆柱，该圆柱由沿着骨骼曲线扫描一组横截面剖面定义。在图2 (e)中，我们显示了离散骨架。然后，考虑到语义连接点(红色的骨骼节点)，我们通过计算一个完整的骨骼连通性图(图2 (f))来连接这些分支，以获得底层树几何结构的合理表面表示(图2 (g))。表面网格可以通过附加树叶和纹理直接用于制作真实的树模型。

## NEURAL DECOMPOSITION

#### Our approach: 

In our approach, we first perform a semantic segmentation to indicate points to belong to either a branch or a junction point used later to combine the decomposed parts. 首先执行语义分割，以指明属于分支或稍后用于组合分解部分的连接点的点。Then, instead of directly detecting cylinders, our architecture first computes per-point features and then predicts the neural decomposition. 然后，不是直接检测圆柱体，我们的体系结构首先计算每个点的特征，然后预测神经分解。Our network architecture comprising of three modules is summarized in Fig. 3, where the top row shows the detection of junctions semantic segmentation module, and the other two processing the branches (the fine clustering module and pairwise affinity module). 我们的网络架构由三个模块组成，如图3所示，其中最上面一行是检测连接的语义分割模块，另外两个是处理分支的模块(精细聚类模块和成对亲和模块)。Thus, we propose a fine-to-coarse clustering approach by leveraging the ability of deep neural networks to learn features. 因此，我们提出了一种由细到粗的聚类方法，利用深度神经网络学习特征的能力。

![image-20221127223614936](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\network.png)

图3. 用于神经分解的网络架构：我们的网络的顶部分支(用绿色箭头表示)代表语义分割模块，它学习多尺度的逐点特征，以检测连接部分。其他两个分支(用橙色和蓝色箭头表示)是精细聚类模块和成对亲和模块。前者将局部上下文特征与点向特征向量连接，将输入分解为一组局部圆柱斑块，后者通过学习亲和矩阵将斑块合并。

#### 4.1 Semantic Segmentation

**PointNet++**

集合抽象层：输入点子集P （最远点采样进行下采样，学习局部上下文特征）（多尺度特征学习） 多尺度特征F1 （增加采样半径，下采样） 多分辨率特征F2

特征传播层：（采样，分层） 更新后的特征F3 （P'从原点传播） 点特征F4 

全连通层：将特征向量反馈给一组预测得分S的全连通层，应用sigmoid函数计算某点是否是结点(0-trunk, 1-junction)的概率

1. Foliage segmentation. 树叶分割

   我们使用pointnet++作为分类器来过滤叶子点(对于每个点:0个叶子，1个分支)。

#### 4.2 Branch Clustering 分支聚类

不同的树形状，树枝的数量是无法预先知道的，因此，无法估计在固定数量的群集上的概率分布。

**Two modules.** Initially, we learn to group the points with a relatively large number of clusters. We then automatically merge similar clusters by computing an affinity matrix wherein a pair of clusters belonging to the same branch has a higher affinity than a pair in different branches. 两个模块。最初，我们学习用相对大量的簇将点分组。然后，我们通过计算亲和矩阵自动合并相似的集群，其中属于同一分支的一对集群比属于不同分支的一对集群具有更高的亲和度。

1. Initial fine clustering. 初始精细聚类

   利用采样点的坐标提取局部圆柱特征。由于采样点均匀地覆盖了整个形状，因此使用采样点作为聚类种子。对于输入P中的每个点，模块输出该点属于P '中某点周围的给定局部圆柱区域的概率。然后通过选择概率最高的输入点，得到初始聚类标签{Ci}。

   假设初始簇具有共同的几何结构（局部圆柱体）。 因此可以共享每个点的权重和局部特征的一致性。 为了预测每个点的特征属于哪个初始聚类，将 F5 中的新特征向量输入共享的多层感知器 (MLP) ，输出初始聚类向量 V ，它为每个点分配一个属于 N ′ 个初始簇之一的分数。

   **Shared MLP**: MLP是多层感知机的缩写。Shared MLP 是点云处理网络中的一种说法，强调对点云中的每一个点都采取相同的操作，其本质上与普通MLP没什么不同。在shared mlp中，输入为包含多点的点云，我们对每一个点乘以相同的权重，这就叫做shared weights。而在mlp中，输入为单个向量，因此不需要共享权重。其在网络中的作用即为MLP的作用：特征转换、特征提取。

   采用共享MLP，是因为它具有稀疏连接和参数共享的特性，可以很好地平衡网络性能和内存空间需求。此外，在训练神经网络时，使用更少的权值参数也可以减少过拟合。

2. Merging clusters. 合并集群

   We propose a geometry-aware pairwise affinity module to construct an affinity matrix M by non-linearly auto-encoding the clustering seeds P′ into a latent space, where seeds belonging to the same branch should have similar embedded features. 我们提出了一个几何感知的成对亲和力模块，通过将聚类种子 P' 非线性自动编码到潜在空间中来构建亲和力矩阵 M，其中属于同一分支的种子应具有相似的嵌入特征。

   该方法的核心是准确度量两个聚类Ci和Cj之间的相似性。我们将它们的相似性定义为:

   ![image-20221128113153823](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula1.png)

   其中 α 是初始化为 α =−10 的标量权值，在训练过程中与其他网络参数一起学习。 α 也是模块名称中的“刻度”：缩放余弦距离。Dp 编码两个簇种子之间的欧氏(L2)距离，因为两个相似的簇的位置应该沿着一个分支靠近。Df 表示嵌入特征空间中的相似性。

   **Scaled Cosine Distance**缩放余弦距离：输入特征 Fin，输出特征 Fattention 是值的加权和，其中分配给每个值的注意力权重 A 由查询与相应键的缩放矩阵点积计算得出：

   ![image-20221128152052253](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula2.png)

   注意权重捕获上下文信息并表征特征之间的语义亲和力。将初始的局部上下文特征作为输入特征 Fin = F3。然后，将 F3 送到多层感知器MLP以获得新的更高维特征 F6，它表示查询 Q 和密钥 K。因为只需要计算两者之间的相似性簇，即注意力权重，所以不使用值矩阵 V。之后使用L^2 -normalization 将 F6 中的每个特征向量缩放为单位长度。F6 中每对的点积是特征之间角度的余弦。简化计算，转换归一化的 F6 并使用矩阵乘法：

   ![image-20221128153303361](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula3.png)

   最后，为了将特征相似度转换到概率空间，使用线性变换层来完成特征空间映射。提出了亲和损失来最小化预测亲和矩阵 M' 和地面真值矩阵 M 之间的重建误差。图4显示了基本事实和我们预测的亲和力矩阵的可视化。

   ![image-20221128153455143](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig4.png)

#### 4.3 Loss functions

Our network training is supervised by an efficient loss function containing three components: junction semantic segmentation loss, fine clustering loss and affinity loss: 我们的网络训练由一个有效的损失函数监督，它包含三个组成部分:**连接语义分割损失、精细聚类损失和亲和力损失**:

![image-20221128153646421](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula4.png)

将**语义损失**定义为二元交叉熵损失函数:

![image-20221128153753857](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula5.png)

where yi indicates the ground-truth label, σ is the sigmoid function, and si ∈ S is the predicted score of our network.

**精细聚类**模块为属于局部上下文的每个点pi预测一个分数，使用多重标签交叉熵来度量网络预测与ground-truth 标签дi之间的损失:

![image-20221128154607662](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula6.png)

对**亲和力损失**使用二元交叉熵损失来比较预测的亲和力矩阵 M' 和 ground-truth 矩阵 M：

![image-20221128154842676](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula7and8.png)

We set ω = 0.43,γ = 2 by default in our training task.

#### 4.4 Implementation and Training Details

1. Dataset preparation

   We generated 7, 100 point clouds of trees with ground-truth labels, which were separated into disjoint training models (5, 680 trees), validation models (710 trees), and test models (710 trees). 我们生成了7100个带有地面真实标签的树点云，这些点云被分离为不相交的训练模型(5,680棵树)、验证模型(710棵树)和测试模型(710棵树)。

2. Network training

## TREE MODELING

Our algorithm’s last step aims to provide a compact and fully connected surface representation of tree models, which can be easily used for visualization, procedural editing, or simulation. 算法的最后一步旨在提供树模型的紧凑和完全连接的表面表示，它可以很容易地用于可视化、程序编辑或模拟。

After neural decomposition, each point in the input point cloud is assigned to two attributes: a label of whether it is a junction point or not and a cluster ID of the branch it belongs to. Since the junction regions reveal the most critical topological structure of a tree, we cut the input points into a set of cylindrical branch parts and non-cylindrical junction parts by separating junction points. Each junction part stores the cluster IDs of its nearby connected branches. 经过神经分解后，输入点云中的每个点被分配给两个属性:是否是结点的标签和所属分支的集群ID。由于结点区域揭示了树最关键的拓扑结构，我们通过分离结点将输入点切割成一组圆柱形分支部分和非圆柱形结点部分。每个连接部分存储其附近连接的分支的集群id。

1. Branch reconstruction. 分支重建

   提取分段线性骨架曲线；对于每个骨骼点，计算一个与骨骼曲线正交的剖面曲线；每条剖面曲线都由具有 m 个顶点（m = 10）的平面正多边形表示；通过放样过程将该广义圆柱体转换为表面网格的补丁。

2. Computing the joint points. 计算关节点

   检查连接部分的每个点 p：如果 p 属于分支簇 Bi 并且它的一个或多个邻居点来自另一个分支簇 Bj ，我们称点 p 为边界 Bi 和 Bj 之间的点。 收集所有边界点并计算它们的中心作为该连接部分的连接点。

3. Connecting branches. 连接分支

   全连通骨架图；Hermite曲线。

4. Foliage synthesis. 树叶合成

   自动合成叶。

## RESULTS AND EV ALUATION

#### 6.1 Evaluation

1. Robustness.

   ![image-20221128205702495](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig6.png)

   图 6. 对来自我们测试数据集的两个合成示例的评估，其中我们展示了分解和重建过程的逐步结果。 对于每个示例，从左到右，我们展示了输入点云、连接点检测、初始集群、合并集群、提取的骨架以及我们最终重建的纹理模型。

   ![image-20221128210056905](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig7.png)

   图 7. 从真实数据重建树的几个结果。（使用具有缺失区域、噪声和稀疏性的低质量真实输入重建了几棵树） 从左到右：参考照片、输入点云、我们的树叶分割和分支分解、重建模型以及添加树叶和纹理的渲染结果。 (a)-(c) 中的输入点云分别通过使用 64、66、40 幅图像的多视图立体重建获得。

   We also evaluate the performance of the foliage segmentation network quantitatively with **precision, recall, accuracy, and F1 scores**:通过精度、召回率、准确性和F1分数定量评估叶分割网络的性能：

   ![image-20221128210442759](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\formula9.png)

   ![image-20221128210544730](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig8.png)

   图 8. 验证集的准确性、精确度、召回率和 F1 分数作为树叶分割训练时期的函数。

2. Ablation study. 消融实验

   The scaled cosine distance and focal loss play a crucial role in improving the prediction of the affinity matrix. 缩放余弦距离和焦距损失对改进亲和矩阵的预测起着至关重要的作用。

   ![image-20221128211819014](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig9.png)

   ![image-20221128211857475](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\table1and2.png)

   ![image-20221128212034377](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig10.png)

#### 6.2  Comparison

1. Comparison to the decomposition methods. 与分解方法的比较

   Rand Index (RI) and Normalized Mutual Information (NMI)

   ![image-20221128215219688](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig11.png)

   ![image-20221128215313754](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\table3.png)

2. Comparison to skeletonization methods. 与骨骼化方法的比较

   ![image-20221128221138590](E:\hhhhhe\littleHe\study\paper-notes\notes\fig12.png)

   It demonstrates that the skeleton obtained by our TreePartNet is better to maintain the branching fidelity of trees.这说明我们的TreePartNet获得的骨架能够更好地保持树的分支保真度。

3. Comparison to tree reconstruction methods. 与树重构方法的比较

   ![image-20221128221545579](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig14.png)

   ![image-20221128221610815](E:\hhhhhe\littleHe\study\paper-notes\notes\TreePartNet images\fig15.png)

#### 6.3  Limitations

We present a neural network approach that successfully learns a  cylindrical representation for 3D tree reconstruction from raw point  clouds.  我们提出了一种神经网络方法，该方法成功地从原始点云中学习了用于 3D 树重建的圆柱表示。

1. While producing convincing results, our approach is based  on supervised learning which heavily relies on labeled synthetic  data, thus our capability is limited by the richness of the training  data. 在产生令人信服的结果的同时，我们的方法基于严重依赖标记合成数据的监督学习，因此我们的能力受到训练数据丰富性的限制。 
2. Second, due to the inherent nature of the involved neural  networks, we are unable to handle large-scale point clouds with  hundreds of thousands of points.  Downsampling them to a low  number of points would results in losing many branches.  其次，由于所涉及的神经网络的固有性质，我们无法处理具有数十万个点的大规模点云。 将它们下采样到少量点会导致丢失许多分支。
3. Third,  since we focus on reconstructing branching, our method is more  suitable for trees with distinct branches (e.g. elm, maple, oak).  We  fail to model trees with significant leaf cover (e.g. spruce, fir) or  other forms of plants such as palms, flowers, and climbing plants. 第三，由于我们专注于重建分枝，因此我们的方法更适用于具有不同分枝的树木（例如榆树、枫树、橡树）。 我们未能对具有大量叶盖的树木（例如云杉、冷杉）或其他形式的植物（例如棕榈树、花卉和攀援植物）进行建模。

## CONCLUSION AND FUTURE WORK

In this paper we propose to use a three-fold network architecture  to reconstruct tree skeletons from point clouds. Our approach combines a neural decomposition into local cylindrical shapes with  robust branching detection to yield accurate tree reconstructions. In a post-processing step the elements of the fine grain cylindrical  decomposition are combined to larger generalized cylinders in order to achieve a data-efficient reconstruction. An evaluation shows  that our method is better in reconstructing elements of branching  structures than state-of-the-art methods, our reconstructed trees  meet artificially scanned input models faithfully.

在本文中，我们建议使用三重网络架构从点云重建树骨架。 我们的方法将神经分解为局部圆柱形与稳健的分支检测相结合，以产生准确的树重建。 在后处理步骤中，细粒度圆柱分解的元素被组合成更大的广义圆柱，以实现数据高效的重建。 一项评估表明，我们的方法在重建分支结构元素方面优于最先进的方法，我们重建的树忠实地满足了人工扫描的输入模型。











