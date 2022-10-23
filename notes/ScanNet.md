http://www.scan-net.org/

https://github.com/ScanNet/ScanNet

## Abstract

- ScanNet:  An RGB-D video dataset containing 2.5M views in 1513 scenes annotated with ==3D camera poses==, ==surface reconstructions==, and ==semantic segmentations==. ScanNet：一个 RGB-D 视频数据集，包含 1513 个场景中的 250 万个视图，并用 3D 相机姿势、表面重建和语义分割进行注释。


- Collect Data: We designed an easy-to-use and scalable RGB-D ==capture system== that includes automated surface reconstruction and crowdsourced semantic annotation. 收集数据：我们设计了一个易于使用且可扩展的 RGB-D 捕获系统，其中包括自动表面重建和众包语义注释。


- Result: We show that using this data helps achieve state-of-the-art performance on several 3D scene understanding tasks, including 3D object classification, semantic voxel labeling, and CAD model retrieval. 结果：我们表明，使用这些数据有助于在几个 3D 场景理解任务上实现最先进的性能，包括 3D 对象分类、语义体素标记和 CAD 模型检索。


## Introduciton

- #### Current: 

- 1. Many of the current RGB-D datasets are orders of magnitude smaller than their 2D counterparts. 许多当前的RGB-D数据集比其2D对应数据项小几个数量级。
  2. Labels are added manually by expert users (typically by the paper authors) which limits their overall size and scalability. 手动添加注释限制了标签的整体大小和可伸缩性。

- #### Result: 

- ==ScanNet==, a dataset of richlyannotated RGB-D scans of real-world environments containing 2.5M RGB-D images in 1513 scans acquired in 707 distinct spaces. 一个包含丰富注释的真实世界环境的 RGB-D 扫描数据集，包含在 707 个不同空间中获得的 1513 次扫描中的 250 万张 RGB-D 图像。

- #### Advantages and disadvantages: 

- 1. The sheer magnitude of this dataset is larger than any other. 该数据集的绝对规模比任何其他数据集都大。 
  2. What makes it particularly valuable for research in scene understanding is its ==annotation with estimated calibration parameters, camera poses, 3D surface reconstructions, textured meshes, dense object-level semantic segmentations, and aligned CAD models==. 它对场景理解研究特别有价值的是它的注释与估计的校准参数、相机姿态、3D 表面重建、纹理网格、密集对象级语义分割和对齐的 CAD 模型。

- #### Problem and Solution: 

- 1. Pro:  How can we design a framework that allows many people to collect and annotate large amounts of RGB-D data? 我们如何设计一个允许多人收集和注释大量RGB-D数据的框架？

     Sol: Built a capture pipeline to help novices acquire semantically-labeled 3D models of scenes. 建立了一个捕获管道来帮助新手获取场景的语义标记的3D模型。The paper discusses our study of these issues and documents our experience with scaling up RGB-D scan collection (20 people) and annotation (500 crowd workers).该论文讨论了我们对这些问题的研究，并记录了我们在扩大 RGB-D 扫描收集（20 人）和注释（500 名群众工作者）方面的经验。

  2. Pro: Can we use the rich annotations and data quantity provided in ScanNet to learn better 3D models for scene understanding? 我们可以利用ScanNet提供的丰富的标注和数据量来学习更好的3D模型来进行场景理解吗？

     Sol: we trained 3D deep networks with the data provided by ScanNet and tested their performance on several scene understanding tasks, including 3D object classifification, semantic voxel labeling, and CAD model retrieval. 我们用ScanNet提供的数据训练了3D深度网络，并在几个场景理解任务上测试了它们的性能，这些任务包括3D对象分类，语义体素标记和CAD模型检索。For the semantic voxel labeling task, we introduce a new volumetric CNN architecture. 对于语义体素标记任务，我们引入了一种新的体积CNN体系结构。

- #### contributions

- 1. A large 3D dataset containing 1513 RGB-D scans of over 707 unique indoor environments with estimated camera parameters, surface reconstructions, textured meshes, semantic segmentations. We also provide CAD model placements for a subset of the scans. 大型3D数据集，包含对707多个独特室内环境的1513 RGB-D扫描，并具有估计的相机参数，表面重建，纹理化网格，语义分割。我们还为扫描的一部分提供CAD模型放置。
  2. A design for effificient 3D data capture and annotation suitable for novice users. 适合新手用户的高效3D数据捕获和注释设计。
  3. New RGB-D benchmarks and improved results for state-of-the art machine learning methods on 3D object classifification, semantic voxel labeling, and CAD model retrieval. 在3D对象分类，语义体素标记和CAD模型检索方面的最新机器学习方法，新的RGB-D基准和改进的结果。
  4. A complete open source acquisition and annotation framework for dense RGB-D reconstructions. 用于密集RGB-D重建的完整的开源获取和注释框架。

## Previous Work

- NYU v2; SUN RGB-D; SUN3D; Armeni's indoor dataset
- SceneNN; PiGraphs (most similar to ours)
- In contrast, we design our ==RGB-D acquisition framework== specifically for ==ease-of-use== by untrained users and for ==scalable processing== through crowdsourcing. 相比之下，我们设计的RGB-D采集框架专门针对未经训练的用户的易用性，以及通过众包进行可扩展处理。This allows us to acquire a ==significantly larger dataset== with ==more annotations== (currently, 1513 sequences are reconstructed and labeled). 这允许我们获得一个显著更大的数据集，有更多的注释(目前，1513个序列被重建和标记)。

## Dataset Acquisition Framework

- #### Main goal: 

  1. 本节的重心：the design of the framework used to acquire the ScanNet dataset 用于获取ScanNet数据集框架的设计
  2. 框架设计的目标：Our main goal driving the design of our framework was to allow untrained users to capture semantically labeled surfaces of indoor scenes with commodity hardware. 我们驱动框架设计的主要目标是允许未经训练的用户使用商用硬件捕获室内场景的语义标记表面。Thus the RGB-D scanning system must be trivial to use, the data processing robust and automatic, the semantic annotations crowdsourced, and the flow of data through the system handled by a tracking server. 因此，RGB-D扫描系统必须使用简单，数据处理健壮且自动化，语义注释众包，通过系统的数据流由跟踪服务器处理。

- #### RGB-D Scanning

  1. Hardware 硬件: Structure sensor结构传感器 and iPad Air2; Depth frames are captured at a resolution of 640 × 480 and color at 1296 × 968 pixels.深度帧捕获分辨率为640 × 480像素，颜色为1296 × 968像素。
  2. Calibration 校准: Obtain intrinsic parameters for both depth and color sensors, and an extrinsic transformation of depth to color. 获得深度和颜色传感器的内在参数，以及深度到颜色的外在转换。
  3. User Interface 用户界面: We designed an iOS app with a simple live RGB-D video capture UI. 设计了一个带有简单的实时RGB-D视频捕捉UI的iOS应用程序。
  4. Storage 存储: We store scans as compressed RGB-D data on the device flash memory so that a stable internet connection is not required during scanning. 将扫描作为压缩RGB-D数据存储在设备闪存中，这样扫描期间不需要稳定的互联网连接。

- #### Surface Reconstruction

  1. Dense Reconstruction 密度重建: volumetric fusion体积融合; BundleFusion system; VoxelHashing; Marching Cubes; 
  1. Orientation 方向 取向: We automatically align it and all camera poses to a common coordinate frame with the z-axis as the up vector, and the xy plane aligned with the floor plane. 自动对齐它和所有相机姿势到一个公共坐标框架，以z轴为上向量，xy平面与地板平面对齐。
  1. Validation 验证: We automatically discard scan sequences that are short, have high residual reconstruction error, or have low percentage of aligned frames. We then manually check for and discard reconstructions with noticeable misalignments. 自动丢弃短的扫描序列、残差重构误差高的扫描序列或对齐帧百分比低的扫描序列；手动检查并丢弃有明显偏差的重构。

- #### Semantic Annotation

  1. Instance-level Semantic Labeling 实例级语义标签: 
     - Our first annotation step is to obtain a set of object instance-level labels directly on each reconstructed 3D surface mesh. 第一个注释步骤是在每个重构的3D表面网格上直接获得一组对象实例级标签。
     - WebGL interface: over-segmentation 过分割; Crowd worker: annotate
     - Challenge1: To enable efficient annotation by workers who have no prior experience with the task, or 3D interfaces in general. 让没有任务经验的工作人员或一般 3D 界面的工作人员能够进行有效的注释。 A simple painting metaphor
     - Challenge2: To allow for freeform text labels, to reduce the inherent bias and scalability issues of pre-selected label lists. 允许使用自由格式的文本标签，以减少预选标签列表的固有偏差和可伸缩性问题。To guide users for consistency and coverage of basic object types. 指导用户保持基本对象类型的一致性和覆盖率。The interface provides autocomplete functionality over all labels 该接口提供了对所有标签的自动完成功能
     - Several additional design details: A simple distance check for connectedness is used to disallow labeling of disconnected surfaces with the same label 使用一个简单的连接距离检查来禁止用相同的标签标记断开的表面; We first show a full turntable rotation of each reconstruction and instruct workers to change the view using a rotating turntable metaphor. 我们首先展示了每个重建的完整转盘旋转，并指示工人使用旋转转盘比喻来改变视图。

  2. CAD Model Retrieval and Alignment CAD模型检索与对齐: 
     - A crowd worker was given a reconstruction already annotated with object instances and asked to place appropriate 3D CAD models to represent major objects in the scene. 众包工作者需要在一个已经标注了对象实例的重构中放置适当的3D CAD模型来表示场景中的主要对象。
     - An assisted object retrieval interface 辅助对象检索界面: 在ShapeNet CAD模型数据集中进行搜索具有相同类别标签的 CAD 模型；Collect sets of CAD models aligned to each ScanNet reconstruction 收集与每个ScanNet重构对齐的CAD模型集
     - The main limitation of this interface is due to the mismatch between the corpus of available CAD models and the objects observed in the ScanNet scans. 该接口的主要限制是由于可用CAD模型的语料库与ScanNet扫描中观察到的对象之间的不匹配。

- #### ScanNet Dataset

  1. we summarize the data we collected using our framework to establish the ScanNet dataset. 总结使用我们的框架来建立ScanNet数据集所收集的数据。
  2. Each scan has been annotated with instance-level semantic category labels through our crowdsourcing task. In total, we deployed 3,391 annotation tasks to annotate all 1513 scans. 通过我们的众包任务，每次扫描都使用实例级语义类别标签进行注释。 我们总共部署了 3,391 个注释任务来注释所有 1513 个扫描。
  3. We have processed all the NYU v2 RGB-D sequences with our framework. 使用我们的框架处理了所有NYU v2 RGB-D序列。获得了NYU v2空间的一组密集重构。
  4. We also deployed the CAD model alignment crowdsourcing task to collect a total of 107 virtual scene interpretations consisting of aligned ShapeNet models placed on a subset of 52 ScanNet scans by 106 workers. 部署了CAD模型对齐众包任务，以收集总共107个虚拟场景解释，包括由106名工作人员放置在52个ScanNet扫描子集上的对齐ShapeNet模型。


## Tasks and Benchmarks

-  We describe the three tasks we developed as benchmarks for demonstrating the value of ScanNet data. 我们将描述我们开发的三个任务，作为展示 ScanNet 数据价值的基准。

-  #### 3D Object Classifification

  1. Use real-world RGBD input for both training and test sets. 使用现实世界的RGBD输入进行训练和测试集。The goal of the task is to classify the object represented by a set of scanned points within a given bounding box. 该任务的目标是对给定边界框内的一组扫描点表示的对象进行分类。For this benchmark, we use 17 categories, with 9, 677 train instances and 2, 606 test instances. 对于这个基准测试，我们使用17个类别，包括9677个训练实例和2606个测试实例。

  2. Network and training

     3D Network-in-Network, without the multi-orientation pooling step. 没有多方向池化步骤

     We use an SGD solver with learning rate 0.01 and momentum 0.9, decaying the learning rate by half every 20 epochs, and training the model for 200 epochs. 我们使用学习率为0.01、动量为0.9的SGD求解器，每20个周期学习率衰减一半，训练模型200个周期。

  3. Benchmark performance

     3D CNN

     ![image-20221020171613316](E:\hhhhhe\littlehe\study\paper-notes\notes\image-20221020171613316.png)

     These results can be slightly improved when mixing training data of ScanNet with partial scans of ShapeNet (last row). 当将ScanNet的训练数据与ShapeNet的部分扫描(最后一行)混合在一起时，这些结果可以略有改善。

-  #### Semantic Voxel Labeling

  1. With our data, we can extend this task to 3D, where the goal is to predict the semantic object label on a per-voxel basis. 有了我们的数据，我们可以将==语义分割==任务扩展到3D，其目标是预测每体素基础上的语义对象标签。

  2. Data Generation

     Across ScanNet, we generate 93, 721 subvolume examples for training, augmented by 8 rotations each (i.e., 749, 768 training samples), from 1201 training scenes. In addition, we extract 18, 750 sample volumes for testing, which are also augmented by 8 rotations each (i.e., 150, 000 test samples) from 312 test scenes. We have 20 object class labels plus 1 class for free space. 通过ScanNet，我们从1201个训练场景中生成了93,721个子卷示例用于训练，每个子卷示例增加了8个旋转(即749,768个训练样本)。此外，我们提取了18750个样本体积用于测试，还从312个测试场景中增加了8个循环(即150000个测试样本)。我们有20个对象类标签加上一个类的空闲空间。

  3. Network and training

     We propose a network which predicts class labels for a column of voxels in a scene according to the occupancy characteristics of the voxels’ neighborhood. 提出了一种根据体素邻域占用特征预测场景中一列体素类标签的网络。

  4. Quantitative Results

     The goal of this task is to predict semantic labels for all visible surface voxels in a given 3D scene; i.e., every voxel on a visible surface receives one of the 20 object class labels. 该任务的目标是预测给定3D场景中所有可见表面体素的语义标签；也就是说，一个可见表面上的每个体素都接收20个对象类标签中的一个。
     
     We use NYU2 labels, and list voxel classification results on ScanNet in Table 7. We achieve an voxel classification accuracy of 73.0% over the set of 312 test scenes, which is based purely on the geometric input (no color is used). 我们使用NYU2标签，并在表7中列出ScanNet上的体素分类结果。在312个测试场景的集合上，我们实现了73.0%的体素分类精度，这纯粹是基于几何输入(不使用颜色)。

-  #### 3D Object Retrieval

   1. "3D ShapeNets: A Deep Representation for Volumetric Shapes"

      For retrieval, we use L2 distance to measure the similarity of the shapes between each pair of testing samples. Given a query from the test set, a ranked list of the remaining test data is returned according to the similarity measure. We evaluate retrieval algorithms using two metrics: (1) mean area under precision-recall curve (AUC) for all the testing queries4; (2) mean average precision (MAP) where AP is defined as the average precision each time a positive sample is returned. 对于检索，我们使用 L2 距离（欧氏距离）来衡量每对测试样本之间形状的相似性。给定来自测试集的查询，根据相似性度量返回剩余测试数据的排序列表。我们使用两个指标评估检索算法：（1）所有测试查询的精确召回曲线下的平均面积（AUC）； (2) 平均精度 (MAP) 其中 AP 定义为每次返回正样本时的平均精度。
   
   2. Retrieval of similar CAD models given (potentially partial) RGB-D scans. 检索给定(可能是部分)RGB-D扫描的相似CAD模型。
   
      A shape embedding where a feature descriptor defines geometric similarity between shapes 形状嵌入，其中特征描述符定义了形状之间的几何相似性
   
   3. the volumetric shape classification network 体积形状分类网络
   
      Nearest neighbors are retrieved based on the ℓ2 distance between the extracted feature descriptors, and measured against the ground truth provided by the CAD model retrieval task. 根据提取的特征描述符之间的距离ℓ2检索最近邻，并与CAD模型检索任务提供的地面真值进行度量。
   
   4. Training on both ==ShapeNet== and ==ScanNet== together is able to find an embedding of shape similarities between both data modalities, resulting in much higher retrieval accuracy. 同时在ShapeNet和ScanNet上进行训练，可以发现两种数据形态之间的形状相似性嵌入，从而获得更高的检索精度。
   

##  Conclusion

- This paper introduces ==ScanNe==t: a large-scale RGBD dataset of 1513 scans with surface reconstructions, instance-level object category annotations, and 3D CAD model placements.  本文介绍了ScanNet：一个包含1513次扫描的大型RGBD数据集，具有表面重建、实例级对象类别注释和3D CAD模型放置。
- To make the collection of this data possible, we designed a scalable RGB-D acquisition and semantic annotation framework that we provide for the benefit of the community. 为了使这些数据的收集成为可能，我们设计了一个可伸缩的RGB-D采集和语义注释框架，我们为社区提供了这个框架。



