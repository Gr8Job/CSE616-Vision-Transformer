# CSE616-Vision-Transformer
CSE616-Vision-Transformer

 
1.	Introduction:	

This report explains the experiment steps of the “Vision Transformer Implementation Project” and discusses the obtained results. The architecture is implemented based on [1], [2], [3] and [4]. The model hyper parameters were selected based on [3] and the CLS classification head was implemented based [4].

The repository includes the following files:

Vision_Transformer.ipynb: This is the model notebook downloaded from colab. You could run in your colab acount. If you wish to load the saved model object or to use the last saved model weights and optimizer status, you will need to download these objects from the file Saved Model and Weights.txt.

Saved Model and Weights.txt: This file has the saved model object, the last saved model weights and the last saved optimizer status.

Transformer_Block_Parameter_Calculations.ipynb: This files illustarates the calculation of the learnable parameters included within the transformer block.

Run_Results.ipynb: This file has all results of the 12 runs.

2.	Dataset Description:

CIFAR100 dataset was used to test the Model. CIFAR100 dataset is publicly available over the internet. It consists of has 100 image classes. Each class has 500 training images and 100 testing images. In total, there exists 50,000 training image and 10,000 testing images. In our implementation, we have used the Tensorflow available dataset object to make use of the available TFDS features like dataset prefetch and dataset shuffling. The following snapshot shows the steps to download the dataset and divide between training, validation and testing splits.
 
3.	Dataset Preprocessing:

The dataset was preprocessed by unifying the size of all images to the same size (defined by the variable IMG_SIZE which was chosen to be 72) and values for all the channels were scaled to be from 0 to 1. A normalization layer (scaling each image to have a zero mean and unity variance) is implemented within the model to make use of the GPU processing if available.
 
4.	Model Hyper-Parameters:

These are hyper parameters which define the model behavior. In this section we explain these parameters the model architecture is presented in the follow sections.
IMG_SIZE: This is image size the model expects to receive as an input. All input images should be normalized to this size.
PATCH_SIZE: Each input image is divided into smaller images (Patches) where each patch has a fixed size of PATCH_SIZE x PATCH_SIZE.
BATCH_SIZE: This is the mini-batch size used by the optimization algorithm.
PATCH_PROJECTION_SIZE: Each sub-image (patch) is flattened and then projected (pass through a linear transformation) to be presented by a vector of the length of the PATCH_PROJECTION.
NUM_ATTENTION_HEADS: This is the number of attention heads within the transformer bloc.
NUM_TRANSFORMER_BLOCKS: This is the number of transformer blocks.
 
 
5.	Model Architecture:

 The model consists of basic 7 functional layers.  The model starts by dividing images into patches, followed by embedding these patches into the model dimension, followed by adding the classification vector CLS, followed by adding the position embedding, followed by adding the transformer blocks and finally adding the classification head. Each layer will be explained in more details in the following subsection. 
 
 
5.1	Overall Model Summary:

The model has around 3 Million learnable parameters and the breakdown for each layer is explained for the section related to each layer.
 
5.2	Data Augmentation Layer:

In this layer, we apply different data augmentation techniques in order to avoid overfitting like random flip, random rotation and random zoom.
 
5.3	Extract Patches Layer: 

In this layer, each image is divided in sub-images (based on the predefined patch size hyper parameter) and these sub-images are flattened so that we could end up by converting each image into a sequence of vectors (patches). When having image size of 72x72x3 and a patch size of 6, we will have a sequence of 144 small images of size 6x6x3. These sub-images are flattened to have a sequence of 144 vectors each is a length of 108 element.  
  
5.4	Patch Embedding Layer: 

In this layer, each vector is projected to another vector of length of projection size (64 elements in our model). This is applied by using a fully connected layer of 64 units and unit activation function to act as a learnable linear transformation. The same transformation is applied over all vectors within the sequence and hence leads to (108*64+64) learnable parameters or 6976 parameter. 
The input size is changed to be a sequence of 144 vectors each consisting of 64 elements.
    
5.5	CLS Embedding Layer: 

In order to use the model for classification, we used the same idea of BERT and a new vector of 64 (CLS) learnable parameters was added to the sequence. This vector will be used later for image classification. This changes the input size to be a sequence of 145 vector each consisting of 64 elements.
 
5.6	Position Embedding Layer: 

In this layer, we add a number to each patch to represent the position of each patch within the sequence (the original). This is done to avoid losing the spatial information encoded in the position of each patch compared to other patches. The layer simply encodes the position information (from 1 to 145) into a vector of 64 element and add this vector to the corresponding patch. There’s no learnable parameters in this layer
 
5.7	Transformer Block Layer: 

This layer implements the transformer block which consists of layer normalization, multi-head self-attention layer, skip or residual connections and finally two fully connected layers with dropout and gelu non-linear activation function. 
 
For the multi-head attention layer, it split the input sequence into 4 different sequences (each sequence is a different attention head). Each sequence is used to train 3 Fully connected layers (each is 64 unit width) to implement the learnable Q, K & V linear transformations used in attention calculations. This step adds 4 (number of heads) *3 (Q, K & V linear transformations) *(64*64+64 FC layer with 64 units) leading to 49,920 learnable parameters. 
After self-attention is calculated for each head, the output of each attention head is concatenated (leading to a vector of 4*64 =256 elements). Finally, that vector is projected back to 64 element size by a new fully connected layer (256*64+64). In total, we will need 49,920+(256*64+64) or 66,368 learnable parameter for each multi-head attention layer. 
Adding the two dense layers and the normalization layer will lead to 83,200 learnable parameters for each transformer block.
 
5.8	Classification Head Layer: 

Finally, a classification head is added consisting of two fully connected layers. Finally, a softmax layer is added with the number of units equal to 100 matching the number of classes. Only the first vector of the sequence (corresponding to the CLS added vector to the sequence) is used for the classification task.
With the vector size of 64, first dense layer of 2048 units, second dense layer of 1024 units and softmax layer with 100 units, this leaves us with the biggest part of the model of (64*2048+2048 + 2048*1024+1024+1024*100+100) or 2,333,796 learnable parameters.
 
6. Experiment Results:

We used 60 Epochs  to train the model over Colab open servers. As a result of the Colab training limitations, the experiment was performed over 12 runs and each run included 5 epochs. An AdamW optimizer was used in the experiment as proposed in [3] with weight decay of 0.0001. Learning rate was set of 0.001 which was dropped to 0.0001 after the 50th epoch.
The model was able to achieve 40% classification accuracy over the testing split and 44% accuracy over the training data. Decreasing the learning rate after the 50th epoch leaded to a performance gain of 2%.  Each epoch consuming around 50 minutes or each run consumed around 5 hours. In total, the model needed around total of 60 hours for training.


7. References:

[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

[2] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[3] https://keras.io/examples/vision/image_classification_with_vision_transformer/

[4] https://medium.com/geekculture/vision-transformer-tensorflow-82ef13a9279

[5] https://github.com/Gr8Job/CSE616-Vision-Transformer



