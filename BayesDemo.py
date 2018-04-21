In this paper, we propose a context-based conceptualization recurrent convolutional neural network called 2ConRCNN, which combines global context and local context together with external knowledge for short text classification. The architecture of our propose model in shown in Figure 1. It consists of four parts: BGRU Layer, a global context-based attention layer, a convolution layer, and a max pooling layer.
3.1 Bidirectional GRU Layer
For the text classification task, it is beneficial to take the left context as well as the right context into consideration since they both are helpful for interpreting the meaning of the current word. In this paper, we utilize bidirectional GRU to combine a word’s the past and future information together to represent its global context. In this way, we can obtain a more precise word meaning. The gated recurrent unit (Bahdanau et al., 2014) (GRU) is a variant of RNNs which takes a sequence of inputs and uses a gating mechanism to compute a hidden state vector based on current input and previous entire history of inputs at each time step. The hidden state at time t can be computed recursively as follows:
(1) ht=(1-zt)ht-1+ztht
The hidden state ht at time t is a interpolation between previous state ht-1 and the candidate state ht. The update gate zt decides what information to keep from previous state and what information to add from the current input. It is computed by:
zt = б(Wzxt +Uzht-1 +bz)
whereбis a sigmoid function, xt is the input at time t. WZ,UZ are weight matrices and bz is a bias which are learned.
The candidate state ht is computed similarily to the way of traditional recurrent neural network (RNN) :
ht = tanh(Whxt +rt⊙(Uhht-1) +bh)
where tanh is a non-linear function, rt is the reset gate and ⊙ denotes element-wise multiplication. When rt is close to zero, then the reset gate forgets the previously computed state. The reset gate rt is updated similarily to the update gate as follows:
rt = б(Wzxt +Uzht-1 +bz)
The bidirectional GRU is essentially a combination of two GRUs that one operates in the forward direction and the other operates in the backward direction. In this way, it leads to two hidden states ht and ht at time t, which can be viewed as a summary of the left context and right context respectively. Their concatenation ht = [ht;ht] provides a summary of global context around the input at time t.
3.2 Context-based Conceptualization 
Conceptualization map a word into an explicit semantic space composed by concepts in knowledge base. In this section, we firstly describe how to 
3.2.1 Knowledge Base Concepts Representation

3.2.2 Concept Attention
Suppose a document is expressed as D = {x1,x2,…,xn}, where n is the number of words it contains. In the previous step, for a word xi, we can get its concept vector Ci ={<ci1: vi1>,…,<cij: vij>}. Not all concepts care related to a word given the context. Hence, we employ an attention-based neural network to dynamically extract such concepts that are important or relevant to the given context and aggregate the representations of those concepts to form a context-based concept vector. Specifically,
uij = tanh(Wvij+b)
aij = exp ()
mi = 
where . In this way, we get a context-based concept vector to represent the external knowledge related to a word given the context. For a word at time t, We combine its global context h with its concept representation called word-concept embedding together and feed them into a convolutional neural network to perform short text classification.
3.3 Convolutional Neural Networks	
A matric H = {h1,h2,h3,…,hl} is obtained from ….


