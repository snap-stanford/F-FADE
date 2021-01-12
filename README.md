## F-FADE: Frequency Factorization for Anomaly Detection in Edge Streams (ACM WSDM 2021)
#### Authors: [Yen-Yu Chang](https://yuyuchang.github.io/), [Pan Li](https://sites.google.com/view/panli-uiuc/home), [Rok Sosic](https://scholar.google.com/citations?user=xlZ4YJcAAAAJ&hl=en&oi=ao), [M. H. Afif](), [Marco Schweighauser](), [Jure Leskovec](https://cs.stanford.edu/people/jure/)
#### [Link to the paper]()
#### [Link to the slides]()
#### [Brief video explanation]()

### Abstract
Edge streams are commonly used to capture interactions in dynamic networks, such as email, social, or computer networks. The problem of detecting anomalies or rare events in edge streams has a wide range of applications. However, it presents many challenges due to lack of labels, a highly dynamic nature of interactions, and the entanglement of temporal and structural changes in the network. Current methods are limited in their ability to address the above challenges and to efficiently process a large number of interactions. Here, we propose **F-FADE**, a new approach for detection of anomalies in edge streams, which uses a novel frequency-factorization technique to efficiently model the time-evolving distributions of frequencies of interactions between node-pairs. The anomalies are then determined based on the likelihood of the observed frequency of each incoming interaction. F-FADE is able to handle in an online streaming setting with a broad variety of anomalies with temporal and structural changes, while requiring only constant memory. Our experiments on one synthetic and six real-world dynamic networks show that F-FADE achieves state of the art performance and may detect anomalies that previous methods are unable to find.

### Datasets
Links to datasets used in the paper:
- [DARPA](https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset)
- [ENRON](https://www.cs.cmu.edu/~enron/)
- [DBLP](https://dblp.uni-trier.de/xml/)

### Dataset format
F-FADE expects the input edge streams to be stored in a single file containing the following four columns in order:
1. `time (int or float)`: time stamp of the edge
2. `source (int)`: source ID of the edge
3. `destination (int)`: destination ID of the edge
4. `label (int)`: label of the edge (1 denotes anomalies, 0 otherwise)

Each line represents an edge. Edge should be sorted in non-decreasing order of their time, source, and destination, and the delimiter should be " ".

### Code setup and Requirements

Recent versions of PyTorch, numpy, scipy, and sklearn. You can install all the required packages using the following command:
```
    $ pip install -r requirements.txt
```

Put test datasets into the `data/` directory.

To initialize the directory needed to store models and outputs, use the following command. This will create `result/` directory.
```
    $ mkdir result
```

### Running the F-FADE code
Run `./run.sh` or the following command to test on DARPA dataset with default hyper-parameters. Check the accuracy (AUC) and anomaly scores in `result/`.
```
    $ python3 main.py --dataset ./data/darpa.txt --embedding_size 200 --t_setup 8000 --W_upd 720 --T_th 120 --alpha 0.999 --epoch 5 --online_train_steps 10 --M 100 --model_dir ./result/
```

### Command line options
This code can be given the following command-line arguments:
  * `--dataset`: the path of the testing data.
  * `--embedding_size`: the number of dimensions of node embeddings.
  * `--t_setup`: the time to setup the model.
  * `--W_upd`: the time interval for model update.
  * `--T_th`: the cut-off threshold of the time to record, which is the inverse of the cut-off threshold frequency.
  * `--alpha`: the decay rate when updating frequency.
  * `--epoch`: the number of training epochs.
  * `--online_train_steps`: the number of online training epochs.
  * `--M`: the upper limit of memory size.
  * `--model_dir`: the directory needed to store models and outputs.
