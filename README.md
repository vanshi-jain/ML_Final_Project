# ML_Final_Project
This is a final course-end project for EECE 5640 Machine Learning and Pattern Recognition taught by Prof. Deniz Erdogmus

# Motivation and Aim behind this project 
Parkinson's disease is one of the common neurological disorders. This disease is caused by the progressive loss of dopaminergic and other subcortical neurons. The common negative effects are flexed posture, tremors at rest, rigidity, akinesia, postural instability, and motor blocks. Generally, motor blocks affect the patients’ leg during walking and this event is referred to as Freezing of Gait (FOG).

Previously conducted research collects the data from PD patients by using wearable devices. These data can be used to detect pre-FOG, a state before FOG occurs. Thus, a warning can be sent from the wearable device to the emergency contact. FOG detection can be solved by using a machine learning approach. 

The conventionally available machine learning libraries are usually used on a personal computer. However, the model size of the conventional machine learning libraries may not be fit for the small device’s storage. There are several machine learning libraries that were developed to be implemented in small devices. One of those libraries is Edge Machine Learning (EdgeML). This project discusses the comparison between an algorithm called ProtoNN from the EdgeML library with several algorithms from conventional machine learning libraries.

# Brief Proposal
Presented a full AI stack of classification for Parkinson's disease patients and investigated the accuracy and memory usage of each saved model after preprocessing the dataset to recognize gait freeze from wearable acceleration sensors placed on legs and hips. Developed and analyzed several supervised machine learning methods including KNN, Decision Trees, SVM, and Random Forests among other specialized classifiers such as ProtoNN invented by Microsoft India Research. Employed EdgeML based on K-Nearest Neighbour classifiers for maximal compression of the model deployment, acceleration resulting in 95% accuracy with a compression ratio of 20 KB (KNN: ProtoNN) concluding as the best model with a size of 2 KB

# The Edge Machine Learning (EdgeML)
Resource scarce devices and sensors on the Internet of Things (IoT) require a machine learning model to have a small footprint in terms of storage, prediction latency, and energy. The EdgeML meets this requirement to make real-time predictions locally on IoT devices without connecting to
the cloud. This model fits in a few kilobytes.

The EdgeML library contains several algorithms, such as 
- Bonsai: Strong and shallow non-linear tree-based classifier.
- ProtoNN: Prototype-based k-nearest neighbors (kNN) classifier.
- EMI-RNN: Training routine to recover the critical signature from time-series data for faster and more accurate RNN predictions.
- Shallow RNN: A meta-architecture for training RNNs that can be applied to streaming data. FastRNN & FastGRNN - FastCells: Fast, Accurate, Stable, and Tiny (Gated) RNN cells. 
- DROCC: Deep Robust One-Class Classfiication for training robust anomaly detectors. 
- RNNPool: An efficient non-linear pooling operator for RAM constrained inference.

These algorithms can train models for classical supervised learning problems with memory requirements that are orders of magnitude lower than other modern ML algorithms. The trained models can be loaded onto edge devices such as IoT devices/sensors and used to make fast and accurate predictions completely offline. The EdgeML's github link is [here](https://github.com/microsoft/EdgeML)

# Dataset
This project uses the Daphnet Freezing of Gait (FoG) Dataset [3] which is devised to benchmark automatic methods to recognize gait freeze from wearable acceleration sensors placed on legs and hip. The dataset was recorded in the lab with an emphasis on generating many freeze events. Users performed three kinds of tasks: straight-line walking, walking with numerous turns, and activity of daily living (ADL) tasks. This dataset was obtained from [here](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait)

# Data Preprocessing
The dataset has several attributes as shown in Table 1. We use column code to differentiate attributes between columns. Columns 1-10 are the attributes, and column 11 is the class labels. The data were collected based on time series, which were stored in the first column in milliseconds. The acceleration data were stored in columns 2-10. The class label is defined in column 11, which has 3 labels, 0 means the user is performing activities unrelated to the
experimental protocol, 1 means activities during the experiments, and 2 means freeze occurs. The snippet data are shown in Table 2. The Wearable Sensor Unit provides the input unit with 3 tri-axial accelerometers placed on the Ankle (A), Leg(L), and Torso(T). Each channel constitutes a signal corresponding to a single axis from an accelerometer. Thus, a total of 9 channels are present and they are represented by the set Γ as shown,

 Γ = {AX, AY, AZ, LX, LY, LZ, TX, TY, TZ}.
| Column | Description                                                                                                                                                                                                                                                                    | Code   |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| 1      | Time of sample in millisecond                                                                                                                                                                                                                                                  | time   |
| 2      | Ankle (shank) acceleration - horizontal forward acceleration [mg]                                                                                                                                                                                                              | A_F    |
| 3      | Ankle (shank) acceleration - vertical [mg]                                                                                                                                                                                                                                     | A_V    |
| 4      | Ankle (shank) acceleration - horizontal lateral [mg]                                                                                                                                                                                                                           | A_L    |
| 5      | Upper leg (thigh) acceleration - horizontal forward acceleration [mg]                                                                                                                                                                                                          | L_F    |
| 6      | Upper leg (thigh) acceleration - vertical [mg]                                                                                                                                                                                                                                 | L_V    |
| 7      | Upper leg (thigh) acceleration - horizontal lateral [mg]                                                                                                                                                                                                                       | L_L    |
| 8      | Trunk acceleration - horizontal forward acceleration [mg]                                                                                                                                                                                                                      | T_F    |
| 9      | Trunk acceleration - vertical [mg]                                                                                                                                                                                                                                             | T_V    |
| 10     | Trunk acceleration - horizontal lateral [mg]                                                                                                                                                                                                                                   | T_L    |
| 11     | Annotation [0, 1, or 2]  The meaning of the annotations are as follows: - 0: not part of the experiment. the user is performing activities unrelated to the experimental protocol, such as debriefing - 1: experiment, no freeze (can be any of stand, walk, turn) - 2: freeze | action |

# Feature Extraction
As the project task is detecting the pre-FOG event, we cannot directly use the raw data. Therefore, we need to conduct data preprocessing to determine the pre-FOG condition. Previous research assumed that the gait cannot enter the FOG state directly from walking state. The identification of segments of pre-FOG data is valuable both for FOG detection and prediction. We also used time-domain features to detect the pre-FOG events. The following features are extracted from the time domain:
- Mean
- Standard deviation
- Variance
- Maximum allowed variance
- Root mean square

The following features are extracted from the frequency domain:
- Freeze Index
- Power
- Energy
- Entropy
- Peak Frequency

# ProtoNN Model
ProtoNN can address the above-mentioned concerns by using three key ideas:
1. Sparse low-d projection: project the entire data in low-d using a sparse projection matrix that is jointly learned to provide good accuracy in the projected space.
2. Prototypes: learn prototypes to represent the entire training dataset. Moreover, learn labels for each prototype to further boost accuracy. This provides additional flexibility and allows us to seamlessly generalize ProtoNN for multi-label or ranking problems.
3. Joint optimization: learn the projection matrix jointly with the prototypes and their labels. Explicit sparsity constraints are imposed on our parameters during the optimization itself so that we can obtain an optimal model within the given model size de-facto, instead of post-facto
pruning to force the model to fit in memory.

# Results
This project compared Pre-FOG detection developed by using ProtoNN with other conventional machine learning algorithms, such as Support Vector Machine (SVM), Decision Tree, Random Forest, and K-Nearest Neighbor (KNN). This project used existing code that implements ProtoNN to detect pre-FOG. However, since the code was not fully works, we need to do bug fixing and modification of the existing code. The modification was necessary because we need to
adjust some incompatible codes and libraries in order to successfully run in our computer. There are two main focuses in this comparison, such as performance and model size. The results are shown in Table 3. Column “Acc” means the accuracy obtained by each classifier. The accuracy was calculated by using the equation (1). Precision and recall were calculated by using equations (2) and (3), respectively. Terms true positives (tp), true negatives (tn), false positives (fp), and false negatives (fn) compare the results of the classifier under test with external judgments. 


| No. | Algorithm                    | Acc. | Precision | Precision |         | Recall |        |         | Model Size |
|-----|------------------------------|------|:---------:|-----------|---------|:------:|--------|---------|------------|
|     |                              |      | NA        | No FOG    | Pre-FOG | NA     | No FOG | Pre-FOG |            |
| 1   | Support Vector Machine (SVM) | 0.72 | -         | 0.75      | 0.35    | -      | 0.93   | 0.1     | 3.99 MB    |
| 2   | Decision Tree                | 0.98 | 0.99      | 0.99      | 0.91    | 1      | 0.99   | 0.87    | 218 KB     |
| 3   | Random Forest                | 0.98 | 0.99      | 0.97      | 0.96    | 1      | 0.98   | 0.74    | 844 KB     |
| 4   | K-Nearest Neighbor (KNN)     | 0.92 | 0.95      | 0.81      | 0.53    | 0.97   | 0.79   | 0.38    | 13.8 MB    |
| 5   | ProtoNN                      | 0.95 | -         | 0.99      | 0.91    | -      | 0.90   | 0.99    | 2.18 KB    |

* NA = No Activity
* No FOG = No Freezing of Gait
* Pre-FOG = Pre-Freezing of Gait
