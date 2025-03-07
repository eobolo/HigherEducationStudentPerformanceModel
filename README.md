![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-Workflow-blue.svg) ![Python](https://img.shields.io/badge/Python-90%25-green.svg) ![Contributors](https://img.shields.io/badge/contributors-1-orange.svg)  

# Problem Statement:
Predicting student academic performance using Machine Learning (ML) Models is crucial for identifying student at risk early. Accurate models trained on previously existly performance dataset of students can help institutions provide timely interventions to improve student success. However, challenges remain in optimizing these predictive models for different educational contexts.

# Data Source: Kaggle 
You can access the dataset from Kaggle:  
[Higher Education Students Performance Evaluation](https://www.kaggle.com/datasets/csafrit2/higher-education-students-performance-evaluation?select=student_prediction.csv)



| **Train Instance** | **Engineer Name** | **Regularizer** | **Optimizer** | **Early Stopping** | **Dropout Rate** | **Accuracy** | **F1 Score** | **Recall** | **Precision** | **Train Loss** | **Val Loss** | **Test Loss** | **Epochs** |  
|--------------------|------------------|---------------|-------------|----------------|--------------|-----------|---------|--------|-----------|------------|-----------|-----------|--------|  
| First Model       | Emmanuel obolo    | -             | Adam        | -              | -            | 0.9813    | 0.9813  | 0.9813 | 0.9839    | 0.0467     | 0.0775    | 0.0938    | 30     |  
| Second Model      | Emmanuel Obolo    | -             | SGD         | -              | -            | 0.8628    | 0.8610  | 0.8149 | 0.9013    | 0.3850     | 0.4058    | 0.4201    | 30     |  
| Third Model       | Emmanuel Obolo    | -             | RMSProp     | -              | -            | 0.9800    | 0.9800  | 0.9800 | 0.9803    | 0.0527     | 0.0820    | 0.1210    | 30     |
| Fourth Model      | Emmanuel Obolo    | -             | Adagrad     | -              | -            | 0.3688    | 0.3398  | 0.3688 | 0.3855    | 1.7644     | 1.7453    | 1.7393    | 30     |  
| Fifth Model       | Emmanuel Obolo    | L2            | LinearSVC   | -              | -            | 0.7883    | 0.7786  | 0.7882 | 0.7859    | -          | -         | -         | 1000   |  
| Sixth Model       | Emmanuel Obolo    | -             | NuSVC       | -              | -            | 0.9680    | 0.9679  | 0.9680 | 0.9681    | -          | -         | -         | Unlimited |  


# Discussion:
At the start of my training process, I decided to split my machine learning models into two sections: one for neural networks and another for traditional ML methods.  

For the neural network section, I designed four different models under a controlled environment. What do I mean by that? Apart from differences in optimizers, loss functions, and learning rates, I kept the architecture(2 hidden layers to prevent overfitting after using 4 hidden layers at first), metrics, epochs, and batch size identical across all four models. This allowed me to compare their performance under the same conditions. After evaluation, the First model, which used Adam as the optimizer and categorical cross-entropy as the loss function, outperformed the others. It achieved the lowest train, validation, and test loss, and its confusion matrix for the multiclass labels was impressive.


For the traditional ML section, I focused on **Support Vector Machines (SVM)**. I divided this into two approaches: one using **Linear Support Vector Classification (LinearSVC)** and the other using **Nu Support Vector Classification (NuSVC)**. My findings showed that **NuSVC consistently outperformed LinearSVC across all metrics**. This is because **NuSVC, by default, uses an RBF (Radial Basis Function) kernel**, which allows it to **capture complex nonlinear relationships between features and labels**, whereas **LinearSVC is restricted to a simple linear decision boundary**. Additionally, the **gamma='scale'** setting in **NuSVC** automatically adapts the kernel coefficient based on the dataset, improving its ability to generalize. The **nu parameter** also provides more flexibility in handling misclassified points and optimizing support vectors. These factors allowed **NuSVC** to better model the underlying data structure compared to **LinearSVC**, which was too rigid in its classification.  

In conclusion, most of these performed well, except for the fourth model (adagrad) that performed poorly under a controlled environment but has the tendency to perform better if some value of the controlled environment were fine tuned, after careful evaluation of it's history  but the best-performing model was the **Neural Network First Model**, which had an edge over the others in terms of loss reduction, overall metrics, and confusion matrix results.


# Video Presentation
Here is a link to the video presentation [Youtube](https://youtu.be/KXc7wgUH644)
