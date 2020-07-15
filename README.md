# HRI-RNN
## Presentation
This project contains the `Python/PyTorch` implementation of **HRI-RNN**, a recurrent GRU-based architecture for online detection of user's engagement decrease in human-robot interactions.

**HRI-RNN** differs from traditional engagement analysis models in that it explicitly uses the robot data to model the *context* of the interaction. Two GRUs—one modeling the user state and one modeling the context—are interconnected to capture the dependencies between the user and the robot (context).

## Classification Task
Given a relatively short HRI sequence of (multimodal) feature vectors—where features characterize either the user or the robot, we want to label the sequence as `1` if it presents signs of user's engagement decrease (SED) or `0` otherwise.

## Dataset
We use the **UE-HRI** dataset to train and test **HRI-RNN**. The `Python` script preprocessing raw HRI data can be found in the `Legacy` directory. For more details on feature extraction and preprocessing, we refer the user to **Ben Youssef et al.**'s work, as it is outside of the scope of this project.
