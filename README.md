# HRI-RNN
## Presentation
This project contains the `Python/PyTorch` implementation of **HRI-RNN**, a recurrent GRU-based architecture for online detection of user's engagement decrease in human-robot interactions.

**HRI-RNN** differs from traditional engagement analysis models in that it explicitly uses the robot data to model the *context* of the interaction. Two GRUs—one modeling the user state and one modeling the context—are interconnected to capture the dependencies between the user and the robot (context).

## Classification Task
Given an HRI sequence of (multimodal) feature vectors, where features characterize either the user or the robot, we want to label the sequence as 1 if it presents signs of user's engagement decrease (SED) or 0 otherwise.
