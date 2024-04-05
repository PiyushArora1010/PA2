# Programming Assignment II

This README file provides detailed explanations of the provided code and instructions on running each file to replicate the reported results.

## Environment Setup
Ensure that you have the required packages installed by running the **environment.yml** file. 

## Prerequisites
### Data Preparation
For each question, specific datasets are necessary:

For Question 01, download and store the VoxCeleb1-H dataset and the Punjabi language data from the Kathbath dataset.

For Question 02, the required generated data is stored under data/LibriMixData/test.

### Model Checkpoints
For each question, download the required model checkpoints:

For Question 01, download the checkpoints for the three pre-trained models ('hubert_large', 'wavlm_base_plus', 'wavlm_large').

For Question 02, follow the provided instructions to obtain the SepFormer model.

## Usage Instructions

For the execution of each subtask, run main.py with the following hyperparameters:
model: Specify the model to be used (e.g., 'hubert_large', 'wavlm_base_plus', etc.).
checkpoint: Provide the path to the model checkpoint.
que: Indicate the specific subtask to be executed (e.g., '1.b', '2.b', etc.).

Example command:

`python main.py --model hubert_large --checkpoint checkpoints/hubert_large --que 1.b`
