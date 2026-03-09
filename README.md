# Federated Learning: IID vs Non-IID Performance Analysis
## Overview

This project presents an experimental analysis of Federated Learning under IID and Non-IID data distributions. The goal of the project is to study how data heterogeneity across distributed clients impacts the performance of federated learning models.

A convolutional neural network (CNN) was trained using centralized learning and federated learning setups, and the results were compared to analyze the effect of different data distributions.

The project was implemented using PyTorch and the Flower federated learning framework.

## Project Objective

The primary objectives of this project were:

To implement a Federated Learning training pipeline

To simulate multiple distributed clients

To analyze the effect of IID vs Non-IID data distributions

To compare centralized training with federated training

To study the impact of data heterogeneity on model performance

## Technologies Used

Python

PyTorch

Flower (Federated Learning Framework)

NumPy

Matplotlib

## Dataset

The MNIST dataset was used in this project.

MNIST is a widely used benchmark dataset for image classification consisting of handwritten digits from 0 to 9.

Dataset details:

Training samples: 60,000

Test samples: 10,000

Image size: 28 × 28 grayscale

The dataset was used to simulate a medical or distributed data environment where data is stored across multiple clients.

## System Architecture

The system follows a client–server federated learning architecture.

## Workflow

The server initializes a global model.

The global model is distributed to multiple clients.

Each client performs local training on its dataset.

Clients send model updates to the server.

The server aggregates the updates using Federated Averaging (FedAvg).

The updated global model is redistributed to clients.

## Data Distribution Setup

Two different data distribution settings were simulated:

### IID Distribution

In the IID setting, data is distributed uniformly across all clients, meaning each client has a similar data distribution.

### Non-IID Distribution

In the Non-IID setting, each client contains biased or skewed data samples, which simulates real-world environments where different institutions or devices generate different types of data.

## Experiments Conducted

The following experiments were performed:

Centralized Model Training

Federated Learning with IID Data

Federated Learning with Non-IID Data

These experiments help analyze the performance gap caused by data heterogeneity.

## Results
| Training Method              | Accuracy |
| ---------------------------- | -------- |
| Centralized Training         | 99.0%    |
| Federated Learning (IID)     | 98.77%   |
| Federated Learning (Non-IID) | 35.26%   |

## Key Observation

Federated learning performs close to centralized training under IID data conditions, but performance drops significantly under Non-IID distributions, highlighting one of the major challenges in federated learning systems.
