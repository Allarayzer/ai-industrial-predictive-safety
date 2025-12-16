# ai-industrial-predictive-safety
AI-based predictive safety system for high-risk industrial facilities
# AI-Based Predictive Safety System for High-Risk Industrial Facilities

## Overview
This repository presents a prototype implementation of an AI-based
predictive safety system designed for early detection of hazardous
conditions at high-risk industrial facilities.

The project is based on the scientific-practical monograph:
"Artificial Intelligence for Preventing Accidents at High-Risk Industrial Facilities".

## Motivation
Despite existing automated monitoring systems, industrial accidents
often occur due to delayed detection of anomalies, equipment degradation,
and human factors.

This project demonstrates a preventive, data-driven approach
combining machine learning and automated response workflows.

## System Architecture
The system consists of the following layers:
- Sensor data acquisition (simulation / real-time sources)
- Data preprocessing and normalization
- Anomaly detection using unsupervised ML
- Risk scoring and decision logic
- Automated alerting via n8n workflows

## Core Features
- Unsupervised anomaly detection (Isolation Forest)
- Hybrid risk scoring combining ML outputs and physical parameters
- Event-driven automation using n8n
- Modular and scalable Python architecture

## Relation to the Monograph
This repository implements the practical concepts described in:
- Chapter 7 – Practical Implementation in Python and n8n
- Chapter 8 – Demonstration Project: AI Predictive Analyzer

## Research and Engineering Contribution
- Original system architecture for predictive industrial safety
- Integration of ML-based anomaly detection with automated workflows
- Demonstration of preventive safety logic applicable to industrial systems

## Reproducibility
The project is implemented as a demonstrational prototype.
Sensor data is simulated, but the architecture is compatible with
industrial protocols such as MQTT, OPC UA, and REST APIs.

## Disclaimer
This project is intended for research and demonstration purposes
and is not intended for direct industrial deployment without validation.

## Author
Ilia Serebriakov  
Engineering Science, City University of New York  
Research interests: AI-based industrial safety, predictive analytics
