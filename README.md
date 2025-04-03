## NYC-Vision-Zero-MLOPS
<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->

Value Proposition: Vision Zero hopes to reduce the amount of general vehicular related incidents. With our proposed system we can create a dashboard using the various types of information present within our datasets to create a model, or models, to classify the risk of possible incidents on any given day. Classifications would be as follows, subject to change; Dangerous, Somewhat Dangerous, Average, Somewhat Safe, and Safe. The NYPD and NYCDOT would also be able to use this information to better predict potential incidents based on existing daily conditions. Leading to faster response times, safer roads.

<!-- What are the business metrics to measure?-->


<!-- Whats the non ML status Quo?-->


### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |                 |                                    |
| Jaden Antoine                   |  Model Training and Training Platforms             |         [Link](https://github.com/JadeAnt/NYC-Vision-Zero-MLOPS/commits/main/?author=JadeAnt)                           |
| Jason Widjaja                   |  Continuous X               |      [Link](https://github.com/JadeAnt/NYC-Vision-Zero-MLOPS/commits/main/?author=JasonW35214)                                 |
| Neha Nainan                     |  Data Pipeline               |          [Link](https://github.com/JadeAnt/NYC-Vision-Zero-MLOPS/commits/main/?author=nehaann23)                             |
| Parth Naik    |  Model Serving and Monitoring               |           [Link](https://github.com/JadeAnt/NYC-Vision-Zero-MLOPS/commits/main/?author=Parthnaik123)                            |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->
![System diagram here](./System_diagram.PNG)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use | Data Source Type |
|--------------|--------------------|-------------------|-------------------|
| Motor Vehicle Collisions - Crashes   |                    |                   |       Dataset/API            |
| Citywide Traffic Statistics   |                    |                   |       Dataset            |
| Vision Zero View Map |                    |                   |         Data Visualization          |
| NYPD TrafficStat Map |                    |                   |         Data Visualization          |
| NYCOpenData |                    |                   |        Dataset Catalog           |
| OpenWeatherMap |                    |                   |        Dataset/API         |
| BERT |                    |                   |         Foundation Model          |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 4 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |      We will be attempting the difficulty point for Model Training and Training Platforms by training a BERT model, which is an LLM that will require a gpu for reduced training and inference times.          |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| `Rasberry-Pi 5` | 1 for entire project duration, 2 hour block twice a week |     As our deployments are on-device focused, an edge device is needed to allow us to test the efficiency of our model and pipeline. Both our canary and production environments will require an edge device for testing artifical users.        |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->
(1) Strategy
- Begin with exploratory data analysis to analyze the data, clean it, and find the features we hypothesize will be the best to work with starting out, before moving on to testing feature selection based methods
- Train, retrain, and test various different machine learning models in order to find which performs the best for our multi-classification based problem. 
  - Some models we would test for example might be (SVM, Random Forest, Decision Tree, Regression (feature selection) etc…)
  - If we do use multiple models, it will most likely be Regression for feature Selection + a classification model. In case of change in relevant features
  - Compare the results of each of the models and choose one, or more, models for our final output
- Track the experiments we perform with each of these models using the MLFlow platform
  - Ran in a separate container with volume, with the dataset split into train, valid, test, and production sets
  - Saving the details of each run such using checkpointing to record version, hyperparameters, loss/metrics, alert for errors, and monitor the health of our hardware and software
- Schedule training jobs using a Ray cluster for our continuous pipeline
  - Using a head node to schedule/manage jobs, data, and serve a dashboard with 2 worker nodes. 
  - Will Use MinIO object store for persistent job storage and save model checkpoints here as well in case of error or interruption
  - Create a separate Jupyter notebook container to submit jobs to cluster through
  - Prometheus and Grafana will be used for metric collection and dashboard visualization


(2) Relevant Diagram Part
- Within the system diagram the bulk of the model training and training platforms section of the project functions within the Development environment.
- The data sourcing from our datasets, exploratory data analysis, model training, tracking with MLFlow, and scheduling using Ray cluster
- Data will be stored within our persistent storage within Chameleon
- Using Docker containers to manage the different sections of the training pipeline
- A retrainning trigger (schedule-based, etc..) will exist here as well to trigger model retraining
- Model versions will be stored in model registry
- Model will move to the Staging environment once all development steps have been completed

(3) Justification
- For our project, our end user would hypothetically use this on an edge device. As such it is important for our final model, or models, to be able to be deployed on such a device at all
- As we will be testing multiple models, optimizations, etc.. It is important that we keep track of all of these model and code versions. As such the need for MLFlow to track our changes and store our data is imperative to our project
- Also, the usage of a Ray cluster to manage our jobs is crucial for our continuous pipeline to work properly

(4) Lecture Material Reference
- Referring back to Units 4 and 5 in our lectures, we will be utilizing training with backpropagation in order to allow our models to learn the appropriate internal representations to better classify our data
- Furthermore, as it was previously stated in lecture, paying explicit attention to our model’s size, velocity, and budget is a MUST. As they need to be able to perform on an edge device with a relatively small model size, with decent velocity, and small budget

(5) Difficulty Points
- Training strategies for large models
  - We plan on testing a BERT model for training in order to see the effectiveness of an LLM on a classification task such as this. 
  - To facilitate the training of an LLM we plan on experimenting with the following training strategies and report measurements to evaluate their effectiveness. These strategies are:
    - Quantization
    - LORA
    - Pruning
- These strategies are what we chose as each focuses on reducing model size and increasing model inference speed, two factors that are extremely important in our planned deployment to an edge device
  - Scheduling hyperparameter tuning jobs
  - Ray Tune will be used to perform hyperparameter optimization
  - This will allow for easier hyperparameter tuning to occur with intelligent scheduling and allowing for faster testing of various configurations. Saving resources on our cluster


#### Model serving and monitoring platforms

(1) Strategy:
We will serve the trained model using a FastAPI-based REST API. This API will be containerized using Docker and deployed on a Chameleon VM with a floating IP. The endpoint will accept input data and return a risk classification label along with a confidence score. This service will be accessible by the front-end dashboard for real-time prediction.

Since the model is expected to run on an edge device (e.g., Rasberry Pi 5), it must be both small and fast. We will explore model-level optimizations including dynamic and static quantization, pruning, and possibly reduced-precision conversion. We may also explore graph optimizations using TorchScript or ONNX.

Once deployed, we will define and run an automated offline evaluation suite. This includes metrics like accuracy, precision, recall, F1-score, and ROC-AUC, along with slice-based analysis (e.g., by borough or time). Known failure modes (e.g., holidays, traffic spikes) will be tested, and results will be logged to MLflow.

Following staging deployment, we will conduct load testing using synthetic input data to measure system latency, throughput, and failure rate. After staging passes, we will run a canary evaluation using simulated users sending live-like inputs to the API. These predictions will be logged and compared with labeled data (as available) for quality checks.

To close the loop, predictions from production use will be logged, a subset labeled, and added to the training dataset for weekly re-training. This process will be integrated into our CI/CD pipeline.

(2) Diagram Description:
- The user or dashboard sends input data to a FastAPI endpoint.
- The API loads the trained model, runs inference, and returns a prediction.
- Inputs and predictions are logged in real-time.
- Evaluation and monitoring scripts analyze logs for drift or degradation.
- MLFlow stores evaluation results and alerts trigger retraining if needed.
- A dashboard (e.g., Streamlit or Grafana) visualizes model health and alerts.

(3) Justification:
Serving the model via a REST API enables seamless integration with the frontend and supports scalable deployment. Edge-device constraints require us to optimize for size and latency. Continuous evaluation ensures the model remains accurate, fair, and robust over time, even as real-world conditions change.

(4) Lecture Material Reference:
This section directly aligns with Units 6 and 7. Unit 6 emphasized serving models efficiently and optimizing for performance, especially on constrained devices. Unit 7 introduced strategies for offline evaluation, load testing, canary deployments, and feedback loops to ensure model quality and responsiveness in production.

(5) Difficulty Points (Optional):
- Multi-platform serving: We plan to evaluate model performance on CPU, GPU, and edge devices.
- Monitor for data drift: We will compare real-time input distributions to training data using statistical methods (e.g., KS-test).
- Monitor for model degradation: We will track prediction quality over time and implement a dashboard to alert on degradation. Retraining will be automatically triggered when necessary using new production-labeled data.

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

(1) Strategy
- Persistent Storage:  
  Provision persistent block storage on Chameleon to store all critical artifacts, including historical datasets and streaming data.
- Offline Data Pipeline:  
  - Data Sources:  
    - OpenWeatherMap (weather data)  
    - NYC Motor Vehicle Collisions Vehicles  
    - Citywide Traffic Statistics  
  - Process:  
    - Extraction: Schedule regular ETL jobs (using Apache Airflow) to pull updated datasets.
    - Transformation: Clean data by handling missing values, normalizing fields, and performing feature extraction.
    - Loading: Store the transformed data in persistent storage.
- Online Data Pipeline:  
  - Data Source:  
    - Edge Device (Raspberry Pi) providing real-time GPS location data.
  - Process:  
    - Data Simulation & Ingestion: Use a lightweight script (via REST API or MQTT) to simulate real-time GPS streaming.
    - Real-Time Processing: Ingest data into a streaming pipeline, applying minimal cleaning (e.g., timestamp validation, coordinate formatting).
    - Storage & Usage:  
      - Store the online data temporarily for immediate use by the Inference API.
      - Log the data in persistent storage for later model re-training and evaluation.

(2) Relevant Diagram Part

(3) Justification
- Offline Pipeline:  
  Automates ingestion and ensures high-quality historical data for robust model training.
- Online Pipeline:  
  Simulates production conditions with real-time data ingestion, enabling immediate inference and continuous learning.
- Persistent Storage:  
  Decouples compute from data, ensuring reliability and ease of maintenance during scaling or system updates.

(4) Lecture Material Reference
- The implementation of data pipeline will reference the Lab 8 manual on Data pipeline. The contents listed for this section is subject to change when the reference is released.


(5) Difficulty Points 
- Data Dashboard:  
  Implement an interactive dashboard to visualize:
    - Real-time data ingestion rates.
    - Data quality metrics.
    - ETL job performance.

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->

(1) Strategy
- Infrastructure-as-code:
  - All infrastructure provisioning and configuration will be defined and version-controlled in this GitHub repository.
  - Will provide both declarative Terraform configurations and imperative python-chi scripts to provision infrastructure.
  - Will choose an automation tool (Ansible, ArgoCD, Argo Workflows, Helm, or python-chi) for installation and configuration of necessary dependencies on provisioned resources. Will use same tool or another for post-deployment configuration management.

- Cloud-native:
  - All infrastructure modifications will be made through Git commits and applied using Terraform or python-chi. No manual changes will be made to running resources.
  - The application will be broken down into independent, containerized services that communicate via APIs. These services will be deployed and scaled separately using Kubernetes.
  - All services, including the machine learning model, will be containerized using Docker and managed through Kubernetes. 
  - The Dockerfile for containerization and Kubernetes deployment file is stored and version-controlled in GitHub. 

- CI/CD and Continuous Training:
  - Source code and ML training pipelines will be stored in Git.
  - Each commit triggers automated testing using GitHub Actions.
  - Once all tests pass, Docker will update the appropriate container.

- Staged deployment
  - Services are promoted from staging -> canary -> production based on automated evaluations. This ensures a consistent and controlled release process.
  - Staging provides a safe environment for pre-production testing and the canary enables production testing with minimal risk.
  - These environments will be separated using Kubernetes.
  - Environment transitions will be defined in GitHub Actions.

(2) Relevant Diagram Part
- The source control section, at the top of the sytem diagram, represents the downstream flow of the Github Actions. The ML system lifecycle and the tranisitons between deployment environments (development, staging, canary, amd production) are clearly defined in the configuration files.

(3) Justification
- Using Git for version control of infrastructure enables reproducibility, auditability, and collaboration.
- Using scripts/configuration files allows for automation which reduces human error and ensures consistency.
- Direct execution on VMs will be avoided, ensuring portability and scalability.

(4) Lecture Material Reference
- The implementation of continuous X will reference the Lab 3 manual on DevOps for ML. The contents listed for this section is subject to change when the reference is released.
