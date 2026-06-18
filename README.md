# SWEM - Student Wellbeing Evaluation & Monitoring

An AI-powered Python application designed to monitor and analyze a student's cognitive and physiological state during study sessions. By tracking posture, blinking and heartrate, the system determines whether a user is **Focused**, **Stressed**, or **Sleepy**.

## 👥 Team Members
* Alessia Radicula
* Beatrice Norace
* Alvaro Burga Larco
* Mo Nabagheh

---
## 📌 Project Overview

This project utilizes real-time data collection and machine learning to classify a user's mental state while studying. It leverages some monitoring techniques to gather inputs from three distinct biometrics:

* **Posture:** Tracked via the computer's webcam.
* **Blinking Rate:** Monitored via the computer's webcam.
* **Heart Rate (HR):** Measured using a Photoplethysmogram (PPG) sensor.

### Key Execution Details:
* **Time Windows:** System parameters are captured and calculated in **1-minute intervals**.
* **Analytics & Statistics:** After processing the session, the application generates and displays detailed statistics tracking the user's state and results over time.
---

## 🚀 Features

The application supports two distinct workflows for data ingestion and analysis:

1. **Real-Time Monitoring:** Capture camera feeds and PPG sensor parameters live during your study session.
2. **Offline File Upload:** * Upload a **video file** to analyze posture and blinking.
   * Upload a **CSV file** containing the recorded Heart Rate (HR) parameters.

* **Dual Model Architecture:** Toggle between Supervised Classification and Unsupervised Anomaly Detection within the UI.
* **Built-in Data Augmentation:** Tools to expand and enrich the training dataset.
---

## 📊 Activity Report & Methodology

### 1. Data Collection & Preprocessing
* **Dataset Creation:** The training data was collected directly from the four group members, capturing videos of each category and its related PPG heart rate measurements.
* **Data Augmentation:** To improve model robustness, generalize predictions, and prevent overfitting, an optional data augmentation pipeline was implemented for the training dataset.

### 2. Machine Learning Approaches
The application gives users the flexibility to choose between two different machine learning paradigms and various 
ML models depending on their analysis goals. 
Additionally, an option to automatically choose the best model according to the training results is available:

| Model Type                | Classification Capability                        | Available Models / Algorithms                                                                                         | Description                                                                                                          |
|:--------------------------|:-------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| **Supervised Learning**   | • 3-Class Classification <br>• Anomaly Detection | • Logistic Regression<br>• Support Vector Machine (SVM)<br>• Random Forest<br>• Gradient Boosting                     | Fully differentiates and labels all three specific states: **Focused**, **Stressed**, and **Sleepy**.                |
| **Unsupervised Learning** | • Anomaly Detection                              | • Isolation Forest<br>• One-Class SVM<br>• Local Outlier Factor (LOF)<br>• K-Means<br>• Gaussian Mixture Models (GMM) | Treats **Focused** as the baseline "Normal" state, while flags **Stressed** and **Sleepy** behaviors as "Anomalous". |

---

## 📁 Repository Map

```text
.
├── backup/              # Support scripts and old versions of the code
├── data/                # Raw and processed datasets
├── docs/                # Proposal, research, reports, and design document
├── results/             # Statistics and information of the used machine learning models 
├── scripts/             # Code that is used to run the application
├── Test Video/          # Folder used to test the functionality of uploading a video
└── Video/               # All videos that were used to train the model
```
---
## 📦 Getting Started

Follow these steps to set up the project locally on your machine.
### 1. Clone the Repository

Clone the repository using your preferred Git client or directly via the command line:

``` bash
git clone https://github.com/momobuilds/Students-Ergonomics-Monitoring.git
```

Then move to the root of the project
``` bash
cd Students-Ergonomics-Monitoring
```

### 2. Set Up the Virtual Environment

We strongly recommend creating a Python virtual environment to manage dependencies cleanly.

#### Create the environment:

``` bash
python -m venv venv
```

#### Activate the environment:

- **macOS / Linux**

``` bash
source venv/bin/activate
```

- **Windows**

``` bash
venv\Scripts\activate
```

### 3. Install Dependencies
With the virtual environment activated, install all required Python packages listed in `requirements.txt`:

``` bash
pip install -r requirements.txt
```

### 4. Run the Application
The project features an interactive web interface built with Streamlit. 
To launch the application, run the following command from the root directory of the project:
``` bash
streamlit run ./scripts/GUI/app.py
```
Once executed, your default web browser should automatically open the application. 
If it doesn't open automatically, look at your terminal output and copy-paste the local URL provided (usually `http://localhost:8501`).
