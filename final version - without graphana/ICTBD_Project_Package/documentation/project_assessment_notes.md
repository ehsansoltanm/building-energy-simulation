# Project Assessment Notes

## 1. File Review Summary:

*   **Course Program (`1-Course programme.pdf` -> `course_programme.txt`):** Outlines course structure, topics (modeling, simulation, comfort, controls, ICT, BMS, IoT, optimization, surrogate models), and exercise structure (Part 1: Design Optimization, Part 2: Building Operation). Exam requires theory test, exercise report/VM, and presentation.
*   **Group Assignment (`Capture.JPG`):** Confirms user is in Group 4, project topic is "Res_flat1", climate is Torino.
*   **Main Notebook (`khodemoon-v3.1.ipynb`):** Implements Part 1 (Design Optimization) using `besos`, `eppy`, `scikit-learn`, and `pymoo`. 
    *   Defines parameters: Roof/Wall insulation thickness, Lighting W/m², Natural Vent ACH, Mech Vent OA/Person.
    *   Defines objectives: Electricity, Heating, Cooling.
    *   Uses LHS for sampling (though `NUM_SAMPLES=1` in code, `simulation_outputs.csv` has 184 results).
    *   Runs EnergyPlus simulations (implied, results loaded from `simulation_outputs.csv`).
    *   Trains Surrogate Models (GPR, MLP).
    *   Performs NSGA-II optimization using the GPR surrogate.
    *   Plots results (surrogate performance, Pareto front).
*   **Previous Script (`lstm_prediction_script.py`):** Focuses on time-series prediction using simulation output (`eplusout.csv`) and weather data (`Torino_IT-hour.epw`), including uncertainty quantification (Ensemble, Conformal, Quantile Regression) and OOD detection. Uses `scikit-learn` (MLP, RandomForest). Relates to "Surrogate Model Deployment" and "Prediction Algorithm".
*   **Environment (`environment.yaml`, `Dockerfile`):** Defines a detailed conda environment and Docker setup for reproducibility, including multiple EnergyPlus versions (9.0.1, 9.4.0, 9.6.0).
*   **Model/Simulation Files (`Res_flat1.idf`, `flat1-.dsb`, `Torino_IT-hour.epw`, `eplusout.csv`, `simulation_outputs.csv`, other `.csv`, `.eso`, `.err`, etc.):** Confirm the use of DesignBuilder, EnergyPlus v9.6 (based on IDF version), the specific building model, and weather file. Show evidence of simulation runs.

## 2. Analysis vs. Course Requirements:

*   **Alignment with Course Topics:** The project strongly aligns with topics like building modeling (IDF), dynamic simulation (EnergyPlus), surrogate modeling (GPR, MLP), and parametric optimization (LHS, NSGA-II).
*   **Alignment with Exercise Structure:**
    *   **Part 1 (Design Optimization):** The work in `khodemoon-v3.1.ipynb` directly addresses this part, covering most steps shown in the flowchart (IDF -> Sampling/DoE -> Simulation -> Surrogate Model -> Optimization).
    *   **Part 2 (Building Operation):** The `lstm_prediction_script.py` touches upon prediction algorithms and surrogate deployment, which are relevant to Part 2. However, elements like stepped simulation, EnergyPlus Gym, BMS integration (MQTT, REST, openHAB, InfluxDB, Grafana), control strategy development, and energy signature analysis appear **missing** based on the provided files.
*   **Specific Implementations:**
    *   **Optimization Parameters:** Insulation, lighting, and ventilation parameters are included. Window *type* optimization seems commented out. Shading control optimization is missing.
    *   **Surrogate Models:** GPR and MLP are used in the optimization notebook. The separate script explores MLP and RandomForest for time-series prediction and UQ.
    *   **Optimization Algorithm:** NSGA-II is used, which is standard for multi-objective optimization.
    *   **Prediction/UQ:** The separate script explores advanced UQ (Ensemble, Conformal, Quantile) and OOD detection, which might go beyond basic requirements but fits the course's ICT focus.




## 3. Evaluation: Completed vs. Pending Tasks:

**Completed Tasks (Based on Provided Files):**

*   **Part 1: Design Optimization (Largely Addressed in `khodemoon-v3.1.ipynb`):**
    *   Building Modeling: IDF file (`Res_flat1.idf`) created (likely via DesignBuilder `flat1-.dsb`).
    *   Parameter Definition: Input parameters for optimization identified (Roof/Wall Insulation Thickness, Lighting Power Density, Natural Ventilation ACH, Mechanical Ventilation OA Flow/Person).
    *   Objective Definition: Output objectives for optimization defined (Electricity, District Heating, District Cooling).
    *   Sampling: Latin Hypercube Sampling (LHS) implemented for generating parameter sets.
    *   Simulation Execution: EnergyPlus simulations were run (evidenced by output files like `simulation_outputs.csv`, `eplusout.csv`, `.eso`, `.mtr`, etc.).
    *   Surrogate Modeling: Gaussian Process Regression (GPR) and Multi-Layer Perceptron (MLP) models trained on simulation results.
    *   Optimization: NSGA-II multi-objective optimization performed using the GPR surrogate model.
    *   Results Analysis: Visualization of surrogate model performance and Pareto front.
*   **Part 2: Building Operation (Partially Addressed in `lstm_prediction_script.py`):**
    *   Prediction Algorithm Development: Time-series prediction models (MLP, RandomForest) developed using simulation output and weather data.
    *   Uncertainty Quantification: Advanced UQ methods (Ensemble, Conformal, Quantile Regression) explored.
    *   Out-of-Distribution Detection: Implemented using Mahalanobis distance.
*   **General Project Setup:**
    *   Environment Definition: Reproducible environment specified using Conda (`environment.yaml`) and Docker (`Dockerfile`).

**Pending Tasks (Based on Course Program & Flowchart):**

*   **Part 1: Design Optimization:**
    *   Parameter Refinement: Window type optimization appears commented out; Shading control optimization seems missing.
    *   Final Optimized IDF: While optimization was run, the final selected optimized IDF configuration isn't explicitly identified.
*   **Part 2: Building Operation (Significant Gaps):**
    *   Stepped Simulation: Implementation of EnergyPlus Gym or similar for co-simulation with external control logic is missing.
    *   BMS Integration: No evidence of setting up or interacting with BMS platforms like openHAB, InfluxDB, or Grafana.
    *   Communication Protocols: No implementation of MQTT or RESTful API communication for device connection/data exchange.
    *   Control Strategy: Development and testing of specific control strategies based on predictions or real-time data are missing.
    *   Energy Signature: Analysis of building energy signature not found.
    *   Real Data Integration: The project uses a standard EPW file; integration with real-time sensor data is not shown.
*   **Exam Deliverables:**
    *   Final Report: Consolidation of all exercise results into a final report (required one week before the exam).
    *   Virtual Machine (VM): Preparation of the VM containing the project setup and results.
    *   Presentation: Preparation of slides for the oral presentation.
    *   Theoretical Test: Preparation for the individual written test.




## 4. Methodology Deviations & Extra Work:

**Methodology Deviations:**

*   **Prediction Approach (`lstm_prediction_script.py`):** While developing a prediction algorithm is part of the course (likely Part 2), the approach taken in this script deviates from the likely intended integration. The script performs offline training and analysis on simulation output (`eplusout.csv`) rather than being integrated into an operational context (like a Building Management System or a co-simulation loop using EnergyPlus Gym). The course flowchart suggests prediction is part of the BMS loop, likely using data streams from the simulation/sensors and feeding into control strategies.
*   **Use of `scikit-learn` vs. LSTM:** The filename `lstm_prediction_script.py` suggests an intention to use LSTM, but the actual implementation uses `scikit-learn` models (MLP, RandomForest). This isn't necessarily wrong, but it's a deviation from the implied tool in the filename.

**Potential Extra Work (Beyond Core Requirements):**

*   **Advanced Uncertainty Quantification (UQ) & OOD Detection:** The detailed exploration of multiple UQ methods (Ensemble, Conformal Prediction, Quantile Regression) and Out-of-Distribution detection in `lstm_prediction_script.py` appears to be extra work. While relevant to ICT and potentially valuable, it might exceed the basic requirements for the prediction component of the exercise.
*   **Detailed Environment Setup (`Dockerfile`, `environment.yaml`):** Creating a comprehensive Dockerfile and Conda environment file shows good practice for reproducibility but might represent extra effort beyond simply listing required packages. Installing multiple EnergyPlus versions (9.0.1, 9.4.0, 9.6.0) within the Dockerfile is definitely extra complexity unless specifically required for comparison purposes (which isn't evident).
*   **Separate Prediction Script:** Developing the `lstm_prediction_script.py` as a standalone analysis, separate from the main optimization workflow in `khodemoon-v3.1.ipynb`, could be considered extra work, especially given its focus on advanced UQ/OOD rather than direct integration into the Part 2 operational simulation.

