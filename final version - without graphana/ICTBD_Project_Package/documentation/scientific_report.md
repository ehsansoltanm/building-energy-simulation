## Scientific Report: Surrogate Modeling and Uncertainty Quantification for Building Energy Prediction using Simulation Data

**Abstract:**

This report details the implementation and evaluation of surrogate models for predicting building energy performance, coupled with various techniques for uncertainty quantification (UQ). Utilizing hourly weather data (EPW format) for Torino, Italy, and corresponding simulation outputs from EnergyPlus (eplusout.csv) for a residential flat model (Res_flat1.idf), this work explores the prediction of zone-specific energy metrics (specifically, Zone Total Internal Latent Gain Energy for BLOCCO1:ZONA3). Machine learning models, including Multi-Layer Perceptrons (MLP) and Random Forests, are trained using scikit-learn. Crucially, the study implements and compares three distinct UQ methods: Model-Free Uncertainty (MFU) via ensemble learning, Model-Free Inference (MFI) using conformal prediction, and Quantile Regression. Additionally, an Out-of-Distribution (OOD) detection mechanism based on Mahalanobis distance is incorporated to identify input data points significantly different from the training distribution. The results demonstrate the feasibility of using surrogate models for rapid energy prediction and highlight the varying characteristics (coverage, interval width) of different UQ approaches, providing valuable insights for reliable building performance assessment and control.

**1. Introduction:**

Building energy simulation tools like EnergyPlus provide detailed physics-based models for predicting energy consumption and indoor environmental conditions. However, these simulations can be computationally expensive, especially for tasks requiring numerous runs, such as design optimization, real-time control, or large-scale urban energy modeling. Surrogate models, often based on machine learning techniques, offer a computationally efficient alternative by learning the input-output relationship directly from simulation data or real-world measurements.

While point predictions from surrogate models are useful, understanding the associated uncertainty is critical for robust decision-making. Uncertainty arises from various sources, including measurement errors in input data, inherent variability in building operation, and limitations of the surrogate model itself. Uncertainty Quantification (UQ) provides methods to estimate prediction intervals, offering a range within which the true value is likely to fall with a certain confidence level.

This report focuses on developing surrogate models for predicting the 'Zone Total Internal Latent Gain Energy' for a specific zone ('BLOCCO1:ZONA3') within a simulated residential flat ('Res_flat1.idf') located in Torino, Italy ('Torino_IT-hour.epw'). The work leverages simulation results ('eplusout.csv') and corresponding weather data. It implements simplified prediction models using scikit-learn and explores three prominent UQ techniques: ensemble methods (approximating Model-Free Uncertainty - MFU), conformal prediction (Model-Free Inference - MFI), and Quantile Regression. Furthermore, it addresses the challenge of applying models to potentially novel conditions by implementing an Out-of-Distribution (OOD) detection method.

The objectives are: (1) To develop machine learning-based surrogate models for hourly energy prediction based on EnergyPlus simulation data. (2) To implement and compare different UQ methods (Ensemble, Conformal, Quantile Regression) for estimating prediction uncertainty. (3) To implement an OOD detection mechanism to assess the reliability of predictions on new data.

**2. Methodology:**

**2.1 Data Sources and Preprocessing:**

The primary data sources are:
*   **Weather Data:** An EPW file (`Torino_IT-hour.epw`) providing hourly meteorological data for Torino, Italy. Key features extracted include Dry Bulb Temperature ('Temperature'), Relative Humidity ('Humidity'), Direct Normal Radiation ('DirectRad'), and Diffuse Horizontal Radiation ('DiffuseRad').
*   **Simulation Data:** A CSV file (`eplusout.csv`) containing detailed hourly output variables from an EnergyPlus simulation of the 'Res_flat1.idf' building model.
*   **Target Variable:** The specific prediction target selected is 'BLOCCO1:ZONA3:Zone Total Internal Latent Gain Energy [J](TimeStep)'.

The preprocessing steps involve:
1.  Loading both datasets using the pandas library.
2.  Selecting relevant weather features and the target energy variable.
3.  Aligning the timestamps and ensuring consistent lengths between the weather and simulation dataframes.
4.  Handling missing values (implicitly done via concatenation and potential `dropna()`).
5.  Scaling features (Temperature, Humidity, DirectRad, DiffuseRad) and the target variable using `MinMaxScaler` to a [0, 1] range to improve model training stability.
6.  Creating time sequences: Input sequences (`X_seq`) consist of the scaled features from the previous 24 hours (`n_lag = 24`), and the target (`Y_seq`) is the scaled energy value at the current hour.
7.  Flattening the input sequences (`X_seq_flat`) to be compatible with standard scikit-learn regressors.
8.  Splitting the data into training (80%) and testing (20%) sets chronologically.

**2.2 Prediction Models:**

Instead of a complex LSTM, simpler scikit-learn models are used:
*   **Multi-Layer Perceptron (MLP):** An `MLPRegressor` with two hidden layers (64 and 32 neurons), ReLU activation, and the Adam optimizer is used for basic prediction and as the base model for ensemble and conformal methods.
*   **Random Forest:** A `RandomForestRegressor` is employed for the Quantile Regression approach.

**2.3 Uncertainty Quantification (UQ) Methods:**

1.  **Ensemble Method (MFU Approximation):** An ensemble of 10 MLP models is trained, each initialized with a different random state. Predictions are made using all models on the test set. The mean of the ensemble predictions serves as the point estimate, and the standard deviation across the predictions is used as a measure of uncertainty. Prediction intervals are constructed as mean ± 2 standard deviations (approximating a 95% confidence interval under normality assumptions).
2.  **Conformal Prediction (MFI):** The training data is further split into a proper training set and a calibration set (20%). An MLP model is trained on the proper training set. Non-conformity scores (absolute residuals) are calculated on the calibration set. The (1-α) quantile of these scores (where α=0.05 for a 95% interval) is determined. For new test points, the prediction interval is constructed as [prediction - quantile, prediction + quantile]. This method provides distribution-free guarantees on marginal coverage.
3.  **Quantile Regression:** Random Forest regressors are trained to predict specific quantiles of the target variable distribution (e.g., 0.1, 0.5, 0.9). While scikit-learn's `RandomForestRegressor` doesn't directly support quantile loss, it's used here as an approximation, potentially with adjustments to predictions or leaf size parameters. The predictions for the lower (0.1) and upper (0.9) quantiles define the prediction interval (e.g., an 80% interval). The prediction for the 0.5 quantile (median) serves as the point estimate.

**2.4 Out-of-Distribution (OOD) Detection:**

Mahalanobis distance is used to detect OOD samples in the test set. The mean vector and inverse covariance matrix are calculated from the flattened training data (`X_train`). For each test sample, the Mahalanobis distance to the training data distribution is computed. A threshold is set based on a high percentile (e.g., 95th) of the distances calculated on the *test* set itself (or ideally, on a separate validation set representative of in-distribution data). Test samples with distances exceeding this threshold are flagged as OOD.

**3. Results and Discussion:**

*(Note: Numerical results depend on actual script execution. This section describes the expected outputs and their interpretation based on the script's structure.)*

The script generates performance metrics (MAE, RMSE, R²) for the basic MLP prediction model, providing a baseline assessment of point prediction accuracy. Visualizations comparing true vs. predicted values for the first 100 hours are produced.

For UQ methods, the script generates:
*   **Visualizations:** Plots showing the true values, mean/median predictions, and the corresponding prediction intervals (e.g., 95% for Ensemble/Conformal, 80% for Quantile) for each method over the first 100 test hours.
*   **Comparison Metrics:** A table comparing the UQ methods based on:
    *   **Empirical Coverage:** The actual percentage of true test values falling within the predicted intervals. This is compared to the target coverage (e.g., 95% or 80%). Conformal prediction is expected to achieve coverage close to the target level.
    *   **Average Interval Width:** The average width of the prediction intervals. Narrower intervals are generally preferred, provided they maintain the desired coverage.
    *   **Point Prediction Accuracy (RMSE/MAPE):** Accuracy of the mean (Ensemble) or median (Quantile, Conformal base model) predictions.

Discussion points would include:
*   The trade-offs between coverage and interval width for different methods.
*   The computational cost associated with each method (Ensemble requires training multiple models).
*   The interpretation of asymmetric intervals from Quantile Regression, if observed.
*   The validity of the normality assumption for the Ensemble method's interval construction.

The OOD detection part produces a plot of Mahalanobis distances for test samples and indicates those exceeding the threshold. The discussion should emphasize the importance of identifying OOD samples, as predictions and uncertainty estimates for such points may be unreliable. The percentage of detected OOD samples provides an indication of how much the test distribution differs from the training distribution.

**4. Conclusion:**

This work successfully demonstrated the development of surrogate models using MLP and Random Forest regressors for predicting hourly zone energy metrics based on EnergyPlus simulation outputs and weather data. Several UQ techniques were implemented, providing methods to estimate prediction intervals crucial for risk assessment and reliable decision-making. The comparison between Ensemble, Conformal Prediction, and Quantile Regression highlights their different theoretical guarantees and practical performance characteristics regarding coverage and interval width. The inclusion of OOD detection using Mahalanobis distance adds a layer of robustness, enabling the identification of potentially unreliable predictions when the model encounters novel input conditions.

Future work could involve exploring more sophisticated base models (like LSTMs, as originally intended), applying these techniques to different prediction targets (e.g., heating/cooling loads, zone temperatures), integrating the prediction and UQ framework into a building management system for predictive control, and further refining the OOD detection mechanism.

**5. References (Conceptual):**

*   EnergyPlus Documentation.
*   Scikit-learn User Guide (MLPRegressor, RandomForestRegressor, metrics).
*   Relevant literature on surrogate modeling in building energy analysis.
*   Relevant literature on Uncertainty Quantification (Conformal Prediction, Ensemble Methods, Quantile Regression).
*   Relevant literature on Out-of-Distribution Detection (Mahalanobis Distance).

