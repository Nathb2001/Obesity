# Bayesian Obesity Diagnosis WebApp

This WebApp uses a Bayesian Network to diagnose obesity after asking the user a few questions about their lifestyle. Unlike traditional machine learning approaches, Bayesian Networks use conditional probabilities between a network of variables, using causal relationships to make predictions about unseen outcomes. This allows for more intuitive reasoning about how these variables influence one another. Additionally, Bayesian Networks enable the use of interventions and counterfactuals to assist in decision-making.

Through interventions, one can simulate the impact of changing certain variables, while counterfactual techniques allow for the exploration of alternative scenarios, helping to understand what might happen under different conditions. These methods show outcomes in parallel situations which traditional machine learning cannot do.  

This repo includes the code for the Web App but also the Jupyter notebooks to showcase how the Bayesian Network was created.

## Features of this repo

- **Exploratory Data Analysis** - Find out how to analyse data identifying between Obesity and user demographics!
- **Data Cleaning & Preprocessing** - Learn how to prepare a dataset so it can be used for Algorithms to generate Bayesian Networks and make causal inferences!
- **Structural Learning Algorithms** - Learn about the algorithms on Bayesys that are able to create Bayesian Networks and understand how to evaluate them using metrics.
- **Causal Inference** - Discover how to make predictions with the Bayesian Network using the Python Library pyAgrum.
- **Counterfactual Modelling** - See how to do modelling to see the alternate outcomes of patients if they had made different lifestyle choices. 
- **Model Deployment and Monitoring** - Explore the code used for implementing the Network online for future predictions and how to connect user activity to a relational database and monitor the model.

Try the WebApp out now at **https://obesity-diagnosis.streamlit.app/**

## Technical Specifications

- **Backend**: Python, Streamlit
- **Libraries**: pyAgrum, pandas, Streamlit-GSheets
- **Data Source**: The app uses a Google Sheets database for storing and retrieving classification results.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the developers of the libraries and tools used in this project.

---
