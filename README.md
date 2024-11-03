##Sequential Resource Trading Using Comparison-Based Gradient Estimation

This is the repository for the numerical and human interaction experiments performed in the paper Sequential Resource Trading Using Comparison-Based Gradient Estimation

The numerical experiments presented in the paper were run in Python 3.9.7 on a Dell XPS 13 laptop with a 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz processor and 16.0 GB of RAM.

## Required Packages
The versions of python packages used to generate our results are given in the requirements.txt file. The packages can be installed by running:

```bash
    pip install -r requirements.txt
```

## Running Numerical Tests
We use two command to run different sets of numerical tests. The first script runs tests without the Greedy Concession Algorithm (GCA):

```bash
    python run_numerical_tests.py [Arguments]
```
### Command-Line Arguments
The available command-line arguments are as follows:

- `--debug <flag>` 
  - Description: Enable debug mode for detailed logging and troubleshooting.
  - Default: Disabled (i.e., debug mode is off).
- `--integer_constraint <flag>` 
  - Description: Apply integer constraints in the trading scenarios.
  - Default: Disabled (i.e., no integer constraints applied).
  
- `--random_seed <int>`
  - Description: Set the random seed for initializing testing scenarios. This ensures reproducibility of results. 
  - Default: 10
  
- `--max_trade_value <int>`
  - Description: Specify the maximum number of items from each category that can be exchanged in a single trade.
  
  - Default: 5
  
- `--theta_closeness <float>`
  - Description: Define the semi-vertical angle threshold for the ST-CR algorithm. This parameter controls the precision of the cone refinement.
  
  - Default: 0.00001
  
- `--deviation_interval <float>`
  - Description: Set the deviation interval for random trading with momentum. This parameter affects the randomness of trading scenarios.
  
  - Default: 0.05
  
- `--max_deviation_magnitude <float>`
  - Description: Determine the maximum deviation magnitude for random trading with momentum.
  
  - Default: 5
  
- `--num_scenarios <int>`
  - Description: Specify the number of randomized testing scenarios to be generated. Note that this argument should be an integer.
  
  - Default: 500
  
- `--offer_budget <float>`
  - Description: Set the offer budget for the negotiation algorithms. This budget dictates the maximum value of offers that can be made.
  
  - Default: 1000


### Results
The testing script outputs the plots that show normalized cumulative benefit as the number of offers increases. The plots are stored in the results folder with the format: `query_benefit_curves_alignment_{mixing_constant}_items_{num_items}`.
We store the numerical results from the paper in the folder `numerical_results/paper_results`.

## Running Numerical Tests with GCA
We now provide an overview of the testing script for the testing scenarios with the Greedy Concession Algorithm. 
```bash
    python run_numerical_tests_GCA.py [Arguments]
```
### Command-Line Arguments
The available command-line arguments are as follows:

- `--debug <flag>` 
  - Description: Enable debug mode for detailed logging and troubleshooting.
  - Default: Disabled (i.e., debug mode is off).
- `--integer_constraint <flag>` 
  - Description: Apply integer constraints in the trading scenarios.
  - Default: Disabled (i.e., no integer constraints applied).
  
- `--random_seed <int>`
  - Description: Set the random seed for initializing testing scenarios. This ensures reproducibility of results. 
  - Default: 10
  
- `--max_trade_value <int>`
  - Description: Specify the maximum number of items from each category that can be exchanged in a single trade.
  
  - Default: 5
  
- `--theta_closeness <float>`
  - Description: Define the semi-vertical angle threshold for the ST-CR algorithm. This parameter controls the precision of the cone refinement.
  
  - Default: 0.00001
  
- `--deviation_interval <float>`
  - Description: Set the deviation interval for random trading with momentum. This parameter affects the randomness of trading scenarios.
  
  - Default: 0.05
  
- `--max_deviation_magnitude <float>`
  - Description: Determine the maximum deviation magnitude for random trading with momentum.
  
  - Default: 5
  
- `--num_scenarios <int>`
  - Description: Specify the number of randomized testing scenarios to be generated. Note that this argument should be an integer.
  
  - Default: 500
  
- `--offer_budget <float>`
  - Description: Set the offer budget for the negotiation algorithms. This budget dictates the maximum value of offers that can be made.
  
  - Default: 1000

- `--test_GCA <bool>`
  - Description: Determine if tests include the GCA baseline
  - Default: True

- `--shrinking_factor <float>`
  - Description: Shrinking Factor for GCA belief update

  - Default: 0.1

- `--sampled_weights <int>`
  - Description: Number of Sampled Weights for GCA

  - Default: 100

- `--softmax_temp <float>`
  - Description: Temperature for GCA Softmax

  - Default: 0.02


### Results
The testing script outputs the plots that show normalized cumulative benefit (offering, responding, and societal) as the number of offers increases. The plots are stored in the numerical_results folder with the format: `gca_query_benefit_curves_alignment_{mixing_constant}_items_{num_items}_{benefit_type}`.
We store the numerical results from the paper in the folder `numerical_results/paper_results`.


## Running GPT Tests
The human interaction test script are located in the gpt_integration folder.
```bash
    cd gpt_integration
```
To run the human interaction test, run the following command

```bash
    python gpt_integration_test.py
```
When the script starts, you should see a link to a locally hosted webpage of the form:
```bash
Starting Website on http://127.0.0.1:5000
 * Serving Flask app 'gpt_integration_test'
 * Debug mode: on  
```
Clicking on the link will take you to a webpage which will ask you for relevant information before trading begins. We note that you will need to provide your own OpenAI API key to run these tests.
###Results
The testing script stores a log of the negotiation with humans. The transcript is stored in a folder with a timestamp (Ex: `chat_folders/1723827209/log.txt`). The example logs that we show in the supplemental materials are stored in `chat_folders\paper_examples`

