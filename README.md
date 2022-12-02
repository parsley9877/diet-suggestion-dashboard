# Diet Design Dashboard (D3)

D3 is an interactive web-based dashboard with primary goal of helping both health experts, and non expert individuals to tailor a healthy diet. It is a web-based product and can be used on any device (smartphone, computer)

by Parsa Assadi

## File Structure

1. `./assets`: CSS code for the dashboard style
2. `./ml_utils`: Machine learning algorithms training/evaluation code base
3. `./tabnet_models`: Saved machine learning models which are used by the dashboard to provide services for the use
4. `./presentation`: presentation pdf
5. `./plots`: Evaluation plots of machine learning algorithms
6. `app.py`: Dashboard controller, which starts the dashboard

## How to run?

1. `git clone git@github.com:parsley9877/diet-suggestion-dashboard.git`
2. `cd diet-suggestion-dashboard`
3. `pip install requirements.txt`
4. `python app.py`

It generates a link to the local port which the app is started on. By clicking on this link, you will see the dashboard in your browser.

## Packages
- dash
- pandas
- plotly
- dash_daq
- numpy
- scipy
- statsmodels
- matplotlib
- torch
- pytorch_tabnet
- sklearn