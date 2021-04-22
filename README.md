# OBIS
 Outlier Based Intersection Selection Framework

This repo contains the code for the OBIS Framework, a smart way to select the right intersections for a traffic prediction model. You can use it for your own traffic dataset, or for the The Hague dataset that you can download here: https://drive.google.com/drive/folders/1Fp9zC_tN40WJYBt0yjfv0irrhdbV2gtb?usp=sharing 
(if for your own dataset, you might need to make changes to the data input - as of now, the framework is configured for the The Hague dataset, which is in the CSV format, with files for each month (and the final row is the total for that month). Furhtermore, you need to be able to select the right sensors). 

To use it for the The Hague dataset, instructions are as follows:
1. Clone this repo.
2. Download dataset from Google Drive & save it in the "data_set" folder.
3. I used Python 3.8 (other versions will probably also work). Make sure to install the requirements.
4. Run "OBIS_Framework.py" - this will take a while, should eventually create a file in the "results" folder named "OBIS_results.pickle"
5. In the results folder are two notebooks, one for plotting purposes (to generate the graphs for the paper) and one for testing the results; "LSTM.ipynb"
6. In the LSTM notebook you can run all cells, which will create LSTMs for each trajectory (4x) and each threshold (9x) (max total 36 training sessions, it will not train if the included intersections are the same). On my machine this takes around 2 hours (CPU training, Ryzen 5 3800X). The LSTM models are not finetuned as this is not the point of this research. 
7. The other cells are for inspecting the results. The Plotting notebook also creates results for the outlier situations. 
8. In all my tests, the 0.4 threshold was performing best. If you apply this framework to another dataset, I'd be very interested in your results! Please contact me at Tom7Tom@live.nl
