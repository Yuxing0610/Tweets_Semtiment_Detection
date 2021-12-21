1. Environment

Please run pip install -r requirements.txt in command line to prepare all the environments ready.
Go to ./src/KrovetzStemmer and run pip install . to prepare preprocessing environment. If error occurs, run backup_pre-processing.py install of pre-processing.py

2. Data

All the data and model parameters are packed in Google Drive, please download data from xxxxx and use this 'data' folder cover origin data folder.

3. Run

run main.py with params you need.

4.Best result

Now the best result is already saved in ./output/submission.csv. In this way, the accuracy is 91.5%.
To recurrent the best result, run python main.py directly, submission.csv will be created in ./output/submission.csv. In this method, the accuracy is 91.4%. This is because we update preprocessing method and change emoji representation slightly while the model stay the same. Due to the hardware resource limitation, we are unable to train a new model. If trained again, we believe the accuracy rate will still go to 91.5%.
