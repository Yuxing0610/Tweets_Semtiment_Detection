# Twitter Sentiment Analysis by SSY Group
## 1.Environment

a) Please run `pip install -r requirements.txt` in command line to prepare environments ready.

b-1) Run `cd ./src/KrovetzStemmer` in command line to go to *KrovetzStemmer* folder and run `pip install .` in command line to prepare preprocessing environment. 

b-2) If error occurs in b-2 (it may have conflicts with other environments), `backup_pre-processing.py` is prepared to run without *KrovetzStemmer* package

## 2.Data Setting

All the data and model parameters are packed in Google Drive, please download data from `https://drive.google.com/drive/folders/1lPgzweagIYhoEad9FFfrO2Lx39Dy0ubJ?usp=sharing` and use this `data` folder cover origin `data` folder.

## 3.File Structure
├── Readme.md                   // help
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js         // 提供给前端的配置
├── test
├── test-service.js
└── tools

## 3.Run

run `run.py` with params you need.

param for BERT/BERTweet: pos_file_path, neg_file_path, test_file_path, model, load_model, load_model_path, num_epoch, batch_size

e.g. `python run.py model=bertweet` you can get our best result.

## 4.Best result

Now the best result is already saved in `./output/submission.csv`, the accuracy rate is 91.5%.

To recurrent the best result, run `python run.py` directly, submission.csv will be created in `./output/submission.csv.` 

Note: preprocessed data is already provided. Or you can use `pre-processing.py` to create it again. If using `backup_pre-processing.py` to preprocess data, the accuracy rate will be 91.4% for slight difference in preprocessing Emoji compared with `pre-processing.py`








