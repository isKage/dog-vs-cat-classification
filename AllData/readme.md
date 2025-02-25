将从 [kaggle 的数据库](https://www.kaggle.com/competitions/dog-vs-cat-classification/data) 
下载的 competitions/dog-vs-cat-classification/ 等文件放入 AllData 文件夹

文件结构为

```bash
AllData
    └── competitions
        └── dog-vs-cat-classification
            ├── dog-vs-cat-classification.zip  # 这是从 kaggle 下载后的压缩包【无需】
            ├── sample_submission.csv  # 这是 kaggle 提交时的格式【无需】
            ├── test
            │   └── test
            │       └── 000000.jpg
            │       ├── ...
            └── train
                └── train
                    ├── cats
                    │   └── cat.0.jpg
                    │   ├── ...
                    └── dogs
                        └── dog.0.jpg
                        ├── ...
```