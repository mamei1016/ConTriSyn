# ConTriSyn: Triple-Drug Synergy Prediction & Supplementary Experiments

This repository contains the complete source code for the **ConTriSyn** model, including data preprocessing, feature engineering, model training, and the scripts used to generate the supplementary figures and statistical analyses for the reviewer responses.

## 📁 1. Project Structure

├── data/                               # 存放所有数据文件
│   ├── ThreeDrugCombs.csv              # 内部训练/验证数据集
│   ├── NCATST_NEW.csv                  # 外部验证数据集 (NCATS)
│   └── cell_three.csv                  # 细胞系基因表达谱数据
├── model/                              # 模型训练
├── feature /                           # 数据处理、特征工程

2. Environment & Dependencies

pip install numpy pandas scikit-learn xgboost scipy statsmodels matplotlib seaborn rdkit

3. Data Preparation

确保 data/ 文件夹下包含以下文件：
1. ThreeDrugCombs.csv: 内部数据，必须包含列 `Cell Name`, `Drug1`, `Drug2`, `Drug3`, `Drug1 Concentration`, `Drug2 Concentration`, `Drug3 Concentration`。
2. NCATST_NEW.csv`: 外部数据，格式同上，但不需要Cell Name 列
