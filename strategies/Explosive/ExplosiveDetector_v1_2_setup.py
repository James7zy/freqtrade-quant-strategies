#!/usr/bin/env python3
"""
ExplosiveDetector v1.2 安装和设置脚本
用于初始化ML模型、创建必要目录和配置环境
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExplosiveDetectorSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"

    def check_python_version(self):
        """检查Python版本"""
        if sys.version_info < (3, 8):
            logger.error("需要Python 3.8或更高版本")
            return False
        logger.info(f"Python版本检查通过: {sys.version}")
        return True

    def install_dependencies(self):
        """安装必要的依赖包"""
        required_packages = [
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "ccxt>=4.0.0",
            "vaderSentiment>=3.3.2",
            "textblob>=0.17.1",
            "requests>=2.28.0"
        ]

        optional_packages = [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "feature-engine>=1.6.0",
            "optuna>=3.0.0"
        ]

        logger.info("开始安装必需依赖...")
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ 已安装: {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ 安装失败: {package} - {e}")
                return False

        logger.info("开始安装可选依赖...")
        for package in optional_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ 已安装 (可选): {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ 可选包安装失败: {package} - {e}")

        return True

    def create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.models_dir,
            self.logs_dir,
            self.data_dir,
            self.data_dir / "cache",
            self.data_dir / "training",
            self.data_dir / "exports"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ 目录已创建: {directory}")

        return True

    def create_sample_training_data(self):
        """创建示例训练数据"""
        import numpy as np
        import pandas as pd

        # 生成模拟训练数据
        np.random.seed(42)
        n_samples = 1000

        features = {
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd_hist': np.random.uniform(-0.1, 0.1, n_samples),
            'bb_position': np.random.uniform(0, 1, n_samples),
            'vol_ratio': np.random.uniform(0.5, 5.0, n_samples),
            'roc_1h': np.random.uniform(-0.1, 0.1, n_samples),
            'roc_4h': np.random.uniform(-0.2, 0.2, n_samples),
            'atr_percent': np.random.uniform(1, 10, n_samples),
            'mfi': np.random.uniform(20, 80, n_samples),
            'willr': np.random.uniform(-100, 0, n_samples),
            'stoch_k': np.random.uniform(0, 100, n_samples),
            'news_sentiment': np.random.uniform(-1, 1, n_samples),
            'news_count': np.random.randint(0, 20, n_samples),
            'social_heat': np.random.uniform(0, 1, n_samples),
            'social_sentiment': np.random.uniform(-1, 1, n_samples),
            'whale_transactions': np.random.randint(0, 50, n_samples),
            'exchange_flow_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'oi_change': np.random.uniform(-50, 50, n_samples),
            'funding_rate': np.random.uniform(-0.01, 0.01, n_samples),
            'long_short_ratio': np.random.uniform(0.3, 3.0, n_samples),
            'btc_correlation': np.random.uniform(-1, 1, n_samples)
        }

        df = pd.DataFrame(features)

        # 生成标签 (1 = 暴涨, 0 = 不暴涨)
        # 复杂的逻辑来模拟真实情况
        explosion_probability = (
            (df['vol_ratio'] > 2.0).astype(int) * 0.3 +
            (df['roc_1h'] > 0.05).astype(int) * 0.25 +
            (df['news_sentiment'] > 0.2).astype(int) * 0.2 +
            (df['social_heat'] > 0.5).astype(int) * 0.15 +
            (df['whale_transactions'] > 10).astype(int) * 0.1
        )

        df['target'] = (explosion_probability > 0.6).astype(int)

        # 保存训练数据
        training_file = self.data_dir / "training" / "sample_training_data.csv"
        df.to_csv(training_file, index=False)
        logger.info(f"✅ 示例训练数据已创建: {training_file}")

        return True

    def initialize_ml_models(self):
        """初始化机器学习模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            import pandas as pd
            import joblib

            # 加载训练数据
            training_file = self.data_dir / "training" / "sample_training_data.csv"
            if not training_file.exists():
                logger.error("训练数据不存在，请先运行create_sample_training_data")
                return False

            df = pd.read_csv(training_file)
            X = df.drop(['target'], axis=1)
            y = df['target']

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 创建集成模型
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            lr = LogisticRegression(random_state=42, max_iter=1000)

            voting_clf = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='probability'
            )

            # 训练模型
            logger.info("开始训练ML模型...")
            voting_clf.fit(X_train_scaled, y_train)

            # 评估模型
            from sklearn.metrics import accuracy_score, classification_report
            y_pred = voting_clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"模型训练完成，准确率: {accuracy:.4f}")
            logger.info("分类报告:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # 保存模型
            model_path = self.models_dir / "explosive_detector_v1_2.pkl"
            scaler_path = self.models_dir / "explosive_detector_v1_2_scaler.pkl"

            joblib.dump(voting_clf, model_path)
            joblib.dump(scaler, scaler_path)

            logger.info(f"✅ 模型已保存: {model_path}")
            logger.info(f"✅ 标准化器已保存: {scaler_path}")

            # 保存特征列表
            feature_columns = X.columns.tolist()
            features_path = self.models_dir / "feature_columns.json"
            with open(features_path, 'w') as f:
                json.dump(feature_columns, f)

            logger.info(f"✅ 特征列表已保存: {features_path}")

            return True

        except ImportError as e:
            logger.error(f"ML库导入失败: {e}")
            return False
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            return False

    def create_config_templates(self):
        """创建配置文件模板"""
        # API配置模板
        api_config = {
            "news_apis": {
                "cryptopanic": {
                    "api_key": "your_cryptopanic_api_key_here",
                    "enabled": False
                },
                "newsapi": {
                    "api_key": "your_newsapi_key_here",
                    "enabled": False
                }
            },
            "social_apis": {
                "twitter": {
                    "bearer_token": "your_twitter_bearer_token_here",
                    "enabled": False
                },
                "reddit": {
                    "client_id": "your_reddit_client_id_here",
                    "client_secret": "your_reddit_secret_here",
                    "enabled": False
                }
            },
            "onchain_apis": {
                "glassnode": {
                    "api_key": "your_glassnode_api_key_here",
                    "enabled": False
                },
                "etherscan": {
                    "api_key": "your_etherscan_api_key_here",
                    "enabled": False
                }
            }
        }

        api_config_path = self.base_dir / "api_config_template.json"
        with open(api_config_path, 'w') as f:
            json.dump(api_config, f, indent=2)

        logger.info(f"✅ API配置模板已创建: {api_config_path}")

        return True

    def run_setup(self):
        """运行完整的设置流程"""
        logger.info("🚀 开始ExplosiveDetector v1.2设置...")

        steps = [
            ("检查Python版本", self.check_python_version),
            ("安装依赖包", self.install_dependencies),
            ("创建目录结构", self.create_directories),
            ("创建示例训练数据", self.create_sample_training_data),
            ("初始化ML模型", self.initialize_ml_models),
            ("创建配置模板", self.create_config_templates)
        ]

        for step_name, step_func in steps:
            logger.info(f"执行步骤: {step_name}")
            if not step_func():
                logger.error(f"❌ 步骤失败: {step_name}")
                return False
            logger.info(f"✅ 步骤完成: {step_name}")

        logger.info("🎉 ExplosiveDetector v1.2设置完成！")
        logger.info("\n下一步操作:")
        logger.info("1. 编辑api_config_template.json配置你的API密钥")
        logger.info("2. 运行回测以验证策略: freqtrade backtesting --strategy ExplosiveDetector_v1_2")
        logger.info("3. 启动实时监控: freqtrade trade --strategy ExplosiveDetector_v1_2 --dry-run")

        return True

if __name__ == "__main__":
    setup = ExplosiveDetectorSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)