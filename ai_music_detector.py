#!/usr/bin/env python3

import numpy as np
import librosa
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os
import glob
import warnings
warnings.filterwarnings('ignore')

class MathematicalAudioFeatureExtractor:
    """基于数学原理的音频特征提取器"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        
    def extract_spectral_mathematical_features(self, y):
        """频域数学特征提取（增强版）"""
        features = {}
        
        # 计算STFT
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # 1. 谱质心数学统计特征
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)[0]
        features['centroid_mean'] = np.mean(spectral_centroids)
        features['centroid_std'] = np.std(spectral_centroids)
        features['centroid_regularity'] = 1.0 / (1.0 + np.std(np.diff(spectral_centroids)))
        
        # 2. 谱滚降数学特征（85%能量点）
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sr, roll_percent=0.85)[0]
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        features['rolloff_std'] = np.std(spectral_rolloff)
        
        # 3. 谐波-噪声比数学模型
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_energy = np.mean(librosa.feature.rms(y=harmonic))
        percussive_energy = np.mean(librosa.feature.rms(y=percussive))
        features['harmonic_noise_ratio'] = harmonic_energy / (percussive_energy + 1e-10)
        
        # 4. 频谱峰度（AI音乐频谱分布更尖锐）
        freq_weights = np.mean(magnitude, axis=1)
        features['spectral_kurtosis'] = stats.kurtosis(freq_weights)
        
        return features
    
    def extract_temporal_mathematical_features(self, y):
        """时域数学特征提取（增强版）"""
        features = {}
        
        # 1. RMS动态范围压缩量（AI音乐压缩更强）
        rms = librosa.feature.rms(y=y)[0]
        features['dynamic_range_compression'] = np.std(rms) / (np.mean(rms) + 1e-10)
        
        # 2. 零交叉率数学特征（AI人声常出现伪高频）
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_variance'] = np.var(zcr)
        
        # 3. 自相关峰间距方差（节拍抖动jitter）
        autocorr = np.correlate(y, y, mode='full')[len(y)-1:]
        autocorr_norm = autocorr / (autocorr[0] + 1e-10)
        peak_indices = []
        for i in range(1, min(1000, len(autocorr_norm)-1)):
            if (autocorr_norm[i] > autocorr_norm[i-1] and 
                autocorr_norm[i] > autocorr_norm[i+1] and 
                autocorr_norm[i] > 0.1):
                peak_indices.append(i)
        
        if len(peak_indices) > 2:
            peak_intervals = np.diff(peak_indices)
            features['beat_jitter'] = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-10)
        else:
            features['beat_jitter'] = 0.5
            
        # 4. 能量熵（AI音乐能量分布更规律）
        frame_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        frame_energy_norm = frame_energy / (np.sum(frame_energy) + 1e-10)
        features['energy_entropy'] = -np.sum(frame_energy_norm * np.log(frame_energy_norm + 1e-10))
        
        return features
    
    def extract_musical_mathematical_features(self, y):
        """音乐学数学特征提取（增强版）"""
        features = {}
        
        # 1. 节拍间隔标准差（AI音乐节拍更规律）
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
        if len(beats) > 3:
            beat_intervals = np.diff(beats) / self.sr
            features['beat_interval_std'] = np.std(beat_intervals)
        else:
            features['beat_interval_std'] = 0.5
        
        # 2. 调性稳定度（Key consistency）
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        # 计算主调的稳定性
        dominant_key = np.argmax(chroma_mean)
        key_consistency = chroma_mean[dominant_key] / (np.sum(chroma_mean) + 1e-10)
        features['key_consistency'] = key_consistency
        
        # 3. MFCC均值相关性（13×12对相关系数）
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_correlations = []
        for i in range(len(mfcc)-1):
            try:
                corr, _ = stats.pearsonr(mfcc[i], mfcc[i+1])
                if not np.isnan(corr):
                    mfcc_correlations.append(abs(corr))
            except:
                continue
        
        if mfcc_correlations:
            features['mfcc_correlation_mean'] = np.mean(mfcc_correlations)
            features['mfcc_correlation_variance'] = np.var(mfcc_correlations)
        else:
            features['mfcc_correlation_mean'] = 0.5
            features['mfcc_correlation_variance'] = 0.1
        
        # 4. Chromagram最大值/平均值比值
        chroma_max = np.max(chroma_mean)
        chroma_avg = np.mean(chroma_mean)
        features['chroma_peak_ratio'] = chroma_max / (chroma_avg + 1e-10)
        
        return features
    
    def extract_all_features(self, audio_file):
        """提取所有数学特征"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            # 音频预处理
            y = librosa.util.normalize(y)
            
            features = {}
            
            # 提取各类数学特征
            spectral_features = self.extract_spectral_mathematical_features(y)
            temporal_features = self.extract_temporal_mathematical_features(y)
            musical_features = self.extract_musical_mathematical_features(y)
            
            features.update(spectral_features)
            features.update(temporal_features)  
            features.update(musical_features)
            
            return features
            
        except Exception as e:
            print(f"特征提取错误 {audio_file}: {e}")
            # 返回默认特征值
            return {f'feature_{i}': 0.0 for i in range(15)}


class MathematicalAIMusicDetector:
    """基于数学特征的AI音乐检测器"""
    
    def __init__(self):
        self.feature_extractor = MathematicalAudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def _create_interpretable_ensemble(self):
        """创建增强的可解释集成模型"""
        # 随机森林：特征重要性分析
        rf = RandomForestClassifier(
            n_estimators=100,  # 增加树数量
            max_depth=12, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # 逻辑回归：线性可解释性
        lr = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
            C=0.1  # 增加正则化
        )
        
        # 支持向量机：非线性边界
        svm = SVC(
            probability=True, 
            random_state=42,
            class_weight='balanced',
            kernel='rbf',
            C=1.0,
            gamma='scale'
        )
        
        try:
            # 尝试使用XGBoost增强性能
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            # 四模型集成
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('lr', lr), ('svm', svm), ('xgb', xgb)],
                voting='soft'
            )
            print("使用增强四模型集成（RF+LR+SVM+XGB）")
            
        except ImportError:
            # 回退到三模型集成
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
                voting='soft'
            )
            print("使用三模型集成（RF+LR+SVM）")
        
        return ensemble
    
    def train_with_real_data(self, training_data_path="附件六：训练数据"):
        """使用真实训练数据训练模型"""
        print("基于真实音频数据训练模型...")
        
        from sklearn.model_selection import train_test_split
        
        # AI生成音乐文件夹（标签=1）
        ai_folders = ['alice', 'china_vocaloid', 'game', 'gugugaga', 'ikun', 'manbo', 'yiwu']
        # 非AI生成音乐文件夹（标签=0）
        human_folders = ['hanser', 'tianyi_daddy', 'xiangsi', 'xiexiemiao~']
        
        # 收集所有音频文件路径和标签
        audio_files = []
        labels = []
        
        # AI生成音乐
        for folder in ai_folders:
            folder_path = os.path.join(training_data_path, folder)
            if os.path.exists(folder_path):
                # 支持多种音频格式
                for ext in ['*.mp3', '*.aac', '*.wav', '*.flac']:
                    files = glob.glob(os.path.join(folder_path, ext))
                    audio_files.extend(files)
                    labels.extend([1] * len(files))  # AI=1
                print(f"AI类别 {folder}: 找到 {len(glob.glob(os.path.join(folder_path, '*.*')))} 个文件")
        
        # 人类创作音乐  
        for folder in human_folders:
            folder_path = os.path.join(training_data_path, folder)
            if os.path.exists(folder_path):
                for ext in ['*.mp3', '*.aac', '*.wav', '*.flac']:
                    files = glob.glob(os.path.join(folder_path, ext))
                    audio_files.extend(files)
                    labels.extend([0] * len(files))  # 人类=0
                print(f"人类类别 {folder}: 找到 {len(glob.glob(os.path.join(folder_path, '*.*')))} 个文件")
        
        print(f"总计: {len(audio_files)} 个音频文件")
        print(f"AI生成: {sum(labels)} 个, 人类创作: {len(labels) - sum(labels)} 个")
        
        if len(audio_files) == 0:
            raise FileNotFoundError("错误: 未找到训练音频文件，请检查附件六：训练数据目录")
        
        # 限制样本数量以避免过长训练时间
        max_samples_per_class = 50
        ai_files = [(f, l) for f, l in zip(audio_files, labels) if l == 1][:max_samples_per_class]
        human_files = [(f, l) for f, l in zip(audio_files, labels) if l == 0][:max_samples_per_class]
        
        selected_files = ai_files + human_files
        audio_files = [f for f, l in selected_files]
        labels = [l for f, l in selected_files]
        
        print(f"选择训练样本: AI={len(ai_files)}, 人类={len(human_files)}")
        
        # 提取特征（支持并行处理）
        print("提取音频特征...")
        try:
            from joblib import Parallel, delayed
            
            def extract_single_feature(audio_file, label):
                try:
                    features = self.feature_extractor.extract_all_features(audio_file)
                    feature_vector = []
                    for name in self.get_feature_names():
                        feature_vector.append(features.get(name, 0.0))
                    return feature_vector, label
                except Exception as e:
                    print(f"特征提取失败 {os.path.basename(audio_file)}: {e}")
                    return None, None
            
            # 并行处理（使用2个进程避免过度占用资源）
            results = Parallel(n_jobs=2, verbose=1)(
                delayed(extract_single_feature)(af, lbl) 
                for af, lbl in zip(audio_files, labels)
            )
            
            # 过滤有效结果
            features_list = []
            valid_labels = []
            for feat, lbl in results:
                if feat is not None:
                    features_list.append(feat)
                    valid_labels.append(lbl)
                    
        except ImportError:
            # 回退到串行处理
            print("并行处理不可用，使用串行处理...")
            features_list = []
            valid_labels = []
            
            for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
                print(f"处理 {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                try:
                    features = self.feature_extractor.extract_all_features(audio_file)
                    feature_vector = []
                    for name in self.get_feature_names():
                        feature_vector.append(features.get(name, 0.0))
                    
                    features_list.append(feature_vector)
                    valid_labels.append(label)
                    
                except Exception as e:
                    print(f"特征提取失败 {audio_file}: {e}")
                    continue
        
        if len(features_list) == 0:
            raise RuntimeError("错误: 无法提取任何有效特征，请检查音频文件格式")
        
        # 转换为numpy数组
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        print(f"有效特征矩阵: {X.shape}")
        
        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        self.model = self._create_interpretable_ensemble()
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"训练集准确率: {train_score:.3f}")
        print(f"测试集准确率: {test_score:.3f}")
        
        # 自适应阈值优化
        self.optimal_threshold = self._find_optimal_threshold(X_test_scaled, y_test)
        print(f"最优阈值: {self.optimal_threshold:.3f}")
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"交叉验证准确率: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
        
        return self
    
    def _find_optimal_threshold(self, X_test, y_test):
        """在验证集上寻找最优阈值（最大化F1-score）"""
        from sklearn.metrics import f1_score
        
        # 获取预测概率
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        # 扫描阈值范围 0.3-0.7
        for threshold in np.arange(0.3, 0.71, 0.02):
            predictions = (probabilities >= threshold).astype(int)
            f1 = f1_score(y_test, predictions, average='weighted')
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"最佳F1-score: {best_f1:.3f} (阈值={best_threshold:.3f})")
        return best_threshold
    
    def get_feature_names(self):
        """获取增强特征名称列表（18维）"""
        return [
            # 频域特征 (7维)
            'centroid_mean', 'centroid_std', 'centroid_regularity',
            'rolloff_mean', 'rolloff_std', 
            'harmonic_noise_ratio', 'spectral_kurtosis',
            
            # 时域特征 (5维)  
            'dynamic_range_compression', 'zcr_mean', 'zcr_variance',
            'beat_jitter', 'energy_entropy',
            
            # 音乐学特征 (6维)
            'beat_interval_std', 'key_consistency', 
            'mfcc_correlation_mean', 'mfcc_correlation_variance',
            'chroma_peak_ratio'
        ]
    
    def train(self, training_data_path="附件六：训练数据"):
        """训练模型的主要接口"""
        return self.train_with_real_data(training_data_path)
    
    def predict_single(self, audio_file):
        """预测单个音频文件（使用自适应阈值）"""
        features = self.feature_extractor.extract_all_features(audio_file)
        
        # 确保特征顺序一致
        feature_vector = []
        for name in self.get_feature_names():
            feature_vector.append(features.get(name, 0.0))
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        probability = self.model.predict_proba(X_scaled)[0]
        
        # 使用自适应阈值进行预测
        threshold = getattr(self, 'optimal_threshold', 0.5)
        prediction = int(probability[1] >= threshold)
        
        return {
            'prediction': prediction,
            'probability_ai': probability[1],
            'probability_human': probability[0],
            'threshold_used': threshold,
            'features': features
        }
    
    def get_feature_importance_explanation(self):
        """获取特征重要性解释"""
        if hasattr(self.model.named_estimators_['rf'], 'feature_importances_'):
            importance = self.model.named_estimators_['rf'].feature_importances_
            feature_importance = list(zip(self.get_feature_names(), importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\n=== 数学特征重要性分析 ===")
            for name, imp in feature_importance[:10]:
                print(f"{name}: {imp:.3f}")
                
            return feature_importance
        return []


def test_single_file():
    """测试单个文件的检测"""
    detector = MathematicalAIMusicDetector()
    
    try:
        print("训练模型...")
        detector.train()
        
        # 测试附件四
        test_file = "附件四：测试音乐.mp3"
        if os.path.exists(test_file):
            print(f"\n测试文件: {test_file}")
            result = detector.predict_single(test_file)
            print(f"预测结果: {'AI生成' if result['prediction'] == 1 else '人类创作'}")
            print(f"AI概率: {result['probability_ai']:.3f}")
            print(f"使用阈值: {result['threshold_used']:.3f}")
        else:
            print("未找到测试文件")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_file()
    else:
        # 完整训练和测试
        print("=== AI音乐检测器 - 基于真实训练数据 ===")
        
        detector = MathematicalAIMusicDetector()
        
        try:
            print("使用附件六训练数据训练模型...")
            detector.train()
            
            print("\n✅ AI音乐检测器训练完成!")
            print("基于数学特征的可解释性检测模型")
            
            # 显示特征重要性
            detector.get_feature_importance_explanation()
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            print("请确保附件六：训练数据目录存在且包含音频文件")