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
import warnings
warnings.filterwarnings('ignore')

class MathematicalAudioFeatureExtractor:
    """基于数学原理的音频特征提取器"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        
    def extract_spectral_mathematical_features(self, y):
        """频域数学特征提取"""
        features = {}
        
        # 计算STFT
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # 1. 谱质心分布规律性
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)[0]
        features['centroid_regularity'] = 1.0 / (1.0 + np.std(np.diff(spectral_centroids)))
        features['centroid_trend_linearity'] = np.abs(stats.pearsonr(range(len(spectral_centroids)), spectral_centroids)[0])
        
        # 2. 谐波-噪声比数学模型
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_energy = np.mean(librosa.feature.rms(y=harmonic))
        percussive_energy = np.mean(librosa.feature.rms(y=percussive))
        features['harmonic_noise_ratio'] = harmonic_energy / (percussive_energy + 1e-10)
        features['harmonic_dominance'] = harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)
        
        # 3. 频谱包络平滑度
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sr)[0]
        features['spectral_envelope_smoothness'] = 1.0 / (1.0 + np.std(np.diff(spectral_rolloff)))
        
        # 4. 频率分布数学特征
        freqs = librosa.fft_frequencies(sr=self.sr)
        freq_weights = np.mean(magnitude, axis=1)
        features['freq_distribution_skewness'] = stats.skew(freq_weights)
        features['freq_distribution_kurtosis'] = stats.kurtosis(freq_weights)
        
        return features
    
    def extract_temporal_mathematical_features(self, y):
        """时域数学特征提取"""
        features = {}
        
        # 1. 动态范围数学分析
        rms = librosa.feature.rms(y=y)[0]
        features['dynamic_range_compression'] = np.std(rms) / (np.mean(rms) + 1e-10)
        features['amplitude_regularity'] = 1.0 / (1.0 + np.var(np.diff(rms)))
        
        # 2. 微时间结构规律性
        autocorr = np.correlate(y, y, mode='full')[len(y)-1:]
        autocorr_norm = autocorr / autocorr[0]
        peak_indices = []
        for i in range(1, min(1000, len(autocorr_norm)-1)):
            if autocorr_norm[i] > autocorr_norm[i-1] and autocorr_norm[i] > autocorr_norm[i+1]:
                if autocorr_norm[i] > 0.1:  # 阈值
                    peak_indices.append(i)
        
        if len(peak_indices) > 1:
            peak_intervals = np.diff(peak_indices)
            features['micro_periodicity'] = 1.0 / (1.0 + np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-10))
        else:
            features['micro_periodicity'] = 0.0
            
        # 3. 能量分布数学模型
        frame_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        features['energy_distribution_entropy'] = -np.sum(frame_energy * np.log(frame_energy + 1e-10))
        features['energy_variance_regularity'] = 1.0 / (1.0 + np.var(frame_energy))
        
        return features
    
    def extract_musical_mathematical_features(self, y):
        """音乐学数学特征提取"""
        features = {}
        
        # 1. 节拍强度数学一致性
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
        if len(beats) > 3:
            beat_intervals = np.diff(beats) / self.sr
            features['tempo_regularity'] = 1.0 / (1.0 + np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
            features['tempo_mathematical_consistency'] = np.abs(stats.pearsonr(range(len(beat_intervals)), beat_intervals)[0])
        else:
            features['tempo_regularity'] = 0.5
            features['tempo_mathematical_consistency'] = 0.5
        
        # 2. 色度特征数学分析
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        features['tonal_stability'] = 1.0 - np.std(chroma_mean) / (np.mean(chroma_mean) + 1e-10)
        features['chroma_mathematical_regularity'] = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-10)
        
        # 3. MFCC数学特征
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        features['mfcc_coefficient_stability'] = 1.0 / (1.0 + np.mean(np.std(mfcc, axis=1)))
        features['mfcc_mathematical_coherence'] = np.mean([np.abs(stats.pearsonr(mfcc[i], mfcc[i+1])[0]) 
                                                         for i in range(len(mfcc)-1)])
        
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
        """创建可解释的集成模型"""
        # 决策树：高可解释性
        rf = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        
        # 逻辑回归：数学关系明确
        lr = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
        # 支持向量机：数学边界清晰
        svm = SVC(
            probability=True, 
            random_state=42,
            class_weight='balanced'
        )
        
        # 集成模型
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
            voting='soft'
        )
        
        return ensemble
    
    def train_with_synthetic_data(self):
        """使用合成数据训练模型"""
        print("基于数学特征模式生成训练数据...")
        
        # 生成合成训练数据
        np.random.seed(42)
        
        # AI音乐特征模式（基于数学分析的假设）
        ai_features = []
        human_features = []
        
        for _ in range(100):  # AI样本
            features = {
                'centroid_regularity': np.random.normal(0.8, 0.1),  # AI音乐谱质心更规律
                'centroid_trend_linearity': np.random.normal(0.9, 0.05),  # AI音乐线性趋势更强
                'harmonic_noise_ratio': np.random.normal(3.0, 0.5),  # AI音乐谐波比例更高
                'harmonic_dominance': np.random.normal(0.7, 0.1),
                'spectral_envelope_smoothness': np.random.normal(0.85, 0.1),  # 频谱包络更平滑
                'freq_distribution_skewness': np.random.normal(0.2, 0.3),
                'freq_distribution_kurtosis': np.random.normal(2.5, 0.5),
                'dynamic_range_compression': np.random.normal(0.3, 0.1),  # AI音乐动态范围更压缩
                'amplitude_regularity': np.random.normal(0.8, 0.1),
                'micro_periodicity': np.random.normal(0.9, 0.05),  # 微周期性更强
                'energy_distribution_entropy': np.random.normal(-2.0, 0.5),
                'energy_variance_regularity': np.random.normal(0.85, 0.1),
                'tempo_regularity': np.random.normal(0.95, 0.03),  # 节拍更规律
                'tempo_mathematical_consistency': np.random.normal(0.9, 0.05),
                'tonal_stability': np.random.normal(0.8, 0.1)  # 调性更稳定
            }
            ai_features.append(list(features.values()))
            
        for _ in range(100):  # 人类音乐样本
            features = {
                'centroid_regularity': np.random.normal(0.4, 0.15),  # 人类音乐变化更大
                'centroid_trend_linearity': np.random.normal(0.3, 0.2),
                'harmonic_noise_ratio': np.random.normal(1.5, 0.8),
                'harmonic_dominance': np.random.normal(0.5, 0.15),
                'spectral_envelope_smoothness': np.random.normal(0.5, 0.2),
                'freq_distribution_skewness': np.random.normal(0.8, 0.5),
                'freq_distribution_kurtosis': np.random.normal(4.0, 1.0),
                'dynamic_range_compression': np.random.normal(0.7, 0.2),  # 动态范围更大
                'amplitude_regularity': np.random.normal(0.4, 0.2),
                'micro_periodicity': np.random.normal(0.3, 0.2),
                'energy_distribution_entropy': np.random.normal(-1.0, 0.8),
                'energy_variance_regularity': np.random.normal(0.4, 0.2),
                'tempo_regularity': np.random.normal(0.6, 0.2),  # 节拍变化更大
                'tempo_mathematical_consistency': np.random.normal(0.4, 0.2),
                'tonal_stability': np.random.normal(0.5, 0.2)
            }
            human_features.append(list(features.values()))
        
        # 构建训练数据
        X = np.array(ai_features + human_features)
        y = np.array([1] * len(ai_features) + [0] * len(human_features))
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model = self._create_interpretable_ensemble()
        self.model.fit(X_scaled, y)
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"模型交叉验证准确率: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
        
        # 保存特征名称
        self.feature_names = list({
            'centroid_regularity', 'centroid_trend_linearity', 'harmonic_noise_ratio',
            'harmonic_dominance', 'spectral_envelope_smoothness', 'freq_distribution_skewness',
            'freq_distribution_kurtosis', 'dynamic_range_compression', 'amplitude_regularity',
            'micro_periodicity', 'energy_distribution_entropy', 'energy_variance_regularity',
            'tempo_regularity', 'tempo_mathematical_consistency', 'tonal_stability'
        })
        
        return self
    
    def predict_single(self, audio_file):
        """预测单个音频文件"""
        features = self.feature_extractor.extract_all_features(audio_file)
        
        # 确保特征顺序一致
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(features.get(name, 0.0))
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability_ai': probability[1],
            'probability_human': probability[0],
            'features': features
        }
    
    def get_feature_importance_explanation(self):
        """获取特征重要性解释"""
        if hasattr(self.model.named_estimators_['rf'], 'feature_importances_'):
            importance = self.model.named_estimators_['rf'].feature_importances_
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\n=== 数学特征重要性分析 ===")
            for name, imp in feature_importance[:10]:
                print(f"{name}: {imp:.3f}")
                
            return feature_importance
        return []


if __name__ == "__main__":
    # 示例使用
    detector = MathematicalAIMusicDetector()
    detector.train_with_synthetic_data()
    
    print("AI音乐检测器已准备就绪!")
    print("基于数学特征的可解释性检测模型")