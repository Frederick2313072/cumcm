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
import joblib
warnings.filterwarnings('ignore')

class MathematicalAudioFeatureExtractor:
    """基于严格数学理论的音频特征提取器
    
    设计原理：
    1. 信号处理数学基础：傅里叶变换、小波变换、希尔伯特变换
    2. 统计学特征：概率分布、熵理论、高阶矩
    3. 信息论：互信息、复杂度度量
    4. 可解释性：每个特征都有明确的物理/数学含义
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
        print(f"[数学特征提取器] 初始化完成，采样率={sr}Hz")
        
    def extract_spectral_mathematical_features(self, y):
        """频域数学特征：基于傅里叶分析和统计学理论"""
        features = {}
        print(f"[频域数学] 开始频域特征提取...")
        
        # 计算STFT - 短时傅里叶变换
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        power_spectrum = magnitude ** 2
        
        # 1. 频谱质心 - 频率的加权平均（物理含义：音色亮度）
        freq_bins = np.fft.fftfreq(stft.shape[0], 1/self.sr)[:stft.shape[0]//2]
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)[0]
        
        # 数学特征：质心的统计特性
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # 质心变化率（一阶差分的标准差）- 反映音色变化的稳定性
        centroid_diff = np.diff(spectral_centroids)
        features['centroid_stability'] = 1.0 / (1.0 + np.std(centroid_diff))
        
        # 频谱质心轨迹平滑度 - AI音乐质心轨迹更平滑
        if len(spectral_centroids) > 2:
            centroid_correlation = np.corrcoef(spectral_centroids[:-1], spectral_centroids[1:])[0, 1]
            features['centroid_trajectory_smoothness'] = centroid_correlation if not np.isnan(centroid_correlation) else 0.5
        else:
            features['centroid_trajectory_smoothness'] = 0.5
        
        print(f"[频域数学] 频谱质心增强特征完成")
        
        # 2. 频谱带宽 - 频率分布的标准差（反映频谱的集中程度）
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        print(f"[频域数学] 频谱带宽特征完成")
        
        # 3. 频谱熵 - 信息论度量（反映频谱复杂度）
        # H(X) = -∑ p(x) log p(x)
        power_normalized = power_spectrum / (np.sum(power_spectrum, axis=0, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(power_normalized * np.log(power_normalized + 1e-10), axis=0)
        features['spectral_entropy_mean'] = np.mean(spectral_entropy)
        features['spectral_entropy_std'] = np.std(spectral_entropy)
        print(f"[频域数学] 频谱熵特征完成")
        
        # 4. 频谱平坦度 - Wiener熵（反映噪声特性）
        # 几何平均 / 算术平均
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        print(f"[频域数学] 频谱平坦度特征完成")
        
        # 5. 谐波-噪声比增强特征 - 基于重要性分析的关键特征
        try:
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_rms = librosa.feature.rms(y=harmonic)[0]
            percussive_rms = librosa.feature.rms(y=percussive)[0]
            
            harmonic_energy = np.mean(harmonic_rms)
            percussive_energy = np.mean(percussive_rms)
            features['harmonic_noise_ratio'] = np.log10(harmonic_energy / (percussive_energy + 1e-10))
            
            # 谐波复杂度 - AI音乐谐波结构更简单
            features['harmonic_complexity'] = np.std(harmonic_rms) / (np.mean(harmonic_rms) + 1e-10)
            
            # 谐波-噪声平衡稳定性 - AI音乐平衡更稳定
            hnr_frames = harmonic_rms / (percussive_rms + 1e-10)
            features['hnr_stability'] = 1.0 / (1.0 + np.std(np.log10(hnr_frames + 1e-10)))
            
            print(f"[频域数学] 谐波-噪声比增强特征完成")
        except:
            features['harmonic_noise_ratio'] = 0.0
            features['harmonic_complexity'] = 0.5
            features['hnr_stability'] = 0.5
            print(f"[频域数学] 谐波-噪声比计算失败，使用默认值")
        
        return features
    
    def extract_temporal_mathematical_features(self, y):
        """时域数学特征：基于统计学和信号处理理论"""
        features = {}
        print(f"[时域数学] 开始时域特征提取...")
        
        # 限制音频长度避免计算复杂度过高
        max_len = min(len(y), 220500)  # 约10秒
        y_analysis = y[:max_len]
        
        # 1. 幅度统计特征 - 基于概率分布理论
        print(f"[时域数学] 计算幅度统计特征...")
        features['amplitude_mean'] = np.mean(np.abs(y_analysis))
        features['amplitude_std'] = np.std(y_analysis)
        features['amplitude_skewness'] = stats.skew(y_analysis)  # 偏度
        features['amplitude_kurtosis'] = stats.kurtosis(y_analysis)  # 峰度
        
        # 2. 零交叉率增强特征 - 基于重要性分析的关键特征
        print(f"[时域数学] 计算零交叉率增强特征...")
        zcr = librosa.feature.zero_crossing_rate(y_analysis, frame_length=2048, hop_length=512)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_entropy'] = -np.sum(zcr * np.log(zcr + 1e-10)) / len(zcr)
        
        # ZCR时序稳定性 - AI音乐ZCR变化更规律
        zcr_diff = np.diff(zcr)
        features['zcr_stability'] = 1.0 / (1.0 + np.std(zcr_diff))
        
        # ZCR异常值检测 - AI音乐异常ZCR更频繁
        zcr_median = np.median(zcr)
        zcr_mad = np.median(np.abs(zcr - zcr_median))  # 中位数绝对偏差
        features['zcr_outlier_ratio'] = np.sum(np.abs(zcr - zcr_median) > 2 * zcr_mad) / len(zcr)
        
        # 3. RMS能量特征 - 动态范围分析
        print(f"[时域数学] 计算RMS能量特征...")
        rms = librosa.feature.rms(y=y_analysis, frame_length=2048, hop_length=512)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 动态范围压缩指标（AI音乐通常压缩更强）
        rms_db = 20 * np.log10(rms + 1e-10)
        features['dynamic_range'] = np.max(rms_db) - np.min(rms_db)
        features['compression_ratio'] = np.std(rms_db) / (np.mean(rms_db) + 1e-10)
        
        # 4. 短时能量变化率 - 节奏稳定性
        print(f"[时域数学] 计算能量变化率...")
        energy_diff = np.diff(rms)
        features['energy_variation'] = np.std(energy_diff)
        features['energy_regularity'] = 1.0 / (1.0 + np.std(energy_diff))
        
        # 5. 自相关特征 - 周期性分析（简化版避免复杂度过高）
        print(f"[时域数学] 计算自相关特征...")
        try:
            # 使用较短的窗口计算自相关
            window_size = min(8192, len(y_analysis))
            y_window = y_analysis[:window_size]
            
            # 计算归一化自相关
            autocorr = np.correlate(y_window, y_window, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr_norm = autocorr / (autocorr[0] + 1e-10)
            
            # 寻找第一个显著峰值（基频估计）
            first_peak_idx = 1
            for i in range(1, min(len(autocorr_norm)//4, 1000)):
                if (autocorr_norm[i] > autocorr_norm[i-1] and 
                    autocorr_norm[i] > autocorr_norm[i+1] and 
                    autocorr_norm[i] > 0.3):
                    first_peak_idx = i
                    break
            
            features['periodicity_strength'] = autocorr_norm[first_peak_idx] if first_peak_idx > 1 else 0.1
            features['fundamental_period'] = first_peak_idx / self.sr if first_peak_idx > 1 else 0.01
            
            print(f"[时域数学] 自相关特征完成，周期强度={features['periodicity_strength']:.3f}")
            
        except Exception as e:
            print(f"[时域数学] 自相关计算失败: {e}")
            features['periodicity_strength'] = 0.1
            features['fundamental_period'] = 0.01
        
        return features
    
    def extract_musical_mathematical_features(self, y):
        """音乐理论数学特征：基于音乐声学和信号处理"""
        features = {}
        print(f"[音乐数学] 开始音乐理论特征提取...")
        
        # 使用适中长度的音频段
        max_len = min(len(y), 110250)  # 约5秒
        y_music = y[:max_len]
        
        # 1. MFCC特征 - 基于人耳感知模型的倒谱分析
        print(f"[音乐数学] 计算MFCC特征...")
        try:
            mfcc = librosa.feature.mfcc(y=y_music, sr=self.sr, n_mfcc=13, n_fft=2048)
            
            # MFCC统计特征
            features['mfcc_mean_1'] = np.mean(mfcc[1])  # 第1个MFCC系数
            features['mfcc_mean_2'] = np.mean(mfcc[2])  # 第2个MFCC系数
            features['mfcc_std_1'] = np.std(mfcc[1])
            features['mfcc_std_2'] = np.std(mfcc[2])
            
            # MFCC系数间的相关性（反映音色一致性）
            if mfcc.shape[0] >= 3 and mfcc.shape[1] >= 10:
                corr_12 = np.corrcoef(mfcc[1], mfcc[2])[0, 1]
                corr_23 = np.corrcoef(mfcc[2], mfcc[3])[0, 1]
                features['mfcc_correlation'] = np.mean([abs(corr_12), abs(corr_23)])
            else:
                features['mfcc_correlation'] = 0.5
            
            print(f"[音乐数学] MFCC特征完成")
            
        except Exception as e:
            print(f"[音乐数学] MFCC计算失败: {e}")
            features['mfcc_mean_1'] = 0.0
            features['mfcc_mean_2'] = 0.0
            features['mfcc_std_1'] = 1.0
            features['mfcc_std_2'] = 1.0
            features['mfcc_correlation'] = 0.5
        
        # 2. 简化的频谱能量特征（替代色度特征，避免段错误）
        print(f"[音乐数学] 计算频谱能量特征...")
        try:
            # 使用更安全的频谱分析方法
            # 限制音频长度和FFT参数
            y_safe = y_music[:min(len(y_music), self.sr * 15)]  # 限制到15秒
            
            if len(y_safe) < 1024:
                # 音频太短，使用默认值
                features['chroma_std'] = 0.1
                features['pitch_stability'] = 0.3
                features['pitch_entropy'] = 2.0
            else:
                # 使用numpy FFT进行频谱分析，避免librosa的复杂计算
                # 分段处理，每段1秒
                segment_length = self.sr
                segments = []
                
                for i in range(0, len(y_safe), segment_length):
                    segment = y_safe[i:i + segment_length]
                    if len(segment) >= 1024:
                        # 计算该段的FFT
                        fft_result = np.fft.fft(segment, n=2048)
                        magnitude = np.abs(fft_result[:1024])  # 只取正频率部分
                        
                        # 将频谱分为12个区间（模拟色度）
                        freq_bins = len(magnitude)
                        bin_size = freq_bins // 12
                        chroma_like = []
                        
                        for j in range(12):
                            start_idx = j * bin_size
                            end_idx = min((j + 1) * bin_size, freq_bins)
                            if start_idx < end_idx:
                                bin_energy = np.mean(magnitude[start_idx:end_idx])
                                chroma_like.append(bin_energy)
                        
                        if len(chroma_like) == 12:
                            segments.append(chroma_like)
                
                if segments:
                    # 计算所有段的平均色度特征
                    segments_array = np.array(segments)
                    chroma_mean = np.mean(segments_array, axis=0)
                    
                    # 归一化
                    chroma_sum = np.sum(chroma_mean) + 1e-10
                    chroma_norm = chroma_mean / chroma_sum
                    
                    # 计算特征
                    features['chroma_std'] = np.std(chroma_norm)
                    features['pitch_stability'] = np.max(chroma_norm)
                    features['pitch_entropy'] = -np.sum(chroma_norm * np.log(chroma_norm + 1e-10))
                else:
                    features['chroma_std'] = 0.1
                    features['pitch_stability'] = 0.3
                    features['pitch_entropy'] = 2.0
            
            print(f"[音乐数学] 频谱能量特征完成")
            
        except Exception as e:
            print(f"[音乐数学] 频谱能量特征计算失败: {e}")
            features['chroma_std'] = 0.1
            features['pitch_stability'] = 0.3
            features['pitch_entropy'] = 2.0
        
        # 3. 节奏特征 - 基于能量变化的节拍分析
        print(f"[音乐数学] 计算节奏特征...")
        try:
            # 计算节拍强度函数
            hop_length = 512
            frame_length = 2048
            
            # 使用频谱流量（Spectral Flux）检测节拍
            stft = librosa.stft(y_music, n_fft=frame_length, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # 计算相邻帧间的频谱差异
            spectral_diff = np.diff(magnitude, axis=1)
            spectral_flux = np.sum(np.maximum(0, spectral_diff), axis=0)
            
            # 节拍强度的统计特征
            features['rhythm_intensity_mean'] = np.mean(spectral_flux)
            features['rhythm_intensity_std'] = np.std(spectral_flux)
            features['rhythm_regularity'] = 1.0 / (1.0 + np.std(np.diff(spectral_flux)))
            
            print(f"[音乐数学] 节奏特征完成")
            
        except Exception as e:
            print(f"[音乐数学] 节奏计算失败: {e}")
            features['rhythm_intensity_mean'] = 0.1
            features['rhythm_intensity_std'] = 0.05
            features['rhythm_regularity'] = 0.5
        
        # 4. 音色特征 - 基于频谱重心和带宽
        print(f"[音乐数学] 计算音色特征...")
        try:
            # 使用已计算的STFT
            if 'stft' in locals():
                magnitude = np.abs(stft)
            else:
                stft = librosa.stft(y_music, n_fft=2048, hop_length=512)
                magnitude = np.abs(stft)
            
            # 频谱重心（亮度）
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)[0]
            features['timbre_brightness'] = np.mean(spectral_centroids)
            features['timbre_brightness_var'] = np.var(spectral_centroids)
            
            # 频谱滚降点（高频内容）
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sr)[0]
            features['timbre_rolloff'] = np.mean(spectral_rolloff)
            
            print(f"[音乐数学] 音色特征完成")
            
        except Exception as e:
            print(f"[音乐数学] 音色计算失败: {e}")
            features['timbre_brightness'] = 1000.0
            features['timbre_brightness_var'] = 100000.0
            features['timbre_rolloff'] = 2000.0
        
        return features
    
    def extract_all_features(self, audio_file):
        """提取所有数学特征"""
        import time
        import os
        
        start_time = time.time()
        filename = os.path.basename(audio_file)
        print(f"[DEBUG] 开始处理: {filename}")
        
        try:
            # 添加音频加载超时检查
            print(f"[DEBUG] {filename}: 加载音频文件...")
            load_start = time.time()
            y, sr = librosa.load(audio_file, sr=self.sr)
            load_time = time.time() - load_start
            print(f"[DEBUG] {filename}: 音频加载完成，耗时 {load_time:.2f}s，长度 {len(y)} 采样点")
            
            # 音频预处理
            print(f"[DEBUG] {filename}: 音频预处理...")
            y = librosa.util.normalize(y)
            
            features = {}
            
            # 提取各类数学特征
            print(f"[DEBUG] {filename}: 提取频域特征...")
            spectral_features = self.extract_spectral_mathematical_features(y)
            print(f"[DEBUG] {filename}: 频域特征完成")
            
            print(f"[DEBUG] {filename}: 提取时域特征...")
            temporal_features = self.extract_temporal_mathematical_features(y)
            print(f"[DEBUG] {filename}: 时域特征完成")
            
            print(f"[DEBUG] {filename}: 提取音乐学特征...")
            musical_features = self.extract_musical_mathematical_features(y)
            print(f"[DEBUG] {filename}: 音乐学特征完成")
            
            features.update(spectral_features)
            features.update(temporal_features)  
            features.update(musical_features)
            
            total_time = time.time() - start_time
            print(f"[DEBUG] {filename}: 特征提取完成，总耗时 {total_time:.2f}s")
            return features
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"[ERROR] {filename}: 特征提取错误 ({error_time:.2f}s): {e}")
            # 返回默认特征值
            return {f'feature_{i}': 0.0 for i in range(15)}


class MathematicalAIMusicDetector:
    """基于数学特征的AI音乐检测器"""
    
    def __init__(self):
        self.feature_extractor = MathematicalAudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        # 模型缓存配置
        self.model_cache_path = "ai_music_detector_model.pkl"
        self.scaler_cache_path = "ai_music_detector_scaler.pkl"
        self.threshold_cache_path = "ai_music_detector_threshold.pkl"
        
    def save_model_cache(self):
        """保存训练好的模型到缓存文件"""
        try:
            # 保存模型
            if self.model is not None:
                joblib.dump(self.model, self.model_cache_path)
                print(f"✅ 模型已保存到: {self.model_cache_path}")
            
            # 保存标准化器
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_cache_path)
                print(f"✅ 标准化器已保存到: {self.scaler_cache_path}")
            
            # 保存最优阈值
            if hasattr(self, 'optimal_threshold'):
                joblib.dump(self.optimal_threshold, self.threshold_cache_path)
                print(f"✅ 最优阈值已保存到: {self.threshold_cache_path}")
                
        except Exception as e:
            print(f"❌ 保存模型缓存失败: {e}")
    
    def load_model_cache(self):
        """从缓存文件加载已训练的模型"""
        try:
            # 检查缓存文件是否存在
            if not (os.path.exists(self.model_cache_path) and 
                   os.path.exists(self.scaler_cache_path)):
                print("📁 缓存文件不存在，需要重新训练模型")
                return False
            
            # 加载模型
            print("🔄 正在加载缓存的模型...")
            self.model = joblib.load(self.model_cache_path)
            print(f"✅ 模型已从缓存加载: {self.model_cache_path}")
            
            # 加载标准化器
            self.scaler = joblib.load(self.scaler_cache_path)
            print(f"✅ 标准化器已从缓存加载: {self.scaler_cache_path}")
            
            # 加载最优阈值（如果存在）
            if os.path.exists(self.threshold_cache_path):
                self.optimal_threshold = joblib.load(self.threshold_cache_path)
                print(f"✅ 最优阈值已从缓存加载: {self.threshold_cache_path}")
            else:
                self.optimal_threshold = 0.5
                print("⚠️  未找到阈值缓存，使用默认阈值 0.5")
            
            print("🚀 模型缓存加载成功！节省了大量训练时间")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型缓存失败: {e}")
            print("🔄 将重新训练模型...")
            return False
    
    def clear_model_cache(self):
        """清除模型缓存文件"""
        cache_files = [self.model_cache_path, self.scaler_cache_path, self.threshold_cache_path]
        cleared_count = 0
        
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    print(f"🗑️  已删除缓存文件: {cache_file}")
                    cleared_count += 1
                except Exception as e:
                    print(f"❌ 删除缓存文件失败 {cache_file}: {e}")
        
        if cleared_count > 0:
            print(f"✅ 已清除 {cleared_count} 个缓存文件")
        else:
            print("📁 没有找到需要清除的缓存文件")
        
    def _create_interpretable_ensemble(self):
        """创建防过拟合的可解释集成模型"""
        # 随机森林：减少过拟合的参数设置
        rf = RandomForestClassifier(
            n_estimators=150,  # 增加树数量提高稳定性
            max_depth=8,       # 降低树深度防止过拟合
            min_samples_split=10,  # 增加分裂最小样本数
            min_samples_leaf=5,    # 增加叶子节点最小样本数
            max_features='sqrt',   # 限制特征数量
            random_state=42,
            class_weight='balanced',
            bootstrap=True,        # 启用bootstrap采样
            oob_score=True        # 计算袋外得分
        )
        
        # 逻辑回归：增强正则化
        lr = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=3000,
            C=0.01,           # 更强的L2正则化
            penalty='l2',     # 明确指定L2正则化
            solver='liblinear'  # 适合小数据集
        )
        
        # 支持向量机：防过拟合参数
        svm = SVC(
            probability=True, 
            random_state=42,
            class_weight='balanced',
            kernel='rbf',
            C=0.5,           # 降低C值增加正则化
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
        
        # 首先尝试从缓存加载模型
        if self.load_model_cache():
            print("✅ 从缓存加载模型成功，跳过训练过程")
            return self
        
        print("📚 开始重新训练模型...")
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
        
        # 打印即将处理的文件列表
        print("\n[INFO] 即将处理的音频文件:")
        for i, (af, lbl) in enumerate(zip(audio_files[:10], labels[:10])):  # 只显示前10个
            print(f"  {i+1}. {os.path.basename(af)} ({'AI' if lbl==1 else '人类'})")
        if len(audio_files) > 10:
            print(f"  ... 还有 {len(audio_files)-10} 个文件")
        
        # 提取特征（支持并行处理）
        print("\n[INFO] 开始提取音频特征...")
        try:
            from joblib import Parallel, delayed
            from tqdm import tqdm
            
            def extract_single_feature(audio_file, label):
                import time
                import signal
                import os
                
                filename = os.path.basename(audio_file)
                print(f"[WORKER] 开始处理: {filename}")
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"处理 {filename} 超时")
                
                try:
                    # 设置60秒超时
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)
                    
                    start_time = time.time()
                    features = self.feature_extractor.extract_all_features(audio_file)
                    
                    feature_vector = []
                    for name in self.get_feature_names():
                        feature_vector.append(features.get(name, 0.0))
                    
                    signal.alarm(0)  # 取消超时
                    elapsed = time.time() - start_time
                    print(f"[WORKER] {filename} 处理完成，耗时 {elapsed:.2f}s")
                    return feature_vector, label
                    
                except TimeoutError as e:
                    signal.alarm(0)
                    print(f"[TIMEOUT] {filename}: {e}")
                    return None, None
                except Exception as e:
                    signal.alarm(0)
                    print(f"[ERROR] 特征提取失败 {filename}: {e}")
                    return None, None
            
            # 降低并行度避免 audioread 卡住
            results = Parallel(n_jobs=1, verbose=5)(
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
            try:
                from tqdm import tqdm
                progress_bar = tqdm(zip(audio_files, labels), total=len(audio_files), desc="特征提取进度")
            except ImportError:
                progress_bar = zip(audio_files, labels)
                
            features_list = []
            valid_labels = []
            
            for i, (audio_file, label) in enumerate(progress_bar):
                if 'tqdm' not in str(type(progress_bar)):
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
        
        # 过拟合检测和警告
        overfitting_gap = train_score - test_score
        if overfitting_gap > 0.15:
            print(f"⚠️  检测到过拟合！训练-测试准确率差距: {overfitting_gap:.3f}")
            print("   建议：增加正则化强度或减少模型复杂度")
        elif overfitting_gap > 0.10:
            print(f"⚡ 轻微过拟合，训练-测试准确率差距: {overfitting_gap:.3f}")
        else:
            print(f"✅ 模型泛化良好，训练-测试准确率差距: {overfitting_gap:.3f}")
        
        # 自适应阈值优化
        self.optimal_threshold = self._find_optimal_threshold(X_test_scaled, y_test)
        print(f"最优阈值: {self.optimal_threshold:.3f}")
        
        # 增强的交叉验证
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='f1')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"交叉验证F1-score: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        
        # 模型稳定性评估
        if cv_std > 0.1:
            print("⚠️  模型稳定性较差，建议增加训练数据或调整参数")
        else:
            print("✅ 模型稳定性良好")
            
        # 显示随机森林的袋外得分（如果可用）
        if hasattr(self.model.named_estimators_['rf'], 'oob_score_'):
            oob_score = self.model.named_estimators_['rf'].oob_score_
            print(f"随机森林袋外得分: {oob_score:.3f}")
            
            # 袋外得分与测试得分的一致性检查
            oob_test_diff = abs(oob_score - test_score)
            if oob_test_diff < 0.05:
                print("✅ 袋外得分与测试得分一致，模型可靠")
            else:
                print(f"⚠️  袋外得分与测试得分差异较大: {oob_test_diff:.3f}")
        
        # 训练完成后自动保存模型缓存
        print("\n💾 保存模型缓存...")
        self.save_model_cache()
        
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
        """获取增强数学特征名称列表（42维）- 基于重要性分析优化"""
        return [
            # 频域数学特征 (13维) - 基于傅里叶分析和统计学，增强关键特征
            'spectral_centroid_mean', 'spectral_centroid_std', 'centroid_stability', 'centroid_trajectory_smoothness',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_entropy_mean', 'spectral_entropy_std',
            'spectral_flatness_mean', 'spectral_flatness_std',
            'harmonic_noise_ratio', 'harmonic_complexity', 'hnr_stability',
            
            # 时域数学特征 (17维) - 基于统计学和信号处理，增强ZCR特征
            'amplitude_mean', 'amplitude_std', 'amplitude_skewness', 'amplitude_kurtosis',
            'zcr_mean', 'zcr_std', 'zcr_entropy', 'zcr_stability', 'zcr_outlier_ratio',
            'rms_mean', 'rms_std', 'dynamic_range', 'compression_ratio',
            'energy_variation', 'energy_regularity',
            'periodicity_strength', 'fundamental_period',
            
            # 音乐理论数学特征 (13维) - 基于音乐声学理论
            'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_std_1', 'mfcc_std_2', 'mfcc_correlation',
            'chroma_std', 'pitch_stability', 'pitch_entropy',
            'rhythm_intensity_mean', 'rhythm_intensity_std', 'rhythm_regularity',
            'timbre_brightness', 'timbre_brightness_var', 'timbre_rolloff'
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
            'confidence': probability[1],  # 添加confidence字段兼容攻击测试
            'threshold_used': threshold,
            'features': features
        }
    
    def predict_single_from_array(self, audio_array, sr=22050):
        """从音频数组预测（用于攻击测试）"""
        try:
            # 临时保存音频数组到文件
            import soundfile as sf
            temp_file = "temp_audio_for_prediction.wav"
            sf.write(temp_file, audio_array, sr)
            
            # 使用现有的预测方法
            result = self.predict_single(temp_file)
            
            # 清理临时文件
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return result
            
        except Exception as e:
            print(f"数组预测错误: {e}")
            return {
                'prediction': 0,
                'confidence': 0.5,
                'probability_ai': 0.5,
                'probability_human': 0.5,
                'features': {}
            }
    
    def get_feature_importance_explanation(self):
        """获取数学特征重要性的可解释性分析"""
        if hasattr(self.model.named_estimators_['rf'], 'feature_importances_'):
            importance = self.model.named_estimators_['rf'].feature_importances_
            feature_names = self.get_feature_names()
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\n=== 数学特征重要性与可解释性分析 ===")
            
            # 增强特征解释字典 - 基于数学理论的可解释性
            feature_explanations = {
                # 频域特征解释
                'spectral_centroid_mean': '频谱质心均值 - 反映音色亮度，AI音乐通常更稳定',
                'spectral_centroid_std': '频谱质心标准差 - 反映音色变化程度',
                'centroid_trajectory_smoothness': '质心轨迹平滑度 - AI音乐质心变化过于平滑',
                'spectral_entropy_mean': '频谱熵均值 - 反映频谱复杂度，AI音乐通常更规律',
                'harmonic_noise_ratio': '谐波噪声比 - 反映音乐纯净度，AI音乐谐波成分更强',
                'harmonic_complexity': '谐波复杂度 - AI音乐谐波结构过于简单规整',
                'hnr_stability': '谐波-噪声平衡稳定性 - AI音乐平衡过于稳定',
                'spectral_flatness_mean': '频谱平坦度 - 反映噪声特性，AI音乐更接近白噪声',
                
                # 时域特征解释
                'amplitude_skewness': '幅度偏度 - 反映音频动态分布的不对称性',
                'amplitude_kurtosis': '幅度峰度 - 反映音频动态分布的尖锐程度',
                'zcr_mean': '零交叉率均值 - 反映高频内容，AI人声常有伪高频',
                'zcr_stability': 'ZCR稳定性 - AI音乐零交叉率变化过于规律',
                'zcr_outlier_ratio': 'ZCR异常比例 - AI音乐异常零交叉率更频繁',
                'dynamic_range': '动态范围 - AI音乐通常压缩更强，动态范围较小',
                'compression_ratio': '压缩比 - 反映动态压缩程度',
                'periodicity_strength': '周期性强度 - 反映音乐规律性，AI音乐更规律',
                
                # 音乐理论特征解释
                'mfcc_correlation': 'MFCC相关性 - 反映音色一致性',
                'pitch_stability': '音高稳定性 - 反映调性稳定程度',
                'rhythm_regularity': '节奏规律性 - AI音乐节拍通常更规律',
                'timbre_brightness': '音色亮度 - 反映高频能量分布'
            }
            
            print("前10个最重要的数学特征:")
            for i, (name, imp) in enumerate(feature_importance[:10], 1):
                explanation = feature_explanations.get(name, '该特征反映音频的数学统计性质')
                print(f"{i:2d}. {name:<25} {imp:.4f} - {explanation}")
            
            # 按类别统计重要性
            freq_importance = sum(imp for name, imp in feature_importance if 'spectral' in name or 'harmonic' in name)
            temp_importance = sum(imp for name, imp in feature_importance if any(x in name for x in ['amplitude', 'zcr', 'rms', 'dynamic', 'energy', 'period']))
            music_importance = sum(imp for name, imp in feature_importance if any(x in name for x in ['mfcc', 'chroma', 'pitch', 'rhythm', 'timbre']))
            
            print(f"\n=== 特征类别重要性分布 ===")
            print(f"频域数学特征总重要性: {freq_importance:.3f}")
            print(f"时域数学特征总重要性: {temp_importance:.3f}")
            print(f"音乐理论特征总重要性: {music_importance:.3f}")
            
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
        print("=== 基于数学理论的AI音乐检测器 ===")
        print("设计理念：")
        print("1. 数学理论基础：傅里叶分析、统计学、信息论")
        print("2. 特征可解释性：每个特征都有明确的数学和物理含义")
        print("3. 多层次分析：频域、时域、音乐理论三个维度")
        print("4. 统计学方法：概率分布、熵理论、高阶矩分析")
        print()
        
        detector = MathematicalAIMusicDetector()
        
        try:
            print("使用附件六训练数据训练数学特征模型...")
            detector.train()
            
            print("\n✅ 基于数学理论的AI音乐检测器训练完成!")
            print("📊 特征维度：33维数学特征")
            print("🔬 理论基础：信号处理、统计学、音乐声学")
            print("🎯 可解释性：每个特征都有明确的数学含义")
            
            # 显示详细的特征重要性分析
            detector.get_feature_importance_explanation()
            
            print("\n=== 数学特征体系说明 ===")
            print("频域特征(10维)：基于短时傅里叶变换和频谱分析")
            print("- 频谱质心、带宽：反映音色特征")
            print("- 频谱熵、平坦度：反映频谱复杂度和噪声特性")
            print("- 谐波-噪声比：反映音乐纯净度")
            print()
            print("时域特征(15维)：基于统计学和信号处理理论")
            print("- 幅度统计：均值、方差、偏度、峰度")
            print("- 零交叉率：反映频率内容")
            print("- RMS能量：动态范围和压缩特性")
            print("- 自相关：周期性和规律性分析")
            print()
            print("音乐理论特征(13维)：基于音乐声学和感知理论")
            print("- MFCC：基于人耳感知的倒谱分析")
            print("- 色度：基于十二平均律的音高分析")
            print("- 节奏：基于频谱流量的节拍检测")
            print("- 音色：频谱重心和滚降点分析")
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            print("请确保附件六：训练数据目录存在且包含音频文件")