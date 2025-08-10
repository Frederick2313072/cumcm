#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI音乐质量评价系统 - 问题2解决方案
基于数学建模的多维度AI音乐评分系统

设计理念：
1. 区分AI参与程度：辅助生成 vs 直接生成
2. 评估音乐质量：技术质量、艺术质量、听觉体验
3. 综合评分：基于层次分析法的加权评分模型
4. 数学基础：信号处理、统计学、音乐声学、心理声学
"""

import numpy as np
import librosa
import scipy.stats as stats
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import warnings
import os
import glob
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

class AIMusicQualityEvaluator:
    """基于数学理论的AI音乐质量评价器
    
    核心功能：
    1. AI参与度分析 - 区分人工辅助vs完全AI生成
    2. 多维度质量评估 - 技术、艺术、听觉体验
    3. 综合评分系统 - 0-100分制，基于AHP权重分配
    4. 可解释性报告 - 详细的评分依据和改进建议
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
        self.scaler = StandardScaler()
        self.ai_type_classifier = None
        self.quality_regressor = None
        
        # AHP层次分析法确定的权重 (基于专家评分)
        self.dimension_weights = {
            'ai_involvement': 0.25,      # AI参与度权重
            'technical_quality': 0.35,   # 技术质量权重  
            'artistic_quality': 0.25,    # 艺术质量权重
            'listening_experience': 0.15  # 听觉体验权重
        }
        
        # 评分区间定义
        self.score_ranges = {
            'human': (95, 100),                    # 人类创作
            'ai_assisted_high': (85, 95),         # 高质量AI辅助
            'ai_assisted_low': (70, 85),          # 低质量AI辅助  
            'ai_direct_high': (60, 80),           # 高质量AI直接生成
            'ai_direct_low': (20, 60),            # 低质量AI直接生成
            'ai_generated_poor': (0, 30)          # 质量极差的AI生成
        }
    
    def extract_ai_involvement_features(self, y: np.ndarray) -> Dict[str, float]:
        """提取AI参与度相关特征
        
        基于以下假设：
        1. 人工调校会留下微妙的不规律性痕迹
        2. 完全AI生成往往过于规律和"完美"
        3. 技术复杂度反映创作者的专业程度
        """
        features = {}
        
        print("[AI参与度] 分析人工调校痕迹...")
        
        # 1. 音高微调检测 - 人工调校会有细微的音高偏移
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr, threshold=0.1)
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if pitches[index, t] > 0 else 0
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if len(pitch_track) > 10:
                # 音高变化的微观不规律性 - 人工调校特征
                pitch_micro_variance = np.var(np.diff(pitch_track))
                features['pitch_micro_irregularity'] = min(pitch_micro_variance / 1000, 1.0)
                
                # 音高量化程度 - AI生成往往过于量化
                pitch_quantization = self._calculate_quantization_degree(pitch_track)
                features['pitch_quantization_degree'] = pitch_quantization
            else:
                features['pitch_micro_irregularity'] = 0.5
                features['pitch_quantization_degree'] = 0.5
        except:
            features['pitch_micro_irregularity'] = 0.5
            features['pitch_quantization_degree'] = 0.5
        
        # 2. 时值调整检测 - 人工调校的节拍微调
        print("[AI参与度] 分析节拍微调痕迹...")
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
            if len(beats) > 5:
                beat_intervals = np.diff(beats) / self.sr
                # 节拍间隔的微观变化 - 人类演奏特征
                beat_micro_variance = np.var(beat_intervals)
                features['beat_micro_irregularity'] = min(beat_micro_variance * 100, 1.0)
                
                # 节拍规律性 - AI生成往往过于规律
                beat_regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
                features['beat_regularity'] = np.clip(beat_regularity, 0, 1)
            else:
                features['beat_micro_irregularity'] = 0.5
                features['beat_regularity'] = 0.5
        except:
            features['beat_micro_irregularity'] = 0.5
            features['beat_regularity'] = 0.5
        
        # 3. 创作复杂度分析
        print("[AI参与度] 分析创作复杂度...")
        
        # 和声复杂度 - 基于简化的频谱分析（避免色度计算）
        try:
            # 使用安全的频谱分析替代色度特征
            y_short = y[:min(len(y), self.sr * 10)]  # 限制到10秒
            if len(y_short) >= 2048:
                # 使用numpy FFT进行安全的频谱分析
                fft_result = np.fft.fft(y_short, n=4096)
                magnitude = np.abs(fft_result[:2048])
                
                # 将频谱分为12个区间模拟和弦复杂度
                freq_bins = len(magnitude)
                bin_size = max(1, freq_bins // 12)
                active_bins = 0
                
                for i in range(12):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, freq_bins)
                    if start_idx < end_idx:
                        bin_energy = np.mean(magnitude[start_idx:end_idx])
                        if bin_energy > np.percentile(magnitude, 70):  # 活跃频段
                            active_bins += 1
                
                chord_complexity = active_bins
            else:
                chord_complexity = 3  # 默认值
            
            features['harmonic_complexity'] = min(chord_complexity / 6, 1.0)
        except:
            features['harmonic_complexity'] = 0.5
        
        # 旋律复杂度 - 基于音高变化熵
        if len(pitch_track) > 10:
            pitch_changes = np.diff(pitch_track)
            pitch_change_entropy = stats.entropy(np.histogram(pitch_changes, bins=20)[0] + 1e-10)
            features['melodic_complexity'] = min(pitch_change_entropy / 3, 1.0)
        else:
            features['melodic_complexity'] = 0.5
        
        # 4. 技术精细度评估
        print("[AI参与度] 评估技术精细度...")
        
        # 动态控制精细度 - 基于RMS能量变化
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_gradient = np.gradient(rms)
        dynamic_control_fineness = np.std(rms_gradient)
        features['dynamic_control_fineness'] = min(dynamic_control_fineness * 10, 1.0)
        
        # 频谱雕刻精细度 - 基于频谱变化
        stft = librosa.stft(y)
        spectral_gradient = np.mean(np.abs(np.gradient(np.abs(stft), axis=1)))
        features['spectral_sculpting_fineness'] = min(spectral_gradient / 100, 1.0)
        
        print(f"[AI参与度] 特征提取完成，共{len(features)}个特征")
        return features
    
    def extract_technical_quality_features(self, y: np.ndarray) -> Dict[str, float]:
        """提取技术质量特征
        
        基于客观的音频技术指标：
        1. 音频保真度 - 信噪比、失真度
        2. 混音质量 - 频谱平衡、立体声像
        3. 音色自然度 - 谐波结构、共振峰
        """
        features = {}
        
        print("[技术质量] 分析音频保真度...")
        
        # 1. 音频保真度指标
        
        # 信噪比估算 - 基于信号和噪声的能量比
        try:
            # 使用频谱门限法估算噪声
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            noise_threshold = np.percentile(magnitude, 10)  # 底部10%作为噪声
            signal_energy = np.mean(magnitude[magnitude > noise_threshold] ** 2)
            noise_energy = np.mean(magnitude[magnitude <= noise_threshold] ** 2)
            snr_estimate = 10 * np.log10(signal_energy / (noise_energy + 1e-10))
            features['snr_estimate'] = np.clip(snr_estimate / 60, 0, 1)  # 归一化到0-1
        except:
            features['snr_estimate'] = 0.5
        
        # 总谐波失真估算 - 基于谐波分析
        try:
            # 简化的THD估算：非基频能量占比
            pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
            fundamental_energy = np.sum(magnitudes[magnitudes > np.percentile(magnitudes, 90)])
            total_energy = np.sum(magnitudes)
            thd_estimate = 1 - (fundamental_energy / (total_energy + 1e-10))
            features['thd_estimate'] = 1 - np.clip(thd_estimate, 0, 1)  # 越小越好，取反
        except:
            features['thd_estimate'] = 0.5
        
        # 2. 混音质量指标
        print("[技术质量] 分析混音质量...")
        
        # 频谱平衡度 - 各频段能量分布均匀性
        try:
            # 将频谱分为低、中、高频三段
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(fft), 1/self.sr)
            magnitude_spectrum = np.abs(fft)
            
            # 频段划分：低频0-500Hz, 中频500-4000Hz, 高频4000Hz+
            low_freq_mask = (freqs >= 0) & (freqs <= 500)
            mid_freq_mask = (freqs > 500) & (freqs <= 4000)
            high_freq_mask = freqs > 4000
            
            low_energy = np.mean(magnitude_spectrum[low_freq_mask])
            mid_energy = np.mean(magnitude_spectrum[mid_freq_mask])
            high_energy = np.mean(magnitude_spectrum[high_freq_mask])
            
            # 频谱平衡度：三个频段能量的标准差越小越平衡
            freq_balance = 1 - np.std([low_energy, mid_energy, high_energy]) / np.mean([low_energy, mid_energy, high_energy])
            features['frequency_balance'] = np.clip(freq_balance, 0, 1)
        except:
            features['frequency_balance'] = 0.5
        
        # 动态范围质量
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range_db = 20 * np.log10(np.max(rms) / (np.min(rms[rms > 0]) + 1e-10))
        features['dynamic_range_quality'] = np.clip(dynamic_range_db / 40, 0, 1)  # 40dB为满分
        
        # 3. 音色自然度指标
        print("[技术质量] 分析音色自然度...")
        
        # 谐波结构自然度 - 基于HPSS分离
        try:
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_rms = librosa.feature.rms(y=harmonic)[0]
            percussive_rms = librosa.feature.rms(y=percussive)[0]
            
            # 谐波-冲击平衡度
            hp_balance = np.corrcoef(harmonic_rms, percussive_rms)[0, 1]
            features['harmonic_naturalness'] = (hp_balance + 1) / 2 if not np.isnan(hp_balance) else 0.5
        except:
            features['harmonic_naturalness'] = 0.5
        
        # 频谱一致性 - 时间维度上的频谱稳定性
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
            spectral_consistency = 1 - (np.std(spectral_centroids) / np.mean(spectral_centroids))
            features['spectral_consistency'] = np.clip(spectral_consistency, 0, 1)
        except:
            features['spectral_consistency'] = 0.5
        
        print(f"[技术质量] 特征提取完成，共{len(features)}个特征")
        return features
    
    def extract_artistic_quality_features(self, y: np.ndarray) -> Dict[str, float]:
        """提取艺术质量特征
        
        基于音乐理论和美学原理：
        1. 旋律创新性 - 音程分布、节奏复杂度
        2. 和声复杂度 - 和弦进行、调性分析
        3. 情感表达 - 音色变化、力度对比
        """
        features = {}
        
        print("[艺术质量] 分析旋律创新性...")
        
        # 1. 旋律创新性指标
        
        # 音程分布熵 - 旋律的音程使用多样性
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
            pitch_sequence = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                if pitches[index, t] > 0:
                    pitch_sequence.append(pitches[index, t])
            
            if len(pitch_sequence) > 5:
                intervals = np.diff(pitch_sequence)
                interval_hist, _ = np.histogram(intervals, bins=20)
                interval_entropy = stats.entropy(interval_hist + 1e-10)
                features['melodic_innovation'] = min(interval_entropy / 3, 1.0)
            else:
                features['melodic_innovation'] = 0.5
        except:
            features['melodic_innovation'] = 0.5
        
        # 节奏复杂度 - 基于onset检测
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=self.sr, units='frames')
            if len(onset_frames) > 5:
                onset_intervals = np.diff(onset_frames)
                rhythm_complexity = stats.entropy(np.histogram(onset_intervals, bins=10)[0] + 1e-10)
                features['rhythmic_complexity'] = min(rhythm_complexity / 2.5, 1.0)
            else:
                features['rhythmic_complexity'] = 0.5
        except:
            features['rhythmic_complexity'] = 0.5
        
        # 2. 和声复杂度指标
        print("[艺术质量] 分析和声复杂度...")
        
        # 和弦进行复杂度 - 基于安全的频谱变化分析
        try:
            # 使用安全的频谱分析替代色度特征
            y_short = y[:min(len(y), self.sr * 15)]  # 限制到15秒
            if len(y_short) >= 4096:
                # 分段分析频谱变化
                segment_length = 2048
                spectral_changes = []
                
                for i in range(0, len(y_short) - segment_length, segment_length // 2):
                    segment1 = y_short[i:i + segment_length]
                    segment2 = y_short[i + segment_length//2:i + segment_length//2 + segment_length]
                    
                    if len(segment1) == segment_length and len(segment2) == segment_length:
                        # 计算两段的频谱
                        fft1 = np.abs(np.fft.fft(segment1, n=2048)[:1024])
                        fft2 = np.abs(np.fft.fft(segment2, n=2048)[:1024])
                        
                        # 计算频谱变化
                        spectral_diff = np.mean(np.abs(fft2 - fft1))
                        spectral_changes.append(spectral_diff)
                
                if spectral_changes:
                    harmonic_motion = np.mean(spectral_changes)
                    features['harmonic_complexity'] = min(harmonic_motion / 1000, 1.0)
                else:
                    features['harmonic_complexity'] = 0.5
            else:
                features['harmonic_complexity'] = 0.5
        except:
            features['harmonic_complexity'] = 0.5
        
        # 调性稳定性分析 - 基于频谱能量分布的稳定性
        try:
            # 使用频谱分析替代色度分析
            if len(y_short) >= 2048:
                # 计算整段音频的频谱分布
                fft_result = np.fft.fft(y_short, n=4096)
                magnitude = np.abs(fft_result[:2048])
                
                # 将频谱分为12个区间（模拟十二平均律）
                freq_bins = len(magnitude)
                bin_size = max(1, freq_bins // 12)
                chroma_like = []
                
                for i in range(12):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, freq_bins)
                    if start_idx < end_idx:
                        bin_energy = np.mean(magnitude[start_idx:end_idx])
                        chroma_like.append(bin_energy)
                
                if len(chroma_like) == 12:
                    chroma_array = np.array(chroma_like)
                    chroma_norm = chroma_array / (np.sum(chroma_array) + 1e-10)
                    
                    # 计算与大调音阶的相似性
                    major_scale = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C大调
                    correlations = []
                    for i in range(12):
                        shifted_scale = np.roll(major_scale, i)
                        corr = np.corrcoef(chroma_norm, shifted_scale)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                    
                    if correlations:
                        tonal_clarity = np.max(correlations)
                        features['tonal_stability'] = (tonal_clarity + 1) / 2
                    else:
                        features['tonal_stability'] = 0.5
                else:
                    features['tonal_stability'] = 0.5
            else:
                features['tonal_stability'] = 0.5
        except:
            features['tonal_stability'] = 0.5
        
        # 3. 情感表达指标
        print("[艺术质量] 分析情感表达...")
        
        # 音色变化丰富度 - 基于MFCC变化
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            mfcc_variance = np.mean(np.var(mfccs, axis=1))
            features['timbral_variation'] = min(mfcc_variance / 10, 1.0)
        except:
            features['timbral_variation'] = 0.5
        
        # 力度对比度 - 基于RMS能量变化
        try:
            rms = librosa.feature.rms(y=y)[0]
            dynamic_contrast = (np.max(rms) - np.min(rms)) / (np.max(rms) + 1e-10)
            features['dynamic_expression'] = np.clip(dynamic_contrast, 0, 1)
        except:
            features['dynamic_expression'] = 0.5
        
        # 情感一致性 - 基于频谱质心轨迹的平滑度
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
            emotional_consistency = np.corrcoef(spectral_centroids[:-1], spectral_centroids[1:])[0, 1]
            features['emotional_consistency'] = (emotional_consistency + 1) / 2 if not np.isnan(emotional_consistency) else 0.5
        except:
            features['emotional_consistency'] = 0.5
        
        print(f"[艺术质量] 特征提取完成，共{len(features)}个特征")
        return features
    
    def extract_listening_experience_features(self, y: np.ndarray) -> Dict[str, float]:
        """提取听觉体验特征
        
        基于心理声学原理：
        1. 整体协调性 - 频谱相关性、时域一致性
        2. 动态变化 - 响度变化率、频谱演化
        3. 空间感 - 立体声宽度、深度感知
        """
        features = {}
        
        print("[听觉体验] 分析整体协调性...")
        
        # 1. 整体协调性指标
        
        # 频谱相关性 - 不同时间段频谱的相似性
        try:
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 计算相邻时间帧的频谱相关性
            correlations = []
            for i in range(magnitude.shape[1] - 1):
                corr = np.corrcoef(magnitude[:, i], magnitude[:, i + 1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            spectral_coherence = np.mean(correlations) if correlations else 0.5
            features['spectral_coherence'] = (spectral_coherence + 1) / 2
        except:
            features['spectral_coherence'] = 0.5
        
        # 时域一致性 - 基于自相关分析
        try:
            # 计算音频的自相关函数
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 寻找主要的周期性成分
            peaks = signal.find_peaks(autocorr[:len(autocorr)//4])[0]
            if len(peaks) > 0:
                periodicity_strength = autocorr[peaks[0]] / autocorr[0] if autocorr[0] != 0 else 0
                features['temporal_consistency'] = np.clip(periodicity_strength, 0, 1)
            else:
                features['temporal_consistency'] = 0.5
        except:
            features['temporal_consistency'] = 0.5
        
        # 2. 动态变化指标
        print("[听觉体验] 分析动态变化...")
        
        # 响度变化率 - 基于感知响度模型
        try:
            # 使用A-weighting近似感知响度
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            loudness_changes = np.abs(np.diff(rms))
            dynamic_activity = np.mean(loudness_changes) / (np.mean(rms) + 1e-10)
            features['dynamic_activity'] = min(dynamic_activity * 10, 1.0)
        except:
            features['dynamic_activity'] = 0.5
        
        # 频谱演化度 - 频谱随时间的变化程度
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
            spectral_evolution = np.std(spectral_centroids) / np.mean(spectral_centroids)
            features['spectral_evolution'] = min(spectral_evolution * 5, 1.0)
        except:
            features['spectral_evolution'] = 0.5
        
        # 3. 空间感指标（对单声道音频的近似分析）
        print("[听觉体验] 分析空间感...")
        
        # 频谱宽度 - 基于频谱带宽
        try:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
            spatial_width = np.mean(spectral_bandwidth) / (self.sr / 2)  # 归一化
            features['spatial_width'] = np.clip(spatial_width, 0, 1)
        except:
            features['spatial_width'] = 0.5
        
        # 深度感知 - 基于混响估算
        try:
            # 简化的混响检测：后期能量衰减分析
            energy_envelope = librosa.feature.rms(y=y, frame_length=4096, hop_length=1024)[0]
            
            # 寻找能量峰值后的衰减模式
            peaks = signal.find_peaks(energy_envelope, height=np.percentile(energy_envelope, 80))[0]
            
            if len(peaks) > 0:
                # 分析第一个峰值后的衰减
                peak_idx = peaks[0]
                if peak_idx < len(energy_envelope) - 10:
                    decay_segment = energy_envelope[peak_idx:peak_idx + 10]
                    decay_rate = np.polyfit(range(len(decay_segment)), decay_segment, 1)[0]
                    depth_perception = min(abs(decay_rate) * 100, 1.0)
                    features['depth_perception'] = depth_perception
                else:
                    features['depth_perception'] = 0.5
            else:
                features['depth_perception'] = 0.5
        except:
            features['depth_perception'] = 0.5
        
        # 整体平衡感 - 各频段能量的时间稳定性
        try:
            # 分析低中高频的时间稳定性
            stft = librosa.stft(y)
            freqs = librosa.fft_frequencies(sr=self.sr)
            
            low_band = np.mean(np.abs(stft[freqs <= 500]), axis=0)
            mid_band = np.mean(np.abs(stft[(freqs > 500) & (freqs <= 4000)]), axis=0)
            high_band = np.mean(np.abs(stft[freqs > 4000]), axis=0)
            
            # 计算各频段的稳定性
            low_stability = 1 - (np.std(low_band) / (np.mean(low_band) + 1e-10))
            mid_stability = 1 - (np.std(mid_band) / (np.mean(mid_band) + 1e-10))
            high_stability = 1 - (np.std(high_band) / (np.mean(high_band) + 1e-10))
            
            overall_balance = np.mean([low_stability, mid_stability, high_stability])
            features['overall_balance'] = np.clip(overall_balance, 0, 1)
        except:
            features['overall_balance'] = 0.5
        
        print(f"[听觉体验] 特征提取完成，共{len(features)}个特征")
        return features
    
    def _calculate_quantization_degree(self, pitch_sequence: List[float]) -> float:
        """计算音高的量化程度
        
        AI生成的音乐往往音高过于量化（接近半音的整数倍）
        人类演奏会有微妙的音高偏移
        """
        if len(pitch_sequence) < 5:
            return 0.5
        
        # 将音高转换为半音单位
        semitones = 12 * np.log2(np.array(pitch_sequence) / 440.0) + 69  # A4 = 440Hz = 69半音
        
        # 计算每个音高与最近半音的偏差
        deviations = np.abs(semitones - np.round(semitones))
        
        # 量化程度：偏差越小，量化程度越高
        quantization_degree = 1 - np.mean(deviations) / 0.5  # 0.5半音为最大合理偏差
        
        return np.clip(quantization_degree, 0, 1)
    
    def extract_comprehensive_features(self, audio_file: str) -> Dict[str, float]:
        """提取音频的全面特征"""
        print(f"\n[综合特征提取] 开始处理: {os.path.basename(audio_file)}")
        
        try:
            # 加载音频
            y, sr = librosa.load(audio_file, sr=self.sr)
            y = librosa.util.normalize(y)
            
            # 提取各维度特征
            ai_features = self.extract_ai_involvement_features(y)
            tech_features = self.extract_technical_quality_features(y)
            art_features = self.extract_artistic_quality_features(y)
            exp_features = self.extract_listening_experience_features(y)
            
            # 合并所有特征
            all_features = {}
            all_features.update({f'ai_{k}': v for k, v in ai_features.items()})
            all_features.update({f'tech_{k}': v for k, v in tech_features.items()})
            all_features.update({f'art_{k}': v for k, v in art_features.items()})
            all_features.update({f'exp_{k}': v for k, v in exp_features.items()})
            
            print(f"[综合特征提取] 完成，共提取{len(all_features)}个特征")
            return all_features
            
        except Exception as e:
            print(f"[错误] 特征提取失败 {audio_file}: {e}")
            # 返回默认特征值
            return {f'feature_{i}': 0.5 for i in range(20)}
    
    def calculate_dimension_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """根据特征计算各维度得分"""
        scores = {}
        
        # 1. AI参与度得分 (0-20分)
        ai_features = [v for k, v in features.items() if k.startswith('ai_')]
        if ai_features:
            # AI参与度越高，得分越高（表示更多人工参与）
            ai_involvement_raw = np.mean(ai_features)
            scores['ai_involvement'] = ai_involvement_raw * 20
        else:
            scores['ai_involvement'] = 10
        
        # 2. 技术质量得分 (0-35分)
        tech_features = [v for k, v in features.items() if k.startswith('tech_')]
        if tech_features:
            tech_quality_raw = np.mean(tech_features)
            scores['technical_quality'] = tech_quality_raw * 35
        else:
            scores['technical_quality'] = 17.5
        
        # 3. 艺术质量得分 (0-30分)
        art_features = [v for k, v in features.items() if k.startswith('art_')]
        if art_features:
            art_quality_raw = np.mean(art_features)
            scores['artistic_quality'] = art_quality_raw * 30
        else:
            scores['artistic_quality'] = 15
        
        # 4. 听觉体验得分 (0-15分)
        exp_features = [v for k, v in features.items() if k.startswith('exp_')]
        if exp_features:
            exp_quality_raw = np.mean(exp_features)
            scores['listening_experience'] = exp_quality_raw * 15
        else:
            scores['listening_experience'] = 7.5
        
        return scores
    
    def calculate_comprehensive_score(self, dimension_scores: Dict[str, float]) -> Tuple[float, str, Dict]:
        """计算综合评分"""
        
        # 加权求和
        total_score = sum(dimension_scores[dim] * self.dimension_weights[dim] 
                         for dim in dimension_scores.keys())
        
        # 确定等级
        if total_score >= 90:
            grade = "优秀"
            category = "human"
        elif total_score >= 80:
            grade = "良好"
            category = "ai_assisted_high"
        elif total_score >= 70:
            grade = "中等"
            category = "ai_assisted_low"
        elif total_score >= 60:
            grade = "及格"
            category = "ai_direct_high"
        elif total_score >= 40:
            grade = "较差"
            category = "ai_direct_low"
        else:
            grade = "差"
            category = "ai_generated_poor"
        
        # 详细分析
        analysis = {
            'total_score': total_score,
            'grade': grade,
            'category': category,
            'dimension_scores': dimension_scores,
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }
        
        # 分析优缺点
        for dim, score in dimension_scores.items():
            max_score = {'ai_involvement': 20, 'technical_quality': 35, 
                        'artistic_quality': 30, 'listening_experience': 15}[dim]
            percentage = score / max_score
            
            if percentage >= 0.8:
                analysis['strengths'].append(f"{dim}: {score:.1f}/{max_score} (优秀)")
            elif percentage <= 0.5:
                analysis['weaknesses'].append(f"{dim}: {score:.1f}/{max_score} (需改进)")
        
        # 改进建议
        if dimension_scores['technical_quality'] / 35 < 0.6:
            analysis['suggestions'].append("提升技术制作水平：改善音频保真度和混音质量")
        if dimension_scores['artistic_quality'] / 30 < 0.6:
            analysis['suggestions'].append("增强艺术表现力：丰富旋律创新和情感表达")
        if dimension_scores['ai_involvement'] / 20 < 0.4:
            analysis['suggestions'].append("增加人工创作参与：减少AI生成痕迹，增加人工调校")
        
        return total_score, grade, analysis
    
    def evaluate_single_audio(self, audio_file: str) -> Dict:
        """评估单个音频文件"""
        print(f"\n{'='*60}")
        print(f"开始评估: {os.path.basename(audio_file)}")
        print(f"{'='*60}")
        
        # 提取特征
        features = self.extract_comprehensive_features(audio_file)
        
        # 计算维度得分
        dimension_scores = self.calculate_dimension_scores(features)
        
        # 计算综合评分
        total_score, grade, analysis = self.calculate_comprehensive_score(dimension_scores)
        
        # 生成报告
        report = {
            'filename': os.path.basename(audio_file),
            'total_score': total_score,
            'grade': grade,
            'category': analysis['category'],
            'dimension_scores': dimension_scores,
            'detailed_analysis': analysis,
            'features': features
        }
        
        # 打印评估结果
        print(f"\n🎵 音频文件: {report['filename']}")
        print(f"📊 综合评分: {total_score:.1f}/100")
        print(f"📈 评价等级: {grade}")
        print(f"🏷️  音乐类型: {analysis['category']}")
        print(f"\n📋 详细得分:")
        for dim, score in dimension_scores.items():
            max_scores = {'ai_involvement': 20, 'technical_quality': 35, 
                         'artistic_quality': 30, 'listening_experience': 15}
            print(f"  • {dim}: {score:.1f}/{max_scores[dim]}")
        
        if analysis['strengths']:
            print(f"\n✅ 主要优点:")
            for strength in analysis['strengths']:
                print(f"  • {strength}")
        
        if analysis['weaknesses']:
            print(f"\n⚠️  主要缺点:")
            for weakness in analysis['weaknesses']:
                print(f"  • {weakness}")
        
        if analysis['suggestions']:
            print(f"\n💡 改进建议:")
            for suggestion in analysis['suggestions']:
                print(f"  • {suggestion}")
        
        return report

if __name__ == "__main__":
    print("=== 基于数学理论的AI音乐质量评价系统 ===")
    print("设计理念：")
    print("1. 多维度评价：AI参与度、技术质量、艺术质量、听觉体验")
    print("2. 数学基础：信号处理、统计学、音乐声学、心理声学")
    print("3. 可解释性：每个评分都有明确的数学和音乐理论依据")
    print("4. 公平性：区分AI辅助生成和AI直接生成，避免一刀切")
    print()
    
    # 创建评价器实例
    evaluator = AIMusicQualityEvaluator()
    
    # 示例：评估单个音频文件（如果存在的话）
    test_files = glob.glob("附件六：训练数据/1-*")[:3]  # 测试前3个文件
    
    if test_files:
        print(f"🔍 找到 {len(test_files)} 个测试文件，开始评估...")
        
        all_reports = []
        for audio_file in test_files:
            try:
                report = evaluator.evaluate_single_audio(audio_file)
                all_reports.append(report)
            except Exception as e:
                print(f"❌ 评估失败 {audio_file}: {e}")
        
        # 统计分析
        if all_reports:
            print(f"\n{'='*60}")
            print("📊 评估统计分析")
            print(f"{'='*60}")
            
            scores = [r['total_score'] for r in all_reports]
            print(f"平均分: {np.mean(scores):.1f}")
            print(f"最高分: {np.max(scores):.1f}")
            print(f"最低分: {np.min(scores):.1f}")
            print(f"标准差: {np.std(scores):.1f}")
            
            # 等级分布
            grades = [r['grade'] for r in all_reports]
            from collections import Counter
            grade_dist = Counter(grades)
            print(f"\n等级分布:")
            for grade, count in grade_dist.items():
                print(f"  {grade}: {count} 个")
    
    else:
        print("❌ 未找到测试音频文件")
        print("请确保训练数据目录存在且包含音频文件")
    
    print(f"\n✅ AI音乐质量评价系统测试完成!")
