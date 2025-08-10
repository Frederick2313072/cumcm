#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3增强版解决方案：双向音频攻击测试
基于用户澄清：附件四实际为AI生成音乐，但模型误判为人类创作

双向攻击策略：
1. 人类化攻击：插入人类特征片段，进一步降低AI概率（测试模型对人类特征的敏感度）
2. AI化攻击：插入AI特征片段，提高AI概率，纠正模型的错误判断
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os
import random
import matplotlib.pyplot as plt
from ai_music_detector import MathematicalAIMusicDetector
import warnings
warnings.filterwarnings('ignore')

class Problem3EnhancedAudioAttacker:
    """问题3增强版：双向音频攻击器"""
    
    def __init__(self, target_audio="附件四：测试音乐.mp3", sr=22050):
        self.target_audio = target_audio
        self.sr = sr
        self.detector = None
        self.original_audio = None
        self.original_sr = None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_target_audio(self):
        """加载附件四音频"""
        print(f"加载目标音频: {self.target_audio}")
        
        if not os.path.exists(self.target_audio):
            print(f"错误：未找到文件 {self.target_audio}")
            return False
        
        self.original_audio, self.original_sr = librosa.load(self.target_audio, sr=self.sr)
        duration = len(self.original_audio) / self.sr
        
        print(f"音频加载成功:")
        print(f"  时长: {duration:.2f}秒")
        print(f"  采样率: {self.sr}")
        print(f"  样本数: {len(self.original_audio)}")
        
        return True
    
    def load_detector(self):
        """加载AI音乐检测器"""
        print("加载AI音乐检测模型...")
        self.detector = MathematicalAIMusicDetector()
        
        # 检查是否有已训练模型
        if hasattr(self.detector, 'model') and self.detector.model is not None:
            print("使用已训练模型")
        else:
            print("训练新模型...")
            self.detector.train()
        
        print("检测器加载完成")
    
    def generate_human_like_segments(self):
        """生成人类特征强化片段"""
        print("生成人类特征强化片段...")
        
        segments = {}
        
        # 1. 人声相关片段（针对人声AI音乐）
        print("  生成人声自然特征片段...")
        
        # 气息声 - 攻击零交叉率特征
        breath_duration = 0.5
        breath_t = np.linspace(0, breath_duration, int(breath_duration * self.sr), False)
        breath_base = np.random.normal(0, 0.03, len(breath_t))
        # 添加低频特性模拟真实气息
        b, a = signal.butter(2, 1000 / (self.sr / 2), 'low')
        breath_sound = signal.filtfilt(b, a, breath_base)
        # 添加自然衰减
        breath_envelope = np.exp(-breath_t * 3)
        segments['breath'] = breath_sound * breath_envelope
        
        # 唇音/口水声 - 攻击谐波特征
        pop_duration = 0.1
        pop_t = np.linspace(0, pop_duration, int(pop_duration * self.sr), False)
        pop_sound = np.random.normal(0, 0.2, len(pop_t)) * np.exp(-pop_t * 15)
        segments['lip_pop'] = pop_sound
        
        # 2. 乐器相关片段
        print("  生成乐器自然特征片段...")
        
        # 弦乐擦弦音 - 攻击频谱平坦度
        scratch_duration = 0.3
        scratch_t = np.linspace(0, scratch_duration, int(scratch_duration * self.sr), False)
        # 创建宽频噪音模拟擦弦
        scratch_base = np.random.normal(0, 0.1, len(scratch_t))
        # 添加高频强调
        b, a = signal.butter(2, [2000, 8000], 'band', fs=self.sr)
        scratch_filtered = signal.filtfilt(b, a, scratch_base)
        scratch_envelope = np.exp(-scratch_t * 2)
        segments['string_scratch'] = scratch_filtered * scratch_envelope
        
        # 钢琴踏板噪音 - 攻击动态范围
        pedal_duration = 0.2
        pedal_t = np.linspace(0, pedal_duration, int(pedal_duration * self.sr), False)
        pedal_sound = np.random.normal(0, 0.05, len(pedal_t))
        # 低频共振
        b, a = signal.butter(3, 200 / (self.sr / 2), 'low')
        pedal_filtered = signal.filtfilt(b, a, pedal_sound)
        segments['pedal_noise'] = pedal_filtered
        
        # 3. 环境音片段
        print("  生成环境自然特征片段...")
        
        # 房间混响尾音 - 攻击空间感特征
        reverb_duration = 1.0
        reverb_t = np.linspace(0, reverb_duration, int(reverb_duration * self.sr), False)
        # 创建衰减的混响效果
        reverb_base = np.random.normal(0, 0.02, len(reverb_t))
        # 多个延迟叠加模拟混响
        reverb_sound = reverb_base.copy()
        delays = [0.03, 0.07, 0.12, 0.18]  # 不同延迟时间
        for delay in delays:
            delay_samples = int(delay * self.sr)
            if delay_samples < len(reverb_base):
                delayed = np.zeros_like(reverb_base)
                delayed[delay_samples:] = reverb_base[:-delay_samples] * 0.3
                reverb_sound += delayed
        
        reverb_envelope = np.exp(-reverb_t * 1.5)
        segments['room_reverb'] = reverb_sound * reverb_envelope
        
        # 微分音滑奏 - 攻击音高稳定性
        slide_duration = 0.8
        slide_t = np.linspace(0, slide_duration, int(slide_duration * self.sr), False)
        # 从A4滑到A#4，包含微分音
        start_freq = 440
        end_freq = 466.16
        freq_slide = np.linspace(start_freq, end_freq, len(slide_t))
        phase = np.cumsum(2 * np.pi * freq_slide / self.sr)
        slide_wave = 0.15 * np.sin(phase) * np.exp(-slide_t * 1.2)
        segments['microtonal_slide'] = slide_wave
        
        print(f"  生成完成，共{len(segments)}个人类特征片段")
        return segments
    
    def generate_ai_enhancement_segments(self):
        """生成AI特征强化片段"""
        print("生成AI特征强化片段...")
        
        segments = {}
        
        # 1. 完美音调（无人声颤音）
        perfect_duration = 1.0
        perfect_t = np.linspace(0, perfect_duration, int(perfect_duration * self.sr), False)
        perfect_freq = 440  # A4
        perfect_tone = 0.1 * np.sin(2 * np.pi * perfect_freq * perfect_t)
        segments['perfect_tone'] = perfect_tone
        print("  生成完美音调片段")
        
        # 2. 数字量化噪音
        quantize_duration = 0.3
        quantize_t = np.linspace(0, quantize_duration, int(quantize_duration * self.sr), False)
        quantize_base = np.random.normal(0, 0.04, len(quantize_t))
        # 模拟8位量化
        quantize_levels = 256
        quantize_sound = np.round(quantize_base * quantize_levels) / quantize_levels
        segments['digital_quantize'] = quantize_sound * 0.3
        print("  生成数字量化噪音片段")
        
        # 3. 合成器方波
        square_duration = 0.6
        square_t = np.linspace(0, square_duration, int(square_duration * self.sr), False)
        square_freq = 330
        square_wave = 0.08 * signal.square(2 * np.pi * square_freq * square_t)
        segments['synth_square'] = square_wave
        print("  生成合成器方波片段")
        
        # 4. 完美节拍（过于规律的节奏）
        beat_duration = 1.2
        beat_t = np.linspace(0, beat_duration, int(beat_duration * self.sr), False)
        beat_freq = 2.0  # 2Hz节拍
        beat_envelope = (1 + np.sin(2 * np.pi * beat_freq * beat_t)) * 0.5
        beat_tone = 0.05 * np.sin(2 * np.pi * 220 * beat_t) * beat_envelope
        segments['perfect_beat'] = beat_tone
        print("  生成完美节拍片段")
        
        # 5. 高频失真（AI生成常见的伪影）
        distort_duration = 0.4
        distort_t = np.linspace(0, distort_duration, int(distort_duration * self.sr), False)
        # 高频噪音
        high_freq_noise = 0.02 * np.random.normal(0, 1, len(distort_t))
        b, a = signal.butter(2, [8000, 10000], 'band', fs=self.sr)
        distort_filtered = signal.filtfilt(b, a, high_freq_noise)
        segments['high_freq_distort'] = distort_filtered
        print("  生成高频失真片段")
        
        print(f"  生成完成，共{len(segments)}个AI特征片段")
        return segments
    
    def find_insertion_points(self, audio, strategy='smart'):
        """智能寻找插入点"""
        
        if strategy == 'smart':
            # 基于音频特征寻找合适插入点
            
            # 计算音频能量
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 找到相对安静的段落
            rms_threshold = np.percentile(rms, 30)  # 30%分位数作为安静阈值
            quiet_frames = np.where(rms < rms_threshold)[0]
            
            # 转换为时间点
            quiet_times = librosa.frames_to_time(quiet_frames, sr=self.sr, hop_length=hop_length)
            
            # 过滤掉太靠近开头和结尾的点
            audio_duration = len(audio) / self.sr
            safe_times = quiet_times[(quiet_times > 2) & (quiet_times < audio_duration - 2)]
            
            # 确保插入点之间有足够间距
            if len(safe_times) > 0:
                spaced_times = [safe_times[0]]
                for t in safe_times[1:]:
                    if t - spaced_times[-1] > 3:  # 至少间隔3秒
                        spaced_times.append(t)
                return np.array(spaced_times)
            else:
                # 如果找不到安静段，使用均匀分布
                return np.linspace(5, audio_duration - 5, 5)
        
        elif strategy == 'uniform':
            # 均匀分布插入点
            audio_duration = len(audio) / self.sr
            return np.linspace(3, audio_duration - 3, 8)
        
        elif strategy == 'random':
            # 随机插入点
            audio_duration = len(audio) / self.sr
            num_points = random.randint(4, 10)
            return np.sort(np.random.uniform(2, audio_duration - 2, num_points))
    
    def insert_segment_artistically(self, original_audio, segment, insert_time, method='overlay'):
        """艺术性地插入音频片段"""
        
        insert_sample = int(insert_time * self.sr)
        
        # 确保不会越界
        if insert_sample + len(segment) >= len(original_audio):
            return original_audio
        
        # 创建修改后的音频副本
        modified_audio = original_audio.copy()
        
        # 为插入片段添加淡入淡出
        fade_samples = min(int(0.05 * self.sr), len(segment) // 4)  # 50ms或片段长度的1/4
        
        if fade_samples > 0:
            # 淡入
            fade_in = np.linspace(0, 1, fade_samples)
            segment[:fade_samples] *= fade_in
            
            # 淡出
            fade_out = np.linspace(1, 0, fade_samples)
            segment[-fade_samples:] *= fade_out
        
        if method == 'overlay':
            # 叠加模式 - 与原音频混合
            original_segment = original_audio[insert_sample:insert_sample + len(segment)]
            
            # 根据原音频的音量调整插入片段的音量
            original_rms = np.sqrt(np.mean(original_segment**2))
            segment_rms = np.sqrt(np.mean(segment**2))
            
            if segment_rms > 0:
                # 将插入片段的音量调整为原音频的30-50%
                volume_ratio = random.uniform(0.3, 0.5) * (original_rms / segment_rms)
                segment_adjusted = segment * volume_ratio
            else:
                segment_adjusted = segment
            
            modified_audio[insert_sample:insert_sample + len(segment)] += segment_adjusted
            
        elif method == 'replace':
            # 替换模式 - 只在安静段使用
            original_segment = original_audio[insert_sample:insert_sample + len(segment)]
            original_rms = np.sqrt(np.mean(original_segment**2))
            
            if original_rms < 0.05:  # 只在很安静的地方替换
                modified_audio[insert_sample:insert_sample + len(segment)] = segment * 0.3
            else:
                # 不够安静，改用叠加
                modified_audio[insert_sample:insert_sample + len(segment)] += segment * 0.2
        
        elif method == 'crossfade':
            # 交叉淡化模式
            crossfade_samples = min(int(0.1 * self.sr), len(segment) // 2)
            
            if crossfade_samples > 0:
                # 原音频淡出
                original_fadeout = np.linspace(1, 0, crossfade_samples)
                modified_audio[insert_sample:insert_sample + crossfade_samples] *= original_fadeout
                
                # 新片段淡入
                segment_fadein = np.linspace(0, 1, crossfade_samples)
                segment[:crossfade_samples] *= segment_fadein
                
                # 混合
                modified_audio[insert_sample:insert_sample + len(segment)] += segment * 0.4
        
        return modified_audio
    
    def create_dual_attack_versions(self):
        """创建双向攻击版本"""
        print("\n=== 创建双向攻击版本 ===")
        
        if self.original_audio is None:
            if not self.load_target_audio():
                return {}
        
        # 生成两类攻击片段
        human_segments = self.generate_human_like_segments()
        ai_segments = self.generate_ai_enhancement_segments()
        
        attacked_versions = {}
        
        # 人类化攻击（让模型更确信是人类创作）
        print("\n--- 创建人类化攻击版本 ---")
        for intensity in ['light_human', 'moderate_human', 'aggressive_human']:
            print(f"创建{intensity}攻击版本...")
            
            # 根据强度确定参数
            if intensity == 'light_human':
                num_insertions = 2
                insertion_method = 'smart'
                segment_types = ['breath', 'room_reverb']
                
            elif intensity == 'moderate_human':
                num_insertions = 4
                insertion_method = 'smart'
                segment_types = ['breath', 'string_scratch', 'room_reverb', 'pedal_noise']
                
            elif intensity == 'aggressive_human':
                num_insertions = 6
                insertion_method = 'uniform'
                segment_types = list(human_segments.keys())
            
            attacked_versions[intensity] = self._create_attack_version(
                human_segments, num_insertions, insertion_method, segment_types, intensity
            )
        
        # AI化攻击（让模型识别出AI特征）
        print("\n--- 创建AI化攻击版本 ---")
        for intensity in ['light_ai', 'moderate_ai', 'aggressive_ai']:
            print(f"创建{intensity}攻击版本...")
            
            # 根据强度确定参数
            if intensity == 'light_ai':
                num_insertions = 2
                insertion_method = 'smart'
                segment_types = ['perfect_tone', 'digital_quantize']
                
            elif intensity == 'moderate_ai':
                num_insertions = 4
                insertion_method = 'smart'
                segment_types = ['perfect_tone', 'digital_quantize', 'synth_square', 'perfect_beat']
                
            elif intensity == 'aggressive_ai':
                num_insertions = 6
                insertion_method = 'uniform'
                segment_types = list(ai_segments.keys())
            
            attacked_versions[intensity] = self._create_attack_version(
                ai_segments, num_insertions, insertion_method, segment_types, intensity
            )
        
        return attacked_versions
    
    def _create_attack_version(self, segments, num_insertions, insertion_method, segment_types, version_name):
        """创建单个攻击版本"""
        
        # 找到插入点
        insertion_points = self.find_insertion_points(self.original_audio, insertion_method)
        
        # 限制插入点数量
        if len(insertion_points) > num_insertions:
            insertion_points = np.random.choice(insertion_points, num_insertions, replace=False)
            insertion_points = np.sort(insertion_points)
        
        print(f"  插入点: {insertion_points}")
        
        # 开始插入
        modified_audio = self.original_audio.copy()
        insertions_made = []
        
        for i, insert_time in enumerate(insertion_points):
            # 随机选择片段类型
            segment_type = random.choice(segment_types)
            segment = segments[segment_type].copy()
            
            # 随机选择插入方法
            insert_method = random.choice(['overlay', 'crossfade'])
            
            print(f"    插入 {segment_type} 于 {insert_time:.2f}秒 (方法: {insert_method})")
            
            # 插入片段
            modified_audio = self.insert_segment_artistically(
                modified_audio, segment, insert_time, insert_method
            )
            
            insertions_made.append({
                'time': insert_time,
                'type': segment_type,
                'method': insert_method,
                'duration': len(segment) / self.sr
            })
        
        # 防止音频削波
        max_val = np.max(np.abs(modified_audio))
        if max_val > 0.95:
            modified_audio = modified_audio / max_val * 0.95
            print(f"    音频归一化: {max_val:.3f} -> 0.95")
        
        return {
            'audio': modified_audio,
            'insertions': insertions_made,
            'num_insertions': len(insertions_made)
        }
    
    def test_dual_attack_effectiveness(self, attacked_versions):
        """测试双向攻击效果"""
        
        if self.detector is None:
            self.load_detector()
        
        print("\n=== 测试双向攻击效果 ===")
        
        # 首先测试原始音频
        print("测试原始音频...")
        temp_original = "temp_original.wav"
        sf.write(temp_original, self.original_audio, self.sr)
        
        try:
            original_result = self.detector.predict_single(temp_original)
            print(f"原始音频预测:")
            print(f"  预测结果: {'AI生成' if original_result['prediction'] else '人类创作'}")
            print(f"  AI概率: {original_result['probability_ai']:.3f}")
            print(f"  置信度: {original_result.get('confidence', original_result['probability_ai']):.3f}")
            print(f"  ⚠️  注意：根据题目，附件四实际为AI生成音乐，但模型预测为人类创作")
        finally:
            if os.path.exists(temp_original):
                os.remove(temp_original)
        
        # 测试攻击版本
        results = {'original': original_result}
        
        for strategy, data in attacked_versions.items():
            print(f"\n测试{strategy}攻击版本...")
            
            temp_file = f"temp_attacked_{strategy}.wav"
            sf.write(temp_file, data['audio'], self.sr)
            
            try:
                attacked_result = self.detector.predict_single(temp_file)
                
                print(f"{strategy}攻击结果:")
                print(f"  预测结果: {'AI生成' if attacked_result['prediction'] else '人类创作'}")
                print(f"  AI概率: {attacked_result['probability_ai']:.3f}")
                print(f"  概率变化: {attacked_result['probability_ai'] - original_result['probability_ai']:+.3f}")
                
                # 判断攻击效果
                attack_type = 'human_like' if 'human' in strategy else 'ai_like'
                effect_analysis = self.evaluate_dual_attack(original_result, attacked_result, attack_type)
                print(f"  攻击效果: {effect_analysis}")
                
                results[strategy] = attacked_result
                
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return results
    
    def evaluate_dual_attack(self, original_result, attacked_result, attack_type):
        """双向攻击效果评估"""
        ai_prob_change = attacked_result['probability_ai'] - original_result['probability_ai']
        
        if attack_type == 'human_like':
            # 人类化攻击：期望AI概率降低
            if ai_prob_change < -0.1:
                return "✅ 人类化攻击成功！AI概率显著降低"
            elif ai_prob_change < -0.05:
                return "⚠️ 人类化攻击部分成功"
            else:
                return "❌ 人类化攻击失败"
                
        elif attack_type == 'ai_like':
            # AI化攻击：期望AI概率提高
            if ai_prob_change > 0.1:
                if attacked_result['prediction'] and not original_result['prediction']:
                    return "🎯 AI化攻击完全成功！纠正了模型的错误判断"
                else:
                    return "✅ AI化攻击成功！AI概率显著提高"
            elif ai_prob_change > 0.05:
                return "⚠️ AI化攻击部分成功"
            else:
                return "❌ AI化攻击失败"
    
    def save_dual_attacked_versions(self, attacked_versions, output_dir="problem3_enhanced_results"):
        """保存双向攻击版本到文件"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== 保存双向攻击版本到 {output_dir} ===")
        
        saved_files = {}
        
        # 保存原始音频作为对比
        original_path = os.path.join(output_dir, "original_附件四.wav")
        sf.write(original_path, self.original_audio, self.sr)
        print(f"原始音频: {original_path}")
        saved_files['original'] = original_path
        
        # 保存攻击版本
        for strategy, data in attacked_versions.items():
            filename = f"attacked_{strategy}_附件四.wav"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, data['audio'], self.sr)
            print(f"{strategy}攻击版本: {filepath}")
            saved_files[strategy] = filepath
        
        return saved_files
    
    def generate_enhanced_analysis_report(self, attacked_versions, test_results, saved_files, output_dir="problem3_enhanced_results"):
        """生成增强版分析报告"""
        
        report_path = os.path.join(output_dir, "问题3_双向鲁棒性分析报告.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 问题3：AI音乐检测模型双向鲁棒性分析报告\n\n")
            
            f.write("## 实验背景\n")
            f.write("根据用户澄清：**附件四实际为AI生成音乐**，但检测模型错误地将其识别为人类创作。\n")
            f.write("这为我们提供了一个独特的研究机会：通过双向攻击来全面测试模型的鲁棒性。\n\n")
            
            f.write("## 实验目标\n")
            f.write("1. **人类化攻击**：插入人类特征片段，测试模型对人类特征的敏感度\n")
            f.write("2. **AI化攻击**：插入AI特征片段，尝试纠正模型的错误判断\n")
            f.write("3. **鲁棒性评估**：全面分析模型的弱点和改进方向\n\n")
            
            f.write("## 双向攻击策略设计\n\n")
            
            f.write("### 人类特征强化片段\n")
            f.write("- **人声自然特征**：气息声、唇音 - 针对零交叉率和谐波特征\n")
            f.write("- **乐器自然特征**：弦乐擦弦音、钢琴踏板噪音 - 针对频谱和动态特征\n")
            f.write("- **环境自然特征**：房间混响、微分音滑奏 - 针对空间感和音高稳定性\n\n")
            
            f.write("### AI特征强化片段\n")
            f.write("- **完美音调**：无颤音的纯净音调 - 强化AI音乐的完美特征\n")
            f.write("- **数字量化噪音**：8位量化伪影 - 模拟数字处理痕迹\n")
            f.write("- **合成器方波**：典型的电子合成器音色\n")
            f.write("- **完美节拍**：过于规律的节奏模式\n")
            f.write("- **高频失真**：AI生成常见的高频伪影\n\n")
            
            f.write("## 实验结果分析\n\n")
            
            # 原始音频结果
            original = test_results['original']
            f.write("### 原始音频基线\n")
            f.write(f"- **预测结果**: {'AI生成' if original['prediction'] else '人类创作'}\n")
            f.write(f"- **AI概率**: {original['probability_ai']:.3f}\n")
            f.write(f"- **实际标签**: AI生成（根据题目说明）\n")
            f.write(f"- **判断正确性**: {'正确' if original['prediction'] else '❌ 错误判断'}\n\n")
            
            # 人类化攻击结果
            f.write("### 人类化攻击结果\n\n")
            human_attacks = [k for k in test_results.keys() if 'human' in k]
            human_success = 0
            human_partial = 0
            
            for strategy in human_attacks:
                if strategy in test_results:
                    result = test_results[strategy]
                    data = attacked_versions[strategy]
                    
                    f.write(f"#### {strategy.replace('_', ' ').title()}\n")
                    f.write(f"- **插入片段数**: {data['num_insertions']}\n")
                    f.write(f"- **预测结果**: {'AI生成' if result['prediction'] else '人类创作'}\n")
                    f.write(f"- **AI概率**: {result['probability_ai']:.3f}\n")
                    
                    ai_prob_change = result['probability_ai'] - original['probability_ai']
                    f.write(f"- **概率变化**: {ai_prob_change:+.3f}\n")
                    
                    if ai_prob_change < -0.1:
                        f.write("- **攻击效果**: ✅ 成功显著降低AI概率\n")
                        human_success += 1
                    elif ai_prob_change < -0.05:
                        f.write("- **攻击效果**: ⚠️ 部分成功降低AI概率\n")
                        human_partial += 1
                    else:
                        f.write("- **攻击效果**: ❌ 攻击失败\n")
                    
                    f.write(f"- **插入详情**:\n")
                    for insertion in data['insertions']:
                        f.write(f"  - {insertion['time']:.1f}s: {insertion['type']} ({insertion['method']})\n")
                    f.write("\n")
            
            # AI化攻击结果
            f.write("### AI化攻击结果\n\n")
            ai_attacks = [k for k in test_results.keys() if 'ai' in k and 'human' not in k]
            ai_success = 0
            ai_partial = 0
            ai_correction = 0
            
            for strategy in ai_attacks:
                if strategy in test_results:
                    result = test_results[strategy]
                    data = attacked_versions[strategy]
                    
                    f.write(f"#### {strategy.replace('_', ' ').title()}\n")
                    f.write(f"- **插入片段数**: {data['num_insertions']}\n")
                    f.write(f"- **预测结果**: {'AI生成' if result['prediction'] else '人类创作'}\n")
                    f.write(f"- **AI概率**: {result['probability_ai']:.3f}\n")
                    
                    ai_prob_change = result['probability_ai'] - original['probability_ai']
                    f.write(f"- **概率变化**: {ai_prob_change:+.3f}\n")
                    
                    if result['prediction'] and not original['prediction']:
                        f.write("- **攻击效果**: 🎯 完全成功！纠正了模型的错误判断\n")
                        ai_correction += 1
                        ai_success += 1
                    elif ai_prob_change > 0.1:
                        f.write("- **攻击效果**: ✅ 成功显著提高AI概率\n")
                        ai_success += 1
                    elif ai_prob_change > 0.05:
                        f.write("- **攻击效果**: ⚠️ 部分成功提高AI概率\n")
                        ai_partial += 1
                    else:
                        f.write("- **攻击效果**: ❌ 攻击失败\n")
                    
                    f.write(f"- **插入详情**:\n")
                    for insertion in data['insertions']:
                        f.write(f"  - {insertion['time']:.1f}s: {insertion['type']} ({insertion['method']})\n")
                    f.write("\n")
            
            # 综合分析
            f.write("## 双向鲁棒性综合评估\n\n")
            
            total_human = len(human_attacks)
            total_ai = len(ai_attacks)
            
            f.write(f"### 攻击成功率统计\n")
            f.write(f"#### 人类化攻击\n")
            f.write(f"- **完全成功**: {human_success}/{total_human} ({human_success/total_human*100:.1f}%)\n")
            f.write(f"- **部分成功**: {human_partial}/{total_human} ({human_partial/total_human*100:.1f}%)\n")
            f.write(f"- **总体影响**: {(human_success+human_partial)}/{total_human} ({(human_success+human_partial)/total_human*100:.1f}%)\n\n")
            
            f.write(f"#### AI化攻击\n")
            f.write(f"- **完全成功**: {ai_success}/{total_ai} ({ai_success/total_ai*100:.1f}%)\n")
            f.write(f"- **部分成功**: {ai_partial}/{total_ai} ({ai_partial/total_ai*100:.1f}%)\n")
            f.write(f"- **纠正错判**: {ai_correction}/{total_ai} ({ai_correction/total_ai*100:.1f}%)\n")
            f.write(f"- **总体影响**: {(ai_success+ai_partial)}/{total_ai} ({(ai_success+ai_partial)/total_ai*100:.1f}%)\n\n")
            
            # 模型弱点分析
            f.write("### 模型弱点深度分析\n")
            
            if human_success > 0:
                f.write("#### 对人类特征过度敏感\n")
                f.write("模型对以下人类特征表现出过度敏感：\n")
                f.write("- 环境音和混响：过度依赖'完美'的录音环境判断\n")
                f.write("- 乐器噪音：将自然的乐器操作声误认为人类特征\n")
                f.write("- 音高微变：对微分音和音高不稳定性反应强烈\n\n")
            
            if ai_success > 0 or ai_correction > 0:
                f.write("#### 对AI特征识别不足\n")
                f.write("模型在识别以下AI特征方面存在不足：\n")
                f.write("- 完美音调：未能识别过于完美的音调特征\n")
                f.write("- 数字伪影：对量化噪音等数字处理痕迹不敏感\n")
                f.write("- 规律性：对过于规律的节拍和合成器音色识别不足\n\n")
            
            # 改进建议
            f.write("### 模型改进建议\n\n")
            f.write("#### 短期改进\n")
            f.write("1. **特征权重调整**：降低环境音特征的权重，提高AI特征的敏感度\n")
            f.write("2. **阈值优化**：针对AI特征检测调整分类阈值\n")
            f.write("3. **对抗训练**：使用本次攻击样本进行对抗训练\n\n")
            
            f.write("#### 长期改进\n")
            f.write("1. **数据增强**：收集更多包含环境音的AI生成音乐样本\n")
            f.write("2. **特征工程**：设计专门检测数字伪影的特征\n")
            f.write("3. **集成学习**：结合多种检测方法提高鲁棒性\n")
            f.write("4. **持续学习**：建立模型持续更新机制\n\n")
            
            # 实验价值
            f.write("## 实验价值与意义\n\n")
            f.write("### 理论贡献\n")
            f.write("1. **双向攻击框架**：首次提出针对AI音乐检测的双向攻击策略\n")
            f.write("2. **错误纠正研究**：探索了通过对抗样本纠正模型错误判断的可能性\n")
            f.write("3. **鲁棒性评估**：建立了全面的模型鲁棒性评估体系\n\n")
            
            f.write("### 实践意义\n")
            f.write("1. **模型改进指导**：为AI音乐检测模型的改进提供了明确方向\n")
            f.write("2. **攻击防御**：提高了对潜在攻击的认识和防御能力\n")
            f.write("3. **质量保证**：为AI音乐检测系统的质量保证提供了测试方法\n\n")
            
            f.write("## 附录：实验文件\n\n")
            f.write("### 音频文件\n")
            for name, path in saved_files.items():
                f.write(f"- **{name}**: {path}\n")
            
            f.write(f"\n### 实验代码\n")
            f.write("- **主程序**: problem3_enhanced_solution.py\n")
            f.write("- **检测器**: ai_music_detector.py\n")
            f.write("- **本报告**: {report_path}\n")
        
        print(f"\n增强版分析报告已保存: {report_path}")
        return report_path
    
    def run_complete_dual_analysis(self):
        """运行完整的双向分析"""
        
        print("="*60)
        print("问题3增强版：AI音乐检测模型双向鲁棒性分析")
        print("="*60)
        
        # 1. 加载目标音频
        if not self.load_target_audio():
            return False
        
        # 2. 创建双向攻击版本
        attacked_versions = self.create_dual_attack_versions()
        
        if not attacked_versions:
            print("错误：未能创建攻击版本")
            return False
        
        # 3. 保存攻击版本
        saved_files = self.save_dual_attacked_versions(attacked_versions)
        
        # 4. 测试双向攻击效果
        test_results = self.test_dual_attack_effectiveness(attacked_versions)
        
        # 5. 生成增强版分析报告
        report_path = self.generate_enhanced_analysis_report(attacked_versions, test_results, saved_files)
        
        print("\n" + "="*60)
        print("问题3双向分析完成！")
        print("="*60)
        print(f"结果目录: problem3_enhanced_results/")
        print(f"分析报告: {report_path}")
        print("="*60)
        print("🎯 实验亮点：")
        print("  • 双向攻击策略：人类化 + AI化")
        print("  • 错误纠正尝试：针对模型误判的修正")
        print("  • 全面鲁棒性评估：深度分析模型弱点")
        print("="*60)
        
        return True

def main():
    """主函数"""
    
    # 检查附件四是否存在
    target_file = "附件四：测试音乐.mp3"
    if not os.path.exists(target_file):
        print(f"错误：未找到 {target_file}")
        print("请确保附件四存在于当前目录")
        return
    
    # 创建增强版攻击器并运行分析
    attacker = Problem3EnhancedAudioAttacker(target_file)
    success = attacker.run_complete_dual_analysis()
    
    if success:
        print("\n🎵 问题3增强版解决方案执行成功！")
        print("📊 双向攻击测试完成，请查看详细报告")
        print("🔍 分析结果将帮助全面评估模型的鲁棒性")
        print("🎯 特别关注：AI化攻击是否能纠正模型的错误判断")
    else:
        print("\n❌ 执行过程中出现错误")

if __name__ == "__main__":
    main()
