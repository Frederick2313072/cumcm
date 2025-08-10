#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3解决方案：对附件四进行音频片段插入攻击测试
通过插入特定音频片段到附件四的AI生成音频中，混淆检测模型同时保持艺术性
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

class Problem3AudioAttacker:
    """问题3专用：附件四音频攻击器"""
    
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
    
    def generate_attack_segments(self):
        """生成攻击音频片段库"""
        print("\n=== 生成音频攻击片段库 ===")
        
        segments = {}
        
        # 1. 人声相关片段（针对人声AI音乐）
        print("生成人声攻击片段...")
        
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
        print("生成乐器攻击片段...")
        
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
        print("生成环境音攻击片段...")
        
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
        
        # 电子设备噪音 - 攻击频谱特征
        electronic_duration = 0.4
        electronic_t = np.linspace(0, electronic_duration, int(electronic_duration * self.sr), False)
        # 50Hz工频噪音
        hum_50hz = 0.02 * np.sin(2 * np.pi * 50 * electronic_t)
        # 高频数字噪音
        digital_noise = 0.01 * np.random.normal(0, 1, len(electronic_t))
        b, a = signal.butter(2, [3000, 8000], 'band', fs=self.sr)
        digital_filtered = signal.filtfilt(b, a, digital_noise)
        segments['electronic_hum'] = hum_50hz + digital_filtered
        
        # 4. 音乐理论攻击片段
        print("生成音乐理论攻击片段...")
        
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
        
        # 复杂和声 - 攻击和声分析
        chord_duration = 1.2
        chord_t = np.linspace(0, chord_duration, int(chord_duration * self.sr), False)
        # 不协和和弦 (C-F#-Bb-E)
        chord_freqs = [261.63, 369.99, 466.16, 329.63]
        chord_wave = np.zeros(len(chord_t))
        for i, freq in enumerate(chord_freqs):
            amplitude = 0.1 * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * chord_t))  # 轻微颤音
            chord_wave += amplitude * np.sin(2 * np.pi * freq * chord_t)
        
        chord_envelope = np.exp(-chord_t * 0.8)
        segments['dissonant_chord'] = chord_wave * chord_envelope
        
        print(f"生成完成，共{len(segments)}个攻击片段")
        
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
    
    def create_attacked_versions(self, attack_strategies=['light', 'moderate', 'aggressive']):
        """创建不同强度的攻击版本"""
        
        if self.original_audio is None:
            if not self.load_target_audio():
                return {}
        
        print("\n=== 创建攻击版本 ===")
        
        # 生成攻击片段
        attack_segments = self.generate_attack_segments()
        
        attacked_versions = {}
        
        for strategy in attack_strategies:
            print(f"\n--- 创建{strategy}攻击版本 ---")
            
            # 根据策略确定参数
            if strategy == 'light':
                num_insertions = 2
                insertion_method = 'smart'
                segment_types = ['breath', 'room_reverb']  # 较自然的片段
                
            elif strategy == 'moderate':
                num_insertions = 4
                insertion_method = 'smart'
                segment_types = ['breath', 'string_scratch', 'room_reverb', 'pedal_noise']
                
            elif strategy == 'aggressive':
                num_insertions = 6
                insertion_method = 'uniform'
                segment_types = list(attack_segments.keys())  # 使用所有片段类型
            
            # 找到插入点
            insertion_points = self.find_insertion_points(self.original_audio, insertion_method)
            
            # 限制插入点数量
            if len(insertion_points) > num_insertions:
                insertion_points = np.random.choice(insertion_points, num_insertions, replace=False)
                insertion_points = np.sort(insertion_points)
            
            print(f"插入点: {insertion_points}")
            
            # 开始插入
            modified_audio = self.original_audio.copy()
            insertions_made = []
            
            for i, insert_time in enumerate(insertion_points):
                # 随机选择片段类型
                segment_type = random.choice(segment_types)
                segment = attack_segments[segment_type].copy()
                
                # 随机选择插入方法
                insert_method = random.choice(['overlay', 'crossfade'])
                
                print(f"  插入 {segment_type} 于 {insert_time:.2f}秒 (方法: {insert_method})")
                
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
                print(f"  音频归一化: {max_val:.3f} -> 0.95")
            
            attacked_versions[strategy] = {
                'audio': modified_audio,
                'insertions': insertions_made,
                'num_insertions': len(insertions_made)
            }
            
            print(f"  {strategy}攻击版本创建完成，共插入{len(insertions_made)}个片段")
        
        return attacked_versions
    
    def test_attack_effectiveness(self, attacked_versions):
        """测试攻击效果"""
        
        if self.detector is None:
            self.load_detector()
        
        print("\n=== 测试攻击效果 ===")
        
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
                print(f"  置信度变化: {attacked_result.get('confidence', attacked_result['probability_ai']) - original_result.get('confidence', original_result['probability_ai']):.3f}")
                
                # 判断攻击是否成功
                prediction_changed = original_result['prediction'] != attacked_result['prediction']
                confidence_change = abs(attacked_result.get('confidence', attacked_result['probability_ai']) - 
                                      original_result.get('confidence', original_result['probability_ai']))
                
                print(f"  预测改变: {'是' if prediction_changed else '否'}")
                print(f"  置信度变化幅度: {confidence_change:.3f}")
                
                if prediction_changed:
                    print(f"  ✅ 攻击成功！模型预测被改变")
                elif confidence_change > 0.2:
                    print(f"  ⚠️  部分成功！显著影响了模型置信度")
                else:
                    print(f"  ❌ 攻击失败，模型保持稳定")
                
                results[strategy] = attacked_result
                
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return results
    
    def save_attacked_versions(self, attacked_versions, output_dir="problem3_results"):
        """保存攻击版本到文件"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== 保存攻击版本到 {output_dir} ===")
        
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
    
    def generate_analysis_report(self, attacked_versions, test_results, saved_files, output_dir="problem3_results"):
        """生成详细分析报告"""
        
        report_path = os.path.join(output_dir, "问题3_鲁棒性分析报告.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 问题3：AI音乐检测模型鲁棒性分析报告\n\n")
            f.write("## 实验目标\n")
            f.write("通过向附件四的AI生成音频中插入特定音频片段，测试问题一建立的检测模型的鲁棒性，")
            f.write("同时保持音乐的艺术性和可欣赏性。\n\n")
            
            f.write("## 实验设计\n\n")
            f.write("### 攻击片段设计\n")
            f.write("基于模型的关键特征，设计了以下类型的攻击片段：\n")
            f.write("1. **人声相关片段**：气息声、唇音 - 针对零交叉率和谐波特征\n")
            f.write("2. **乐器相关片段**：弦乐擦弦音、钢琴踏板噪音 - 针对频谱和动态特征\n")
            f.write("3. **环境音片段**：房间混响、电子噪音 - 针对空间感和频谱特征\n")
            f.write("4. **音乐理论片段**：微分音滑奏、复杂和声 - 针对音高稳定性和和声分析\n\n")
            
            f.write("### 攻击策略\n")
            f.write("- **轻度攻击**：插入2个自然片段（气息声、混响）\n")
            f.write("- **中度攻击**：插入4个混合片段，智能选择插入点\n")
            f.write("- **强度攻击**：插入6个各类片段，均匀分布\n\n")
            
            f.write("## 实验结果\n\n")
            
            # 原始音频结果
            original = test_results['original']
            f.write("### 原始音频分析\n")
            f.write(f"- **预测结果**: {'AI生成' if original['prediction'] else '人类创作'}\n")
            f.write(f"- **AI概率**: {original['probability_ai']:.3f}\n")
            f.write(f"- **置信度**: {original.get('confidence', original['probability_ai']):.3f}\n\n")
            
            # 攻击结果
            f.write("### 攻击效果分析\n\n")
            
            successful_attacks = 0
            significant_changes = 0
            
            for strategy in ['light', 'moderate', 'aggressive']:
                if strategy in test_results:
                    result = test_results[strategy]
                    data = attacked_versions[strategy]
                    
                    f.write(f"#### {strategy.capitalize()}攻击\n")
                    f.write(f"- **插入片段数**: {data['num_insertions']}\n")
                    f.write(f"- **预测结果**: {'AI生成' if result['prediction'] else '人类创作'}\n")
                    f.write(f"- **AI概率**: {result['probability_ai']:.3f}\n")
                    
                    confidence_change = abs(result.get('confidence', result['probability_ai']) - 
                                          original.get('confidence', original['probability_ai']))
                    prediction_changed = original['prediction'] != result['prediction']
                    
                    f.write(f"- **置信度变化**: {confidence_change:.3f}\n")
                    f.write(f"- **预测改变**: {'是' if prediction_changed else '否'}\n")
                    
                    if prediction_changed:
                        f.write("- **攻击效果**: ✅ 成功改变模型预测\n")
                        successful_attacks += 1
                    elif confidence_change > 0.2:
                        f.write("- **攻击效果**: ⚠️ 显著影响置信度\n")
                        significant_changes += 1
                    else:
                        f.write("- **攻击效果**: ❌ 影响较小\n")
                    
                    f.write(f"- **插入详情**:\n")
                    for insertion in data['insertions']:
                        f.write(f"  - {insertion['time']:.1f}s: {insertion['type']} ({insertion['method']})\n")
                    
                    f.write(f"- **音频文件**: {saved_files.get(strategy, 'N/A')}\n\n")
            
            f.write("## 鲁棒性评估\n\n")
            total_attacks = len([s for s in ['light', 'moderate', 'aggressive'] if s in test_results])
            
            f.write(f"### 攻击成功率\n")
            f.write(f"- **完全成功** (改变预测): {successful_attacks}/{total_attacks} ({successful_attacks/total_attacks*100:.1f}%)\n")
            f.write(f"- **部分成功** (显著影响): {significant_changes}/{total_attacks} ({significant_changes/total_attacks*100:.1f}%)\n")
            f.write(f"- **总体影响**: {(successful_attacks+significant_changes)}/{total_attacks} ({(successful_attacks+significant_changes)/total_attacks*100:.1f}%)\n\n")
            
            f.write("### 模型弱点分析\n")
            if successful_attacks > 0:
                f.write("模型存在以下鲁棒性问题：\n")
                f.write("1. **对环境音敏感**：混响、噪音等环境因素影响判断\n")
                f.write("2. **人声特征依赖**：过度依赖人声的完美特征\n")
                f.write("3. **频谱特征脆弱**：频谱相关特征容易被干扰\n")
            else:
                f.write("模型显示出较好的鲁棒性，能够抵抗多种类型的音频攻击。\n")
            
            f.write("\n### 艺术性保持\n")
            f.write("所有攻击版本都采用了以下技术保持音乐的艺术性：\n")
            f.write("- **智能插入点选择**：优先在安静段或音乐间隙插入\n")
            f.write("- **音量自适应调整**：根据原音频动态调整插入片段音量\n")
            f.write("- **淡入淡出处理**：避免突兀的音频拼接\n")
            f.write("- **交叉淡化技术**：自然过渡效果\n\n")
            
            f.write("## 结论与建议\n\n")
            f.write("### 主要发现\n")
            if successful_attacks > total_attacks * 0.5:
                f.write("1. **模型鲁棒性不足**：超过半数攻击成功，模型容易被欺骗\n")
                f.write("2. **特征工程需要改进**：当前特征对环境干扰过于敏感\n")
                f.write("3. **训练数据可能不够多样化**：缺乏包含各种环境音的训练样本\n\n")
            else:
                f.write("1. **模型具有一定鲁棒性**：能抵抗大部分攻击\n")
                f.write("2. **特征设计相对合理**：关键特征不易被简单干扰\n")
                f.write("3. **仍有改进空间**：部分攻击仍能影响模型判断\n\n")
            
            f.write("### 改进建议\n")
            f.write("1. **增强训练数据**：包含更多环境音、录音瑕疵的样本\n")
            f.write("2. **特征鲁棒化**：设计对噪音更不敏感的特征\n")
            f.write("3. **对抗训练**：使用攻击样本进行对抗训练\n")
            f.write("4. **集成方法**：结合多种检测方法提高鲁棒性\n\n")
            
            f.write("## 附录：实验文件\n\n")
            f.write("### 音频文件\n")
            for name, path in saved_files.items():
                f.write(f"- **{name}**: {path}\n")
            
            f.write(f"\n### 实验代码\n")
            f.write("- **主程序**: problem3_solution.py\n")
            f.write("- **检测器**: ai_music_detector.py\n")
            f.write("- **本报告**: {report_path}\n")
        
        print(f"\n分析报告已保存: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """运行完整的问题3分析"""
        
        print("="*60)
        print("问题3：AI音乐检测模型鲁棒性分析")
        print("="*60)
        
        # 1. 加载目标音频
        if not self.load_target_audio():
            return False
        
        # 2. 创建攻击版本
        attacked_versions = self.create_attacked_versions()
        
        if not attacked_versions:
            print("错误：未能创建攻击版本")
            return False
        
        # 3. 保存攻击版本
        saved_files = self.save_attacked_versions(attacked_versions)
        
        # 4. 测试攻击效果
        test_results = self.test_attack_effectiveness(attacked_versions)
        
        # 5. 生成分析报告
        report_path = self.generate_analysis_report(attacked_versions, test_results, saved_files)
        
        print("\n" + "="*60)
        print("问题3分析完成！")
        print("="*60)
        print(f"结果目录: problem3_results/")
        print(f"分析报告: {report_path}")
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
    
    # 创建攻击器并运行分析
    attacker = Problem3AudioAttacker(target_file)
    success = attacker.run_complete_analysis()
    
    if success:
        print("\n🎵 问题3解决方案执行成功！")
        print("📊 请查看生成的报告和音频文件")
        print("🔍 分析结果将帮助评估模型的鲁棒性")
    else:
        print("\n❌ 执行过程中出现错误")

if __name__ == "__main__":
    main()
