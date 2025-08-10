#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3解决方案：AI音乐检测模型鲁棒性测试
通过插入特定音频片段来混淆检测模型，同时保持音乐艺术性
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os
import random
from ai_music_detector import MathematicalAIMusicDetector
import warnings
warnings.filterwarnings('ignore')

class AudioAttackGenerator:
    """音频攻击生成器 - 针对AI音乐检测模型的鲁棒性测试"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        self.detector = None
    
    def load_detector(self):
        """加载已训练的检测模型"""
        print("加载AI音乐检测模型...")
        self.detector = MathematicalAIMusicDetector()
        self.detector.train()
        print("模型加载完成")
    
    def generate_attack_segments(self):
        """生成各种攻击音频片段"""
        segments = {}
        
        print("生成音频攻击片段...")
        
        # 1. 频域攻击片段
        print("  生成频域攻击片段...")
        
        # 白噪声片段 - 攻击频谱平坦度
        duration = 0.5  # 0.5秒
        white_noise = np.random.normal(0, 0.1, int(duration * self.sr))
        segments['white_noise'] = white_noise
        
        # 粉红噪声片段 - 更自然的噪音
        pink_noise = self._generate_pink_noise(duration)
        segments['pink_noise'] = pink_noise
        
        # 纯音调片段 - 攻击频谱集中度
        freq = 440  # A4
        t = np.linspace(0, duration, int(duration * self.sr), False)
        sine_wave = 0.3 * np.sin(2 * np.pi * freq * t)
        # 添加包络以避免突兀
        envelope = np.exp(-t * 2)  # 指数衰减
        segments['sine_wave'] = sine_wave * envelope
        
        # 复合谐波 - 模拟钢琴和弦
        chord_freqs = [261.63, 329.63, 392.00]  # C大调和弦
        chord = np.zeros(int(duration * self.sr))
        for freq in chord_freqs:
            chord += 0.2 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 1.5)
        segments['chord'] = chord
        
        # 2. 时域攻击片段
        print("  生成时域攻击片段...")
        
        # 不规则零交叉率片段 - 模拟气息声
        breath_duration = 0.3
        breath_t = np.linspace(0, breath_duration, int(breath_duration * self.sr), False)
        breath_sound = np.random.normal(0, 0.05, len(breath_t))
        # 添加低频滤波模拟气息特征
        b, a = signal.butter(3, 800 / (self.sr / 2), 'low')
        breath_filtered = signal.filtfilt(b, a, breath_sound)
        segments['breath'] = breath_filtered
        
        # 动态范围攻击 - 突然的强弱变化
        dynamic_duration = 1.0
        dynamic_t = np.linspace(0, dynamic_duration, int(dynamic_duration * self.sr), False)
        # 创建动态变化的正弦波
        dynamic_wave = np.sin(2 * np.pi * 220 * dynamic_t)
        # 添加动态包络：安静-响亮-安静
        dynamic_envelope = np.concatenate([
            np.linspace(0.1, 1.0, len(dynamic_t)//3),
            np.ones(len(dynamic_t)//3),
            np.linspace(1.0, 0.1, len(dynamic_t) - 2*(len(dynamic_t)//3))
        ])
        segments['dynamic_change'] = dynamic_wave * dynamic_envelope
        
        # 3. 音乐理论攻击片段
        print("  生成音乐理论攻击片段...")
        
        # 颤音片段 - 攻击音高稳定性
        vibrato_duration = 0.8
        vibrato_t = np.linspace(0, vibrato_duration, int(vibrato_duration * self.sr), False)
        base_freq = 330
        vibrato_freq = 5  # 5Hz颤音
        vibrato_depth = 10  # 10Hz深度
        instantaneous_freq = base_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * vibrato_t)
        phase = np.cumsum(2 * np.pi * instantaneous_freq / self.sr)
        vibrato_wave = 0.4 * np.sin(phase) * np.exp(-vibrato_t * 0.5)
        segments['vibrato'] = vibrato_wave
        
        # 复杂节奏片段 - 攻击节奏规律性
        rhythm_duration = 2.0
        rhythm_samples = int(rhythm_duration * self.sr)
        rhythm_pattern = np.zeros(rhythm_samples)
        # 创建不规律的节拍模式
        beat_positions = [0, 0.3, 0.7, 1.1, 1.35, 1.8]  # 不规律时间点
        for pos in beat_positions:
            if pos < rhythm_duration:
                start_idx = int(pos * self.sr)
                beat_duration = int(0.1 * self.sr)  # 100ms的击打声
                if start_idx + beat_duration < len(rhythm_pattern):
                    # 创建击打声（快速衰减的噪音）
                    beat_t = np.linspace(0, 0.1, beat_duration, False)
                    beat_sound = np.random.normal(0, 0.3, beat_duration) * np.exp(-beat_t * 20)
                    rhythm_pattern[start_idx:start_idx + beat_duration] = beat_sound
        segments['complex_rhythm'] = rhythm_pattern
        
        print(f"  生成完成，共{len(segments)}个攻击片段")
        return segments
    
    def _generate_pink_noise(self, duration):
        """生成粉红噪声"""
        samples = int(duration * self.sr)
        white = np.random.randn(samples)
        
        # 简单的粉红噪声滤波器
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        
        pink = signal.lfilter(b, a, white)
        # 归一化
        pink = pink / np.max(np.abs(pink)) * 0.1
        return pink
    
    def insert_segments_artistic(self, original_audio, attack_segments, insertion_strategy='moderate'):
        """艺术性地插入攻击片段到原始音频中"""
        
        strategies = {
            'light': {'density': 0.02, 'volume': 0.3},      # 轻度攻击
            'moderate': {'density': 0.05, 'volume': 0.5},   # 中度攻击
            'aggressive': {'density': 0.1, 'volume': 0.7}   # 强度攻击
        }
        
        config = strategies.get(insertion_strategy, strategies['moderate'])
        
        # 复制原始音频
        modified_audio = original_audio.copy()
        total_duration = len(original_audio) / self.sr
        
        # 计算插入点数量
        num_insertions = int(total_duration * config['density'] * len(attack_segments))
        
        print(f"使用{insertion_strategy}策略，计划插入{num_insertions}个片段")
        
        # 找到合适的插入点（避免音频开头和结尾）
        safe_start = int(0.1 * len(original_audio))  # 前10%
        safe_end = int(0.9 * len(original_audio))    # 后10%
        
        insertion_points = sorted(random.sample(
            range(safe_start, safe_end), 
            min(num_insertions, safe_end - safe_start - 1000)
        ))
        
        # 随机选择攻击片段类型
        segment_types = list(attack_segments.keys())
        
        insertions_made = []
        
        for i, insert_pos in enumerate(insertion_points):
            # 随机选择攻击片段
            segment_type = random.choice(segment_types)
            segment = attack_segments[segment_type]
            
            # 调整音量
            segment_scaled = segment * config['volume']
            
            # 确保不会越界
            if insert_pos + len(segment_scaled) < len(modified_audio):
                # 选择插入方式：叠加或替换
                insertion_method = random.choice(['overlay', 'replace'])
                
                if insertion_method == 'overlay':
                    # 叠加到原音频上
                    modified_audio[insert_pos:insert_pos + len(segment_scaled)] += segment_scaled
                else:
                    # 替换原音频片段（用于静音段）
                    # 先检查该段是否相对安静
                    original_segment = original_audio[insert_pos:insert_pos + len(segment_scaled)]
                    if np.mean(np.abs(original_segment)) < 0.1:  # 安静段
                        modified_audio[insert_pos:insert_pos + len(segment_scaled)] = segment_scaled
                    else:
                        # 不够安静，使用叠加
                        modified_audio[insert_pos:insert_pos + len(segment_scaled)] += segment_scaled * 0.3
                
                insertions_made.append({
                    'position': insert_pos / self.sr,
                    'type': segment_type,
                    'method': insertion_method,
                    'duration': len(segment_scaled) / self.sr
                })
        
        # 防止音频削波
        max_val = np.max(np.abs(modified_audio))
        if max_val > 0.95:
            modified_audio = modified_audio / max_val * 0.95
        
        print(f"实际插入了{len(insertions_made)}个片段")
        
        return modified_audio, insertions_made
    
    def test_attack_effectiveness(self, original_audio, modified_audio, filename):
        """测试攻击效果"""
        if self.detector is None:
            self.load_detector()
        
        print(f"\n测试攻击效果: {filename}")
        
        # 保存修改后的音频到临时文件
        temp_file = f"temp_{filename}"
        sf.write(temp_file, modified_audio, self.sr)
        
        try:
            # 测试原始音频
            original_result = self.detector.predict_single_from_array(original_audio)
            
            # 测试修改后的音频
            modified_result = self.detector.predict_single(temp_file)
            
            print(f"原始音频预测: {original_result['prediction']} (置信度: {original_result['confidence']:.3f})")
            print(f"攻击后预测: {modified_result['prediction']} (置信度: {modified_result['confidence']:.3f})")
            
            # 计算攻击成功程度
            confidence_change = abs(original_result['confidence'] - modified_result['confidence'])
            prediction_changed = original_result['prediction'] != modified_result['prediction']
            
            print(f"置信度变化: {confidence_change:.3f}")
            print(f"预测结果改变: {'是' if prediction_changed else '否'}")
            
            attack_success = {
                'confidence_change': confidence_change,
                'prediction_changed': prediction_changed,
                'original_confidence': original_result['confidence'],
                'modified_confidence': modified_result['confidence']
            }
            
            return attack_success
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def run_robustness_test(self, test_audio_file, output_dir="attack_results"):
        """运行完整的鲁棒性测试"""
        
        print("="*60)
        print("AI音乐检测模型鲁棒性测试")
        print("="*60)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载测试音频
        print(f"加载测试音频: {test_audio_file}")
        original_audio, sr = librosa.load(test_audio_file, sr=self.sr)
        
        # 生成攻击片段
        attack_segments = self.generate_attack_segments()
        
        # 测试不同强度的攻击
        strategies = ['light', 'moderate', 'aggressive']
        results = {}
        
        for strategy in strategies:
            print(f"\n--- 测试{strategy}攻击策略 ---")
            
            # 插入攻击片段
            modified_audio, insertions = self.insert_segments_artistic(
                original_audio, attack_segments, strategy
            )
            
            # 保存攻击后的音频
            output_filename = f"attacked_{strategy}_{os.path.basename(test_audio_file)}"
            output_path = os.path.join(output_dir, output_filename)
            sf.write(output_path, modified_audio, self.sr)
            print(f"攻击后音频已保存: {output_path}")
            
            # 测试攻击效果
            attack_result = self.test_attack_effectiveness(
                original_audio, modified_audio, output_filename
            )
            
            results[strategy] = {
                'attack_result': attack_result,
                'insertions': insertions,
                'output_file': output_path
            }
        
        # 生成测试报告
        self._generate_test_report(results, output_dir)
        
        return results
    
    def _generate_test_report(self, results, output_dir):
        """生成测试报告"""
        report_path = os.path.join(output_dir, "robustness_test_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AI音乐检测模型鲁棒性测试报告\n")
            f.write("="*50 + "\n\n")
            
            for strategy, data in results.items():
                f.write(f"攻击策略: {strategy.upper()}\n")
                f.write("-" * 30 + "\n")
                
                attack_result = data['attack_result']
                f.write(f"置信度变化: {attack_result['confidence_change']:.3f}\n")
                f.write(f"预测改变: {'是' if attack_result['prediction_changed'] else '否'}\n")
                f.write(f"原始置信度: {attack_result['original_confidence']:.3f}\n")
                f.write(f"攻击后置信度: {attack_result['modified_confidence']:.3f}\n")
                f.write(f"插入片段数量: {len(data['insertions'])}\n")
                
                # 分析插入的片段类型
                segment_types = {}
                for insertion in data['insertions']:
                    seg_type = insertion['type']
                    segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
                
                f.write("插入片段类型统计:\n")
                for seg_type, count in segment_types.items():
                    f.write(f"  {seg_type}: {count}次\n")
                
                f.write(f"输出文件: {data['output_file']}\n\n")
            
            # 总结
            f.write("总结分析\n")
            f.write("-" * 20 + "\n")
            
            max_confidence_change = max(r['attack_result']['confidence_change'] for r in results.values())
            successful_attacks = sum(1 for r in results.values() if r['attack_result']['prediction_changed'])
            
            f.write(f"最大置信度变化: {max_confidence_change:.3f}\n")
            f.write(f"成功改变预测的攻击: {successful_attacks}/{len(results)}\n")
            
            if max_confidence_change > 0.2:
                f.write("结论: 模型存在鲁棒性问题，容易被特定音频片段混淆\n")
            else:
                f.write("结论: 模型具有较好的鲁棒性\n")
        
        print(f"\n测试报告已保存: {report_path}")

def main():
    """主函数"""
    print("问题3：AI音乐检测模型鲁棒性测试")
    
    # 检查是否有测试音频文件
    test_file = "附件四：测试音乐.mp3"
    if not os.path.exists(test_file):
        print(f"错误: 未找到测试文件 {test_file}")
        print("请确保附件四存在")
        return
    
    # 创建攻击生成器
    attacker = AudioAttackGenerator()
    
    # 运行鲁棒性测试
    results = attacker.run_robustness_test(test_file)
    
    print("\n" + "="*60)
    print("鲁棒性测试完成！")
    print("检查 attack_results 文件夹中的结果文件")
    print("="*60)

if __name__ == "__main__":
    main()

