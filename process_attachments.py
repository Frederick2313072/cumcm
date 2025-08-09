#!/usr/bin/env python3

import os
import zipfile
import pandas as pd
import shutil
from ai_music_detector import MathematicalAIMusicDetector
import warnings
warnings.filterwarnings('ignore')

def extract_audio_files():
    """解压附件一中的音频文件"""
    zip_file = "附件一：待评测音乐.zip"
    extract_dir = "extracted_music"
    
    if not os.path.exists(zip_file):
        print(f"错误：未找到 {zip_file}")
        return None
    
    print(f"解压 {zip_file}...")
    
    # 创建解压目录
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    
    # 解压文件
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    return extract_dir

def process_evaluation_set():
    """处理评估集音频文件"""
    
    print("=== AI生成音乐检测 - 处理附件一 ===")
    print()
    
    # 解压音频文件
    extract_dir = extract_audio_files()
    if not extract_dir:
        return
    
    # 初始化检测器
    print("1. 初始化数学特征检测模型...")
    detector = MathematicalAIMusicDetector()
    detector.train_with_synthetic_data()
    
    # 读取附件二模板
    template_file = "附件二：评估集结果.xlsx"
    df_template = pd.read_excel(template_file)
    
    print(f"2. 处理 {len(df_template)} 个音频文件...")
    
    # 查找所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.aac', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"   发现 {len(audio_files)} 个音频文件")
    
    # 创建文件名映射
    file_mapping = {}
    for audio_file in audio_files:
        basename = os.path.basename(audio_file)
        name_without_ext = os.path.splitext(basename)[0]
        # 尝试提取数字
        import re
        numbers = re.findall(r'\d+', name_without_ext)
        if numbers:
            file_number = int(numbers[0])
            file_mapping[file_number] = audio_file
    
    # 处理每个文件
    results = []
    ai_generated_files = []  # 保存AI生成的音频文件信息用于问题2
    
    for idx, row in df_template.iterrows():
        file_num = int(row['文件名称（不包含扩展名）'])
        
        if file_num in file_mapping:
            audio_file = file_mapping[file_num]
            print(f"   处理 ({idx+1}/{len(df_template)}): 文件 {file_num}")
            
            try:
                result = detector.predict_single(audio_file)
                prediction = result['prediction']
                probability = result['probability_ai']
                
                results.append({
                    '文件名称（不包含扩展名）': file_num,
                    '结果（是AI为1，不是AI为0）': prediction
                })
                
                # 如果判定为AI生成，保存用于问题2
                if prediction == 1:
                    ai_generated_files.append({
                        '文件名称（不包含扩展名）': file_num,
                        'probability': probability,
                        'audio_file': audio_file
                    })
                
            except Exception as e:
                print(f"   处理文件 {file_num} 时出错: {e}")
                results.append({
                    '文件名称（不包含扩展名）': file_num,
                    '结果（是AI为1，不是AI为0）': 0  # 默认为人类创作
                })
        else:
            print(f"   警告：未找到文件 {file_num}")
            results.append({
                '文件名称（不包含扩展名）': file_num,
                '结果（是AI为1，不是AI为0）': 0
            })
    
    # 保存结果到附件二
    df_results = pd.DataFrame(results)
    output_file = "附件二：评估集结果_filled.xlsx"
    df_results.to_excel(output_file, index=False)
    
    # 统计结果
    ai_count = df_results['结果（是AI为1，不是AI为0）'].sum()
    human_count = len(df_results) - ai_count
    
    print(f"3. 检测完成！")
    print(f"   AI生成音乐: {ai_count} 个")
    print(f"   人类创作音乐: {human_count} 个")
    print(f"   结果已保存至: {output_file}")
    
    # 显示特征重要性分析
    detector.get_feature_importance_explanation()
    
    # 为问题2准备数据
    if ai_generated_files:
        print(f"\n4. 为问题2准备评分数据 ({len(ai_generated_files)} 个AI生成音频)")
        generate_scoring_results(ai_generated_files, detector)
    
    return df_results, ai_generated_files

def generate_scoring_results(ai_generated_files, detector):
    """为问题2生成评分结果"""
    
    print("   正在进行AI音乐质量评分...")
    
    scoring_results = []
    
    for file_info in ai_generated_files:
        file_num = file_info['文件名称（不包含扩展名）']
        audio_file = file_info['audio_file'] 
        probability = file_info['probability']
        
        # 基于多个维度计算评分 (0-1范围)
        try:
            # 重新提取特征用于评分
            features = detector.feature_extractor.extract_all_features(audio_file)
            
            # 计算质量评分 (基于数学特征的合理性)
            quality_score = calculate_quality_score(features, probability)
            
            scoring_results.append({
                '文件名称（不包含扩展名）': file_num,
                '评分（范围0-1）': round(quality_score, 3)
            })
            
        except Exception as e:
            print(f"   评分文件 {file_num} 时出错: {e}")
            scoring_results.append({
                '文件名称（不包含扩展名）': file_num,
                '评分（范围0-1）': 0.5  # 默认中等评分
            })
    
    # 保存评分结果
    df_scoring = pd.DataFrame(scoring_results)
    output_scoring_file = "附件三：评分结果_filled.xlsx"
    df_scoring.to_excel(output_scoring_file, index=False)
    
    print(f"   评分结果已保存至: {output_scoring_file}")
    print(f"   平均评分: {df_scoring['评分（范围0-1）'].mean():.3f}")

def calculate_quality_score(features, ai_probability):
    """基于数学特征计算AI音乐质量评分"""
    
    # 质量评分考虑因素：
    # 1. 谐波质量 (30%)
    # 2. 音乐结构合理性 (25%)  
    # 3. 动态特征自然度 (25%)
    # 4. 整体AI置信度调节 (20%)
    
    score = 0.0
    
    # 1. 谐波质量评分
    harmonic_quality = min(1.0, features.get('harmonic_dominance', 0.5))
    harmonic_ratio = min(1.0, features.get('harmonic_noise_ratio', 1.0) / 4.0)
    harmonic_score = (harmonic_quality + harmonic_ratio) / 2 * 0.3
    
    # 2. 音乐结构合理性
    tempo_score = features.get('tempo_regularity', 0.5)
    tonal_score = features.get('tonal_stability', 0.5) 
    structure_score = (tempo_score + tonal_score) / 2 * 0.25
    
    # 3. 动态特征自然度 (过度规律会降分)
    centroid_regularity = features.get('centroid_regularity', 0.5)
    # 适度的不规律性是好的
    dynamic_naturalness = 1.0 - abs(centroid_regularity - 0.7)
    amplitude_naturalness = 1.0 - abs(features.get('amplitude_regularity', 0.5) - 0.6)
    dynamic_score = (dynamic_naturalness + amplitude_naturalness) / 2 * 0.25
    
    # 4. AI置信度调节
    # 高置信度的AI音乐如果质量好，评分应该适中
    confidence_adjustment = 0.2 * (1.0 - abs(ai_probability - 0.7))
    
    total_score = harmonic_score + structure_score + dynamic_score + confidence_adjustment
    
    # 确保评分在0-1范围内
    return max(0.0, min(1.0, total_score))

if __name__ == "__main__":
    process_evaluation_set()