#!/usr/bin/env python3

import os
import pandas as pd
import glob
from ai_music_detector import MathematicalAIMusicDetector
import warnings
warnings.filterwarnings('ignore')

def main():
    """一键运行AI音乐检测脚本"""
    
    print("=== AI生成音乐检测系统 ===")
    print("基于数学特征的可解释性检测模型")
    print()
    
    # 初始化检测器
    print("1. 初始化检测模型...")
    detector = MathematicalAIMusicDetector()
    detector.train()
    
    # 查找val_music文件夹
    val_music_dir = "val_music"
    if not os.path.exists(val_music_dir):
        print(f"错误：未找到 {val_music_dir} 文件夹")
        return
    
    # 获取所有音频文件
    audio_files = []
    for ext in ['*.mp3', '*.aac']:
        audio_files.extend(glob.glob(os.path.join(val_music_dir, ext)))
    
    if not audio_files:
        print(f"错误：在 {val_music_dir} 中未找到MP3或AAC文件")
        return
    
    print(f"2. 发现 {len(audio_files)} 个音频文件待处理")
    
    # 处理每个音频文件
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        print(f"   处理中 ({i}/{len(audio_files)}): {filename}")
        
        try:
            result = detector.predict_single(audio_file)
            results.append({
                '文件名称（不包含扩展名）': filename,
                '结果（是AI为1，不是AI为0）': result['prediction']
            })
            
        except Exception as e:
            print(f"   处理 {filename} 时出错: {e}")
            results.append({
                '文件名称（不包含扩展名）': filename,
                '结果（是AI为1，不是AI为0）': 0  # 默认值
            })
    
    # 生成输出文件
    output_filename = "2210556_val.xlsx"  # 使用学号作为队伍控制号
    df_results = pd.DataFrame(results)
    
    print(f"3. 保存结果到 {output_filename}")
    df_results.to_excel(output_filename, index=False)
    
    # 统计结果
    ai_count = df_results['结果（是AI为1，不是AI为0）'].sum()
    human_count = len(df_results) - ai_count
    
    print(f"4. 检测完成！")
    print(f"   AI生成音乐: {ai_count} 个")
    print(f"   人类创作音乐: {human_count} 个")
    print(f"   结果已保存至: {output_filename}")
    
    # 显示特征重要性
    detector.get_feature_importance_explanation()
    
    return df_results

if __name__ == "__main__":
    main()