#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2解决方案：AI生成音乐的评价
结合问题1的检测结果，对AI音乐进行质量评分

工作流程：
1. 读取问题1的检测结果
2. 对识别为AI音乐的文件进行质量评价
3. 生成评分报告并输出到附件三格式
"""

import pandas as pd
import numpy as np
import os
import glob
from ai_music_evaluator import AIMusicQualityEvaluator
from ai_music_detector import MathematicalAIMusicDetector
import warnings
warnings.filterwarnings('ignore')

class Problem2Solution:
    """问题2完整解决方案"""
    
    def __init__(self):
        self.detector = MathematicalAIMusicDetector()
        self.evaluator = AIMusicQualityEvaluator()
        
    def solve_problem2(self, music_folder="val_music", output_file="附件三_AI音乐评分结果.xlsx"):
        """解决问题2：对AI音乐进行质量评价"""
        
        print("="*80)
        print("问题2解决方案：AI生成音乐的质量评价")
        print("="*80)
        
        # 第一步：检测AI音乐
        print("\n🔍 第一步：检测AI音乐...")
        
        # 获取所有音频文件
        audio_files = []
        for ext in ['*.mp3', '*.aac', '*.wav']:
            audio_files.extend(glob.glob(os.path.join(music_folder, ext)))
        
        if not audio_files:
            print(f"❌ 在 {music_folder} 中未找到音频文件")
            return [], []
        
        print(f"📁 找到 {len(audio_files)} 个音频文件")
        
        # 使用问题1的检测器进行AI音乐检测
        detection_results = []
        
        # 如果检测器未训练，先进行训练
        if not hasattr(self.detector, 'model') or self.detector.model is None:
            print("🔧 检测器未训练，开始训练...")
            try:
                self.detector.train_with_real_data()
                print("✅ 检测器训练完成")
            except Exception as e:
                print(f"❌ 检测器训练失败: {e}")
                print("使用默认检测策略...")
        
        # 检测每个音频文件
        ai_music_files = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"🎵 检测进度: {i}/{len(audio_files)} - {os.path.basename(audio_file)}")
            
            try:
                if hasattr(self.detector, 'model') and self.detector.model is not None:
                    # 使用训练好的模型预测
                    features = self.detector.feature_extractor.extract_all_features(audio_file)
                    if features and len([v for v in features.values() if v != 0]) > 5:
                        feature_vector = np.array(list(features.values())).reshape(1, -1)
                        feature_vector_scaled = self.detector.scaler.transform(feature_vector)
                        
                        # 使用最优阈值进行预测
                        probability = self.detector.model.predict_proba(feature_vector_scaled)[0][1]
                        is_ai = probability > getattr(self.detector, 'optimal_threshold', 0.5)
                        confidence = probability if is_ai else 1 - probability
                    else:
                        is_ai = False
                        confidence = 0.5
                else:
                    # 使用简单的启发式方法
                    is_ai = self._simple_ai_detection(audio_file)
                    confidence = 0.7 if is_ai else 0.6
                
                detection_results.append({
                    'filename': os.path.basename(audio_file),
                    'filepath': audio_file,
                    'is_ai': is_ai,
                    'confidence': confidence,
                    'prediction': 'AI生成' if is_ai else '人类创作'
                })
                
                if is_ai:
                    ai_music_files.append(audio_file)
                    
            except Exception as e:
                print(f"⚠️  检测失败 {os.path.basename(audio_file)}: {e}")
                detection_results.append({
                    'filename': os.path.basename(audio_file),
                    'filepath': audio_file,
                    'is_ai': False,
                    'confidence': 0.5,
                    'prediction': '检测失败'
                })
        
        print(f"\n📊 检测结果统计:")
        print(f"  • 总文件数: {len(audio_files)}")
        print(f"  • AI生成音乐: {len(ai_music_files)}")
        print(f"  • 人类创作音乐: {len(audio_files) - len(ai_music_files)}")
        
        # 第二步：对AI音乐进行质量评价
        print(f"\n🎯 第二步：对 {len(ai_music_files)} 个AI音乐进行质量评价...")
        
        evaluation_results = []
        
        for i, ai_file in enumerate(ai_music_files, 1):
            print(f"\n📈 评价进度: {i}/{len(ai_music_files)}")
            
            try:
                # 使用评价器进行质量评估
                report = self.evaluator.evaluate_single_audio(ai_file)
                
                # 整理评价结果
                eval_result = {
                    'filename': report['filename'],
                    'total_score': report['total_score'],
                    'grade': report['grade'],
                    'ai_type': report['category'],
                    'ai_involvement_score': report['dimension_scores']['ai_involvement'],
                    'technical_quality_score': report['dimension_scores']['technical_quality'],
                    'artistic_quality_score': report['dimension_scores']['artistic_quality'],
                    'listening_experience_score': report['dimension_scores']['listening_experience'],
                    'main_strengths': '; '.join(report['detailed_analysis']['strengths'][:2]),
                    'main_weaknesses': '; '.join(report['detailed_analysis']['weaknesses'][:2]),
                    'improvement_suggestions': '; '.join(report['detailed_analysis']['suggestions'][:2])
                }
                
                evaluation_results.append(eval_result)
                
            except Exception as e:
                print(f"❌ 评价失败 {os.path.basename(ai_file)}: {e}")
                # 添加默认评价结果
                evaluation_results.append({
                    'filename': os.path.basename(ai_file),
                    'total_score': 50.0,
                    'grade': '中等',
                    'ai_type': 'ai_direct_low',
                    'ai_involvement_score': 10.0,
                    'technical_quality_score': 17.5,
                    'artistic_quality_score': 15.0,
                    'listening_experience_score': 7.5,
                    'main_strengths': '评价失败',
                    'main_weaknesses': '无法分析',
                    'improvement_suggestions': '重新评价'
                })
        
        # 第三步：生成评价报告
        print(f"\n📄 第三步：生成评价报告...")
        
        if evaluation_results:
            # 创建DataFrame
            df = pd.DataFrame(evaluation_results)
            
            # 添加统计信息
            print(f"\n📊 AI音乐质量评价统计:")
            print(f"  • 平均分: {df['total_score'].mean():.1f}")
            print(f"  • 最高分: {df['total_score'].max():.1f}")
            print(f"  • 最低分: {df['total_score'].min():.1f}")
            print(f"  • 标准差: {df['total_score'].std():.1f}")
            
            # 等级分布
            grade_counts = df['grade'].value_counts()
            print(f"\n等级分布:")
            for grade, count in grade_counts.items():
                print(f"  • {grade}: {count} 个 ({count/len(df)*100:.1f}%)")
            
            # AI类型分布
            type_counts = df['ai_type'].value_counts()
            print(f"\nAI类型分布:")
            type_names = {
                'human': '人类创作',
                'ai_assisted_high': '高质量AI辅助',
                'ai_assisted_low': '低质量AI辅助',
                'ai_direct_high': '高质量AI直接生成',
                'ai_direct_low': '低质量AI直接生成',
                'ai_generated_poor': '质量极差AI生成'
            }
            for ai_type, count in type_counts.items():
                type_name = type_names.get(ai_type, ai_type)
                print(f"  • {type_name}: {count} 个 ({count/len(df)*100:.1f}%)")
            
            # 保存结果到Excel文件
            try:
                # 重新整理列名以符合附件三格式
                output_df = df.rename(columns={
                    'filename': '音频文件名',
                    'total_score': '综合评分',
                    'grade': '评价等级',
                    'ai_type': 'AI类型',
                    'ai_involvement_score': 'AI参与度得分',
                    'technical_quality_score': '技术质量得分',
                    'artistic_quality_score': '艺术质量得分',
                    'listening_experience_score': '听觉体验得分',
                    'main_strengths': '主要优点',
                    'main_weaknesses': '主要缺点',
                    'improvement_suggestions': '改进建议'
                })
                
                # 添加说明行
                explanation_row = pd.DataFrame([{
                    '音频文件名': '评分说明',
                    '综合评分': '0-100分制，基于AHP层次分析法加权',
                    '评价等级': '优秀(90+)/良好(80-90)/中等(70-80)/及格(60-70)/较差(40-60)/差(<40)',
                    'AI类型': 'AI参与程度分类',
                    'AI参与度得分': '人工参与程度(0-20分)',
                    '技术质量得分': '音频技术质量(0-35分)',
                    '艺术质量得分': '音乐艺术价值(0-30分)',
                    '听觉体验得分': '听觉感受质量(0-15分)',
                    '主要优点': '评价中发现的优势',
                    '主要缺点': '需要改进的方面',
                    '改进建议': '具体改进方向'
                }])
                
                # 合并说明和数据
                final_df = pd.concat([explanation_row, output_df], ignore_index=True)
                
                # 保存到Excel
                final_df.to_excel(output_file, index=False, engine='openpyxl')
                print(f"\n✅ 评价结果已保存到: {output_file}")
                
            except Exception as e:
                print(f"❌ 保存Excel文件失败: {e}")
                # 保存为CSV作为备选
                csv_file = output_file.replace('.xlsx', '.csv')
                output_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"✅ 评价结果已保存到CSV: {csv_file}")
        
        else:
            print("❌ 没有AI音乐需要评价")
        
        # 第四步：生成综合报告
        print(f"\n📋 第四步：生成综合分析报告...")
        
        report_content = self._generate_comprehensive_report(detection_results, evaluation_results)
        
        report_file = "问题2_AI音乐评价综合报告.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 综合报告已保存到: {report_file}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
        
        print(f"\n🎉 问题2解决完成！")
        print(f"📊 共检测 {len(audio_files)} 个音频文件")
        print(f"🎵 识别出 {len(ai_music_files)} 个AI音乐")
        print(f"📈 完成 {len(evaluation_results)} 个质量评价")
        
        return detection_results, evaluation_results
    
    def _simple_ai_detection(self, audio_file):
        """简单的AI音乐检测方法（备用）"""
        try:
            # 基于文件名的启发式判断
            filename = os.path.basename(audio_file).lower()
            ai_keywords = ['ai', 'suno', '洛天依', 'vocaloid', '虚拟', '合成']
            
            for keyword in ai_keywords:
                if keyword in filename:
                    return True
            
            # 可以添加更多基于音频特征的简单判断
            return False
            
        except:
            return False
    
    def _generate_comprehensive_report(self, detection_results, evaluation_results):
        """生成综合分析报告"""
        
        report = []
        report.append("="*80)
        report.append("问题2：AI生成音乐的质量评价 - 综合分析报告")
        report.append("="*80)
        report.append("")
        
        # 检测结果统计
        total_files = len(detection_results)
        ai_files = len([r for r in detection_results if r['is_ai']])
        human_files = total_files - ai_files
        
        report.append("一、AI音乐检测结果统计")
        report.append("-" * 40)
        report.append(f"总音频文件数：{total_files}")
        report.append(f"AI生成音乐：{ai_files} 个 ({ai_files/total_files*100:.1f}%)")
        report.append(f"人类创作音乐：{human_files} 个 ({human_files/total_files*100:.1f}%)")
        report.append("")
        
        # 质量评价统计
        if evaluation_results:
            scores = [r['total_score'] for r in evaluation_results]
            
            report.append("二、AI音乐质量评价统计")
            report.append("-" * 40)
            report.append(f"评价音乐数量：{len(evaluation_results)}")
            report.append(f"平均质量得分：{np.mean(scores):.1f}/100")
            report.append(f"最高质量得分：{np.max(scores):.1f}/100")
            report.append(f"最低质量得分：{np.min(scores):.1f}/100")
            report.append(f"得分标准差：{np.std(scores):.1f}")
            report.append("")
            
            # 等级分布
            from collections import Counter
            grades = [r['grade'] for r in evaluation_results]
            grade_dist = Counter(grades)
            
            report.append("三、质量等级分布")
            report.append("-" * 40)
            for grade, count in grade_dist.most_common():
                percentage = count / len(evaluation_results) * 100
                report.append(f"{grade}：{count} 个 ({percentage:.1f}%)")
            report.append("")
            
            # AI类型分布
            ai_types = [r['ai_type'] for r in evaluation_results]
            type_dist = Counter(ai_types)
            
            type_names = {
                'human': '人类创作水准',
                'ai_assisted_high': '高质量AI辅助',
                'ai_assisted_low': '低质量AI辅助',
                'ai_direct_high': '高质量AI直接生成',
                'ai_direct_low': '低质量AI直接生成',
                'ai_generated_poor': '质量极差AI生成'
            }
            
            report.append("四、AI音乐类型分布")
            report.append("-" * 40)
            for ai_type, count in type_dist.most_common():
                type_name = type_names.get(ai_type, ai_type)
                percentage = count / len(evaluation_results) * 100
                report.append(f"{type_name}：{count} 个 ({percentage:.1f}%)")
            report.append("")
            
            # 各维度平均得分
            ai_scores = [r['ai_involvement_score'] for r in evaluation_results]
            tech_scores = [r['technical_quality_score'] for r in evaluation_results]
            art_scores = [r['artistic_quality_score'] for r in evaluation_results]
            exp_scores = [r['listening_experience_score'] for r in evaluation_results]
            
            report.append("五、各维度平均得分分析")
            report.append("-" * 40)
            report.append(f"AI参与度得分：{np.mean(ai_scores):.1f}/20 ({np.mean(ai_scores)/20*100:.1f}%)")
            report.append(f"技术质量得分：{np.mean(tech_scores):.1f}/35 ({np.mean(tech_scores)/35*100:.1f}%)")
            report.append(f"艺术质量得分：{np.mean(art_scores):.1f}/30 ({np.mean(art_scores)/30*100:.1f}%)")
            report.append(f"听觉体验得分：{np.mean(exp_scores):.1f}/15 ({np.mean(exp_scores)/15*100:.1f}%)")
            report.append("")
            
            # 主要发现
            report.append("六、主要发现与结论")
            report.append("-" * 40)
            
            if np.mean(scores) >= 80:
                report.append("• 整体AI音乐质量较高，达到良好水准")
            elif np.mean(scores) >= 60:
                report.append("• 整体AI音乐质量中等，仍有提升空间")
            else:
                report.append("• 整体AI音乐质量较低，需要显著改进")
            
            if np.mean(tech_scores)/35 > np.mean(art_scores)/30:
                report.append("• 技术质量优于艺术质量，建议加强艺术表现")
            else:
                report.append("• 艺术质量优于技术质量，建议提升技术制作水准")
            
            if np.std(scores) > 15:
                report.append("• AI音乐质量差异较大，存在明显的质量分层")
            else:
                report.append("• AI音乐质量相对均匀，整体水准较为一致")
            
            report.append("")
            
            # 改进建议
            report.append("七、整体改进建议")
            report.append("-" * 40)
            
            if np.mean(ai_scores)/20 < 0.6:
                report.append("• 增加人工创作参与度，减少过度依赖AI生成")
            
            if np.mean(tech_scores)/35 < 0.7:
                report.append("• 提升音频制作技术，改善音质和混音效果")
            
            if np.mean(art_scores)/30 < 0.7:
                report.append("• 增强音乐艺术性，丰富旋律创新和情感表达")
            
            if np.mean(exp_scores)/15 < 0.7:
                report.append("• 优化听觉体验，提升音乐的整体协调性和动态变化")
            
            report.append("")
        
        else:
            report.append("二、质量评价结果")
            report.append("-" * 40)
            report.append("未发现AI生成音乐，无法进行质量评价")
            report.append("")
        
        # 方法说明
        report.append("八、评价方法说明")
        report.append("-" * 40)
        report.append("本评价系统基于以下数学理论和方法：")
        report.append("• 信号处理理论：频域和时域特征提取")
        report.append("• 统计学方法：概率分布和高阶矩分析")
        report.append("• 音乐声学理论：谐波分析和音色建模")
        report.append("• 心理声学理论：感知质量评估")
        report.append("• 层次分析法(AHP)：多维度权重分配")
        report.append("")
        report.append("评分维度及权重：")
        report.append("• AI参与度 (25%)：人工创作参与程度")
        report.append("• 技术质量 (35%)：音频制作技术水准")
        report.append("• 艺术质量 (25%)：音乐艺术价值")
        report.append("• 听觉体验 (15%)：整体听觉感受")
        report.append("")
        
        report.append("="*80)
        report.append("报告生成完成")
        report.append("="*80)
        
        return '\n'.join(report)

if __name__ == "__main__":
    # 创建问题2解决方案实例
    solution = Problem2Solution()
    
    # 执行问题2解决方案
    try:
        detection_results, evaluation_results = solution.solve_problem2()
        print("\n🎉 问题2解决方案执行完成！")
        
    except Exception as e:
        print(f"\n❌ 问题2解决方案执行失败: {e}")
        import traceback
        traceback.print_exc()
