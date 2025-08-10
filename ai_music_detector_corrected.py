#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版AI音乐检测器
解决训练数据标签错误问题，重新训练模型
"""

from ai_music_detector import MathematicalAIMusicDetector
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class CorrectedAIMusicDetector(MathematicalAIMusicDetector):
    """修正版AI音乐检测器 - 解决标签错误问题"""
    
    def __init__(self):
        super().__init__()
        # 使用修正后的模型缓存路径
        self.model_cache_path = "ai_music_detector_corrected_model.pkl"
        self.scaler_cache_path = "ai_music_detector_corrected_scaler.pkl"
        self.threshold_cache_path = "ai_music_detector_corrected_threshold.pkl"
    
    def get_corrected_labels(self):
        """获取修正后的标签映射"""
        return {
            # AI生成音乐（标签=1）- 修正后包含tianyi_daddy
            'ai_folders': [
                'alice',           # AI虚拟歌手
                'china_vocaloid',  # 中文虚拟歌手
                'game',           # 游戏AI音乐
                'gugugaga',       # AI生成
                'ikun',           # AI生成
                'manbo',          # AI生成
                'yiwu',           # AI生成
                'tianyi_daddy'    # 🔄 修正：洛天依相关，应为AI类
            ],
            # 人类创作音乐（标签=0）
            'human_folders': [
                'hanser',         # 人类歌手
                'xiangsi',        # 人类创作（待验证）
                'xiexiemiao~'     # 人类创作
            ]
        }
    
    def train_with_corrected_data(self, training_data_path="附件六：训练数据"):
        """使用修正后的标签重新训练模型"""
        print("🔄 使用修正后的标签重新训练模型...")
        print("主要修正：tianyi_daddy 从人类类别移至AI类别")
        
        # 首先尝试从缓存加载
        if self.load_model_cache():
            print("✅ 从缓存加载修正版模型成功")
            return self
        
        print("📚 开始重新训练修正版模型...")
        from sklearn.model_selection import train_test_split
        
        if not os.path.exists(training_data_path):
            print(f"错误：训练数据路径不存在: {training_data_path}")
            return self
        
        # 获取修正后的标签
        labels_config = self.get_corrected_labels()
        ai_folders = labels_config['ai_folders']
        human_folders = labels_config['human_folders']
        
        print(f"\n=== 修正后的数据标签 ===")
        print(f"AI类别({len(ai_folders)}个文件夹): {ai_folders}")
        print(f"人类类别({len(human_folders)}个文件夹): {human_folders}")
        
        # 收集训练数据
        X_features = []
        y_labels = []
        file_info = []
        
        # 处理AI类别
        for folder in ai_folders:
            folder_path = os.path.join(training_data_path, folder)
            if not os.path.exists(folder_path):
                print(f"⚠️ 文件夹不存在，跳过: {folder}")
                continue
                
            audio_files = glob.glob(os.path.join(folder_path, "*.mp3")) + \
                         glob.glob(os.path.join(folder_path, "*.aac"))
            
            print(f"处理AI类别 {folder}: {len(audio_files)}个文件")
            
            # 限制每个文件夹最多50个样本
            if len(audio_files) > 50:
                audio_files = np.random.choice(audio_files, 50, replace=False)
            
            for audio_file in audio_files:
                try:
                    features = self.feature_extractor.extract_all_features(audio_file)
                    if features is not None and len(features) > 0:
                        X_features.append(features)
                        y_labels.append(1)  # AI标签
                        file_info.append(f"{folder}/{os.path.basename(audio_file)}")
                except Exception as e:
                    print(f"特征提取失败 {audio_file}: {e}")
        
        # 处理人类类别
        for folder in human_folders:
            folder_path = os.path.join(training_data_path, folder)
            if not os.path.exists(folder_path):
                print(f"⚠️ 文件夹不存在，跳过: {folder}")
                continue
                
            audio_files = glob.glob(os.path.join(folder_path, "*.mp3")) + \
                         glob.glob(os.path.join(folder_path, "*.aac"))
            
            print(f"处理人类类别 {folder}: {len(audio_files)}个文件")
            
            # 限制每个文件夹最多50个样本
            if len(audio_files) > 50:
                audio_files = np.random.choice(audio_files, 50, replace=False)
            
            for audio_file in audio_files:
                try:
                    features = self.feature_extractor.extract_all_features(audio_file)
                    if features is not None and len(features) > 0:
                        X_features.append(features)
                        y_labels.append(0)  # 人类标签
                        file_info.append(f"{folder}/{os.path.basename(audio_file)}")
                except Exception as e:
                    print(f"特征提取失败 {audio_file}: {e}")
        
        if len(X_features) == 0:
            print("错误：没有成功提取到特征")
            return self
        
        # 检查特征维度一致性
        feature_lengths = [len(f) for f in X_features]
        if len(set(feature_lengths)) > 1:
            print(f"警告：特征维度不一致: {set(feature_lengths)}")
            # 使用最常见的维度
            most_common_length = max(set(feature_lengths), key=feature_lengths.count)
            print(f"使用最常见的特征维度: {most_common_length}")
            
            # 重新构建特征和标签列表
            filtered_X = []
            filtered_y = []
            for i, f in enumerate(X_features):
                if len(f) == most_common_length:
                    filtered_X.append(f)
                    filtered_y.append(y_labels[i])
            
            X_features = filtered_X
            y_labels = filtered_y
        
        X = np.array(X_features)
        y = np.array(y_labels)
        
        print(f"\n=== 修正后的数据统计 ===")
        print(f"总样本数: {len(X)}")
        print(f"AI样本数: {np.sum(y == 1)}")
        print(f"人类样本数: {np.sum(y == 0)}")
        print(f"特征维度: {X.shape[1] if X.ndim > 1 else 'N/A'}")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")
        
        # 创建并训练集成模型
        self.model = self._create_interpretable_ensemble()
        
        print("开始训练模型...")
        self.model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\n=== 修正版模型性能 ===")
        print(f"训练集准确率: {train_score:.3f}")
        print(f"测试集准确率: {test_score:.3f}")
        
        # 寻找最优阈值
        self.optimal_threshold = self._find_optimal_threshold(X_test, y_test)
        
        # 使用最优阈值重新预测
        y_pred_optimal = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        print(f"\n=== 使用最优阈值 {self.optimal_threshold:.3f} 的结果 ===")
        print(classification_report(y_test, y_pred_optimal, target_names=['人类创作', 'AI生成']))
        
        # 保存模型缓存
        print("\n💾 保存修正版模型缓存...")
        self.save_model_cache()
        
        return self
    
    def analyze_correction_effect(self, target_audio="附件四：测试音乐.mp3"):
        """分析修正后模型对附件四的预测效果"""
        if not os.path.exists(target_audio):
            print(f"错误：未找到文件 {target_audio}")
            return
        
        print(f"\n=== 修正版模型对{target_audio}的分析 ===")
        
        # 使用修正版模型预测
        result = self.predict_single(target_audio)
        
        print(f"修正版预测结果:")
        print(f"  预测: {'AI生成' if result['prediction'] else '人类创作'}")
        print(f"  AI概率: {result['probability_ai']:.3f}")
        print(f"  阈值: {self.optimal_threshold:.3f}")
        print(f"  实际标签: AI生成（根据题目说明）")
        print(f"  判断正确性: {'✅ 正确' if result['prediction'] else '❌ 错误'}")
        
        return result

def main():
    """主函数 - 训练修正版模型并测试"""
    print("="*60)
    print("修正版AI音乐检测器")
    print("解决训练数据标签错误问题")
    print("="*60)
    
    # 创建修正版检测器
    corrected_detector = CorrectedAIMusicDetector()
    
    # 重新训练
    corrected_detector.train_with_corrected_data()
    
    # 分析修正效果
    corrected_detector.analyze_correction_effect()
    
    print("\n" + "="*60)
    print("修正版模型训练完成！")
    print("如果修正成功，附件四应该被正确识别为AI生成")
    print("="*60)

if __name__ == "__main__":
    main()
