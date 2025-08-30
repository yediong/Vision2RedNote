import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # 改为英文字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir="./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self, training_data, save_path="training_curves.png"):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练损失
        train_losses = training_data['train_losses']
        steps = range(0, len(train_losses) * 10, 10)
        ax1.plot(steps, train_losses, label='Training Loss', color='#FF6B6B', linewidth=2)
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 验证损失
        eval_losses = training_data['eval_losses']
        eval_steps = range(0, len(eval_losses) * 100, 100)  # 假设每100步验证一次
        ax2.plot(eval_steps, eval_losses, label='Validation Loss', color='#4ECDC4', linewidth=2)
        ax2.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Validation Steps')
        ax2.set_ylabel('Loss Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {save_path}")
    
    def plot_learning_rate(self, training_data, save_path="learning_rate.png"):
        """绘制学习率变化"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        learning_rates = training_data['learning_rates']
        steps = range(0, len(learning_rates) * 10, 10)
        
        ax.plot(steps, learning_rates, color='#45B7D1', linewidth=2)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"学习率曲线已保存: {save_path}")
    
    def plot_text_similarity_distribution(self, evaluation_data, save_path="text_similarity.png"):
        """绘制文本相似度分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        text_similarities = np.array(evaluation_data['text_similarities'])
        
        ax.hist(text_similarities, bins=20, alpha=0.7, color='#FF6B6B')
        ax.set_title('Text Similarity Distribution', fontweight='bold')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(text_similarities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(text_similarities):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"文本相似度分布已保存: {save_path}")

    def plot_emoji_density_distribution(self, evaluation_data, save_path="emoji_density.png"):
        """绘制Emoji密度分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        emoji_densities = np.array(evaluation_data['emoji_densities'])
        
        ax.hist(emoji_densities, bins=20, alpha=0.7, color='#4ECDC4')
        ax.set_title('Emoji Density Distribution', fontweight='bold')
        ax.set_xlabel('Emoji Density')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(emoji_densities), color='red', linestyle='--',
                label=f'Mean: {np.mean(emoji_densities):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Emoji密度分布已保存: {save_path}")

    def plot_style_score_distribution(self, evaluation_data, save_path="style_score.png"):
        """绘制风格得分分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        style_scores = np.array(evaluation_data['style_scores'])
        
        ax.hist(style_scores, bins=20, alpha=0.7, color='#45B7D1')
        ax.set_title('Xiaohongshu Style Score Distribution', fontweight='bold')
        ax.set_xlabel('Style Score')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(style_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(style_scores):.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"风格得分分布已保存: {save_path}")

    def plot_length_comparison(self, evaluation_data, save_path="length_comparison.png"):
        """绘制长度对比散点图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        true_lengths = np.array(evaluation_data['true_lengths'])
        pred_lengths = np.array(evaluation_data['pred_lengths'])
        
        ax.scatter(true_lengths, pred_lengths, alpha=0.6, color='#FFA07A')
        ax.plot([0, max(true_lengths)], [0, max(true_lengths)], 'r--', alpha=0.8)
        ax.set_title('Predicted vs True Length', fontweight='bold')
        ax.set_xlabel('True Length')
        ax.set_ylabel('Predicted Length')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"长度对比图已保存: {save_path}")

    def plot_fluency_score_distribution(self, evaluation_data, save_path="fluency_score.png"):
        """绘制流畅性得分分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        fluency_scores = np.array(evaluation_data['fluency_scores'])
        
        ax.hist(fluency_scores, bins=20, alpha=0.7, color='#98D8C8')
        ax.set_title('Fluency Score Distribution', fontweight='bold')
        ax.set_xlabel('Fluency Score')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(fluency_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(fluency_scores):.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"流畅性得分分布已保存: {save_path}")

    def plot_comprehensive_score_distribution(self, evaluation_data, save_path="comprehensive_score.png"):
        """绘制综合得分分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        comprehensive_scores = np.array(evaluation_data['comprehensive_scores'])
        
        ax.hist(comprehensive_scores, bins=20, alpha=0.7, color='#F7DC6F')
        ax.set_title('Comprehensive Score Distribution', fontweight='bold')
        ax.set_xlabel('Comprehensive Score')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(comprehensive_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(comprehensive_scores):.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"综合得分分布已保存: {save_path}")

    def plot_style_vs_fluency(self, evaluation_data, save_path="style_vs_fluency.png"):
        """绘制风格得分vs流畅性散点图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据类型正确
        style_scores = np.array(evaluation_data['style_scores'])
        fluency_scores = np.array(evaluation_data['fluency_scores'])
        
        ax.scatter(style_scores, fluency_scores, alpha=0.6, color='#FF6B6B', s=50)
        ax.set_title('Style Score vs Fluency Score', fontweight='bold')
        ax.set_xlabel('Style Score')
        ax.set_ylabel('Fluency Score')
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(style_scores, fluency_scores, 1)
        p = np.poly1d(z)
        ax.plot(style_scores, p(style_scores), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"风格vs流畅性散点图已保存: {save_path}")

    def plot_metrics_correlation(self, evaluation_data, save_path="metrics_correlation.png"):
        """绘制各指标相关性热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建相关性矩阵，确保数据类型正确
        correlation_data = {
            'Text Similarity': np.array(evaluation_data['text_similarities']),
            'Emoji Density': np.array(evaluation_data['emoji_densities']),
            'Style Score': np.array(evaluation_data['style_scores']),
            'Fluency Score': np.array(evaluation_data['fluency_scores']),
            'Comprehensive Score': np.array(evaluation_data['comprehensive_scores'])
        }
        
        correlation_df = pd.DataFrame(correlation_data)
        correlation_matrix = correlation_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax)
        ax.set_title('Metrics Correlation Heatmap', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"指标相关性热力图已保存: {save_path}")

    def plot_main_metrics_summary(self, evaluation_data, save_path="main_metrics_summary.png"):
        """绘制主要指标总结"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 确保数据类型正确
        text_similarities = np.array(evaluation_data['text_similarities'])
        emoji_densities = np.array(evaluation_data['emoji_densities'])
        style_scores = np.array(evaluation_data['style_scores'])
        fluency_scores = np.array(evaluation_data['fluency_scores'])
        
        metrics_names = ['Text Similarity', 'Emoji Density', 'Style Score', 'Fluency Score']
        metrics_values = [
            np.mean(text_similarities),
            np.mean(emoji_densities) * 100,
            np.mean(style_scores),
            np.mean(fluency_scores)
        ]
        
        bars = ax.bar(metrics_names, metrics_values, 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F'])
        ax.set_title('Main Metrics Summary', fontweight='bold')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"主要指标总结已保存: {save_path}")

    def plot_length_accuracy(self, evaluation_data, save_path="length_accuracy.png"):
        """绘制长度预测准确性"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 确保数据类型正确
        true_lengths = np.array(evaluation_data['true_lengths'])
        length_mae = evaluation_data['length_mae']
        length_accuracy = 1 - length_mae / np.mean(true_lengths)
        
        ax.bar(['Length Accuracy'], [length_accuracy], color='#98D8C8')
        ax.set_title('Length Prediction Accuracy', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.text(0, length_accuracy + 0.01, f'{length_accuracy:.3f}', 
            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"长度预测准确性已保存: {save_path}")
    
    def plot_diversity_metrics(self, evaluation_data, save_path="diversity_metrics.png"):
        """绘制多样性指标"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        diversity_data = evaluation_data['diversity_metrics']
        diversity_names = list(diversity_data.keys())
        diversity_values = list(diversity_data.values())
        
        bars = ax.bar(diversity_names, diversity_values, color=['#E74C3C', '#3498DB', '#2ECC71'])
        ax.set_title('Diversity Metrics', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, diversity_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"多样性指标已保存: {save_path}")    
    
    def plot_quality_distribution(self, evaluation_data, save_path="quality_distribution.png"):
        """绘制质量分布饼图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 确保comprehensive_scores是numpy数组
        comprehensive_scores = np.array(evaluation_data['comprehensive_scores'])
        excellent = np.sum(comprehensive_scores >= 80)
        good = np.sum((comprehensive_scores >= 60) & (comprehensive_scores < 80))
        fair = np.sum((comprehensive_scores >= 40) & (comprehensive_scores < 60))
        poor = np.sum(comprehensive_scores < 40)
        
        sizes = [excellent, good, fair, poor]
        labels = ['Excellent (≥80)', 'Good (60-79)', 'Fair (40-59)', 'Poor (<40)']
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Text Quality Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"质量分布饼图已保存: {save_path}")    
    
    def generate_all_plots(self, training_data_path=None, evaluation_data_path=None):
        """生成所有图表"""
        print("开始生成可视化图表...")
        
        # 生成训练相关图表
        if training_data_path and os.path.exists(training_data_path):
            print(f"加载训练数据: {training_data_path}")
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            self.plot_training_curves(training_data)
            self.plot_learning_rate(training_data)
        
        # 生成评估相关图表
        if evaluation_data_path and os.path.exists(evaluation_data_path):
            print(f"加载评估数据: {evaluation_data_path}")
            with open(evaluation_data_path, 'r', encoding='utf-8') as f:
                evaluation_data = json.load(f)
            
            self.plot_text_similarity_distribution(evaluation_data)
            self.plot_emoji_density_distribution(evaluation_data)
            self.plot_style_score_distribution(evaluation_data)
            self.plot_length_comparison(evaluation_data)
            self.plot_fluency_score_distribution(evaluation_data)
            self.plot_comprehensive_score_distribution(evaluation_data)
            self.plot_diversity_metrics(evaluation_data)
            self.plot_style_vs_fluency(evaluation_data)
            self.plot_metrics_correlation(evaluation_data)
            self.plot_quality_distribution(evaluation_data)
            self.plot_main_metrics_summary(evaluation_data)
            self.plot_length_accuracy(evaluation_data)
        
        print(f"所有图表已生成完成！保存在: {self.output_dir}")

def main():
    """主函数"""
    # 设置数据文件路径
    training_data_path = "./qwen2-finetuned0824_3/training_data.json"  # 训练数据路径
    evaluation_data_path = "./evaluation_results/evaluation_data.json"  # 评估数据路径
    
    # 创建可视化器
    visualizer = Visualizer()
    
    # 检查文件是否存在
    if not os.path.exists(training_data_path):
        print(f"警告: 训练数据文件不存在: {training_data_path}")
        training_data_path = None
    
    if not os.path.exists(evaluation_data_path):
        print(f"警告: 评估数据文件不存在: {evaluation_data_path}")
        evaluation_data_path = None
    
    if training_data_path is None and evaluation_data_path is None:
        print("错误: 没有找到任何数据文件！")
        print("请先运行 train.py 和 evaluate.py 生成数据文件")
        return
    
    # 生成所有图表
    visualizer.generate_all_plots(training_data_path, evaluation_data_path)

if __name__ == "__main__":
    main()