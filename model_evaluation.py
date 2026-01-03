import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Комплексная оценка модели на всём датасете
    """
    
    def __init__(self, model_path: str = 'cardio_classifier.pkl', output_dir: str = 'tests'):
        """
        Инициализация оценщика
        
        Args:
            model_path: путь к обученной модели
            output_dir: папка для сохранения результатов
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Загрузка модели
        print(f"Загрузка модели из {model_path}...")
        model_data = joblib.load(model_path)
        
        self.models = model_data['models']
        self.calibrated_models = model_data['calibrated_models']
        self.scalers = model_data['scalers']
        self.optimal_thresholds = model_data['optimal_thresholds']
        self.diseases = model_data['diseases']
        self.disease_weights = model_data['disease_weights']
        
        print(f"✅ Модель загружена: {len(self.diseases)} заболеваний")
        print(f"📁 Результаты будут сохранены в: {self.output_dir}/\n")
    
    def load_dataset(self, dataset_path: str):
        """
        Загрузка полного датасета
        """
        print(f"Загрузка датасета из {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        # Разделение на признаки и метки
        feature_cols = [col for col in df.columns if col not in self.diseases]
        X = df[feature_cols].copy()
        y = df[self.diseases].copy()
        
        # Кодирование пола
        X['Пол'] = X['Пол'].map({'М': 1, 'Ж': 0})
        
        print(f"✅ Загружено {len(df)} пациентов")
        print(f"   Признаков: {len(feature_cols)}")
        print(f"   Заболеваний: {len(self.diseases)}\n")
        
        return X, y, df
    
    def predict_all(self, X: pd.DataFrame):
        """
        Предсказание вероятностей и классов для всего датасета
        """
        print("Предсказание для всех пациентов...")
        
        # Масштабирование
        X_scaled = pd.DataFrame(
            self.scalers['main'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Вероятности
        probabilities = {}
        predictions = {}
        
        for disease in self.diseases:
            if disease in self.calibrated_models:
                proba = self.calibrated_models[disease].predict_proba(X_scaled)[:, 1]
                probabilities[disease] = proba
                
                # Предсказания с оптимальным порогом
                threshold = self.optimal_thresholds.get(disease, 0.5)
                predictions[disease] = (proba >= threshold).astype(int)
            else:
                probabilities[disease] = np.zeros(len(X))
                predictions[disease] = np.zeros(len(X), dtype=int)
        
        print("✅ Предсказания получены\n")
        
        return pd.DataFrame(probabilities), pd.DataFrame(predictions)
    
    def calculate_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, y_proba: pd.DataFrame):
        """
        Расчёт всех метрик
        """
        print("Расчёт метрик...")
        
        metrics = []
        
        for disease in self.diseases:
            try:
                # AUC-ROC
                auc_roc = roc_auc_score(y_true[disease], y_proba[disease])
                
                # PR-AUC
                pr_auc = average_precision_score(y_true[disease], y_proba[disease])
                
                # F1 на оптимальном пороге
                threshold = self.optimal_thresholds.get(disease, 0.5)
                f1 = f1_score(y_true[disease], y_pred[disease])
                
                # Точность и полнота
                tn, fp, fn, tp = confusion_matrix(y_true[disease], y_pred[disease]).ravel()
                
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                metrics.append({
                    'Заболевание': disease,
                    'AUC-ROC': auc_roc,
                    'PR-AUC': pr_auc,
                    'F1@optimal': f1,
                    'Точность': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'Specificity': specificity,
                    'Порог': threshold
                })
            except Exception as e:
                print(f"⚠️  Ошибка для {disease}: {e}")
        
        metrics_df = pd.DataFrame(metrics)
        print("✅ Метрики рассчитаны\n")
        
        return metrics_df
    
    def plot_metrics_table(self, metrics_df: pd.DataFrame):
        """
        Визуализация таблицы метрик с тепловой картой
        """
        print("Создание таблицы метрик...")
        
        fig, ax = plt.subplots(figsize=(14, len(self.diseases) * 0.5 + 2))
        
        # Выбираем основные метрики для визуализации
        display_cols = ['Заболевание', 'AUC-ROC', 'PR-AUC', 'F1@optimal', 
                       'Precision', 'Recall', 'Specificity']
        display_df = metrics_df[display_cols].copy()
        
        # Создаём цветовую карту для значений метрик
        metric_cols = ['AUC-ROC', 'PR-AUC', 'F1@optimal', 'Precision', 'Recall', 'Specificity']
        
        # Таблица
        ax.axis('tight')
        ax.axis('off')
        
        # Форматирование значений
        cell_text = []
        for _, row in display_df.iterrows():
            formatted_row = [row['Заболевание']]
            for col in metric_cols:
                formatted_row.append(f"{row[col]:.3f}")
            cell_text.append(formatted_row)
        
        # Цвета для ячеек
        cell_colors = []
        for _, row in display_df.iterrows():
            row_colors = ['#f0f0f0']  # Цвет для названия заболевания
            for col in metric_cols:
                value = row[col]
                # Градиент от красного к зелёному
                if value >= 0.9:
                    color = '#2ecc71'  # Отлично
                elif value >= 0.8:
                    color = '#27ae60'  # Хорошо
                elif value >= 0.7:
                    color = '#f39c12'  # Средне
                elif value >= 0.6:
                    color = '#e67e22'  # Плохо
                else:
                    color = '#e74c3c'  # Очень плохо
                row_colors.append(color)
            cell_colors.append(row_colors)
        
        table = ax.table(
            cellText=cell_text,
            colLabels=display_cols,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Стиль заголовков
        for i in range(len(display_cols)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Метрики качества модели по заболеваниям', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Добавляем легенду
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc='#2ecc71', label='Отлично (≥0.9)'),
            plt.Rectangle((0,0),1,1, fc='#27ae60', label='Хорошо (≥0.8)'),
            plt.Rectangle((0,0),1,1, fc='#f39c12', label='Средне (≥0.7)'),
            plt.Rectangle((0,0),1,1, fc='#e67e22', label='Плохо (≥0.6)'),
            plt.Rectangle((0,0),1,1, fc='#e74c3c', label='Очень плохо (<0.6)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05), 
                 ncol=5, frameon=False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_table.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'metrics_table.png'}")
        plt.close()
    
    def plot_confusion_matrices(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, max_diseases: int = 6):
        """
        Матрицы ошибок для нескольких заболеваний
        """
        print("Создание матриц ошибок...")
        
        # Выбираем топ-N заболеваний по распространённости
        disease_counts = y_true.sum().sort_values(ascending=False)
        top_diseases = disease_counts.head(max_diseases).index.tolist()
        
        n_diseases = len(top_diseases)
        n_cols = 3
        n_rows = (n_diseases + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_diseases > 1 else [axes]
        
        for idx, disease in enumerate(top_diseases):
            ax = axes[idx]
            
            cm = confusion_matrix(y_true[disease], y_pred[disease])
            
            # Нормализация для процентов
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Тепловая карта
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar=False, square=True, linewidths=2, linecolor='white')
            
            # Добавляем проценты
            for i in range(2):
                for j in range(2):
                    text = ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1f}%)',
                                 ha='center', va='center', fontsize=9, color='gray')
            
            ax.set_xlabel('Предсказание', fontsize=10, fontweight='bold')
            ax.set_ylabel('Истина', fontsize=10, fontweight='bold')
            ax.set_title(f'{disease}\n(n={y_true[disease].sum()} положительных)', 
                        fontsize=11, fontweight='bold')
            ax.set_xticklabels(['Нет', 'Да'])
            ax.set_yticklabels(['Нет', 'Да'])
        
        # Убираем лишние подграфики
        for idx in range(n_diseases, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Матрицы ошибок для основных заболеваний', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'confusion_matrices.png'}")
        plt.close()
    
    def calculate_integrated_risk(self, y_proba: pd.DataFrame):
        """
        Расчёт интегрального риска для каждого пациента
        """
        risks = []
        
        for idx in range(len(y_proba)):
            # Максимальная взвешенная вероятность
            max_weighted_risk = 0
            
            for disease in self.diseases:
                if disease in y_proba.columns:
                    prob = y_proba.iloc[idx][disease]
                    weight = self.disease_weights[disease]
                    weighted_risk = prob * weight * 100
                    
                    if weighted_risk > max_weighted_risk:
                        max_weighted_risk = weighted_risk
            
            # Учёт множественных заболеваний
            high_risk_count = (y_proba.iloc[idx] >= 0.5).sum()
            moderate_risk_count = ((y_proba.iloc[idx] >= 0.3) & (y_proba.iloc[idx] < 0.5)).sum()
            
            if high_risk_count >= 2:
                max_weighted_risk = min(100, max_weighted_risk + (high_risk_count - 1) * 12)
            
            if moderate_risk_count >= 2:
                max_weighted_risk = min(100, max_weighted_risk + (moderate_risk_count - 1) * 5)
            
            risks.append(max_weighted_risk)
        
        return np.array(risks)
    
    def plot_risk_distribution(self, y_proba: pd.DataFrame, y_true: pd.DataFrame):
        """
        Распределение интегрального риска
        """
        print("Создание распределения рисков...")
        
        # Расчёт рисков
        risks = self.calculate_integrated_risk(y_proba)
        
        # Определяем, есть ли у пациента хотя бы одно заболевание
        has_disease = (y_true.sum(axis=1) > 0).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Общее распределение рисков
        ax = axes[0, 0]
        ax.hist(risks, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(risks.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {risks.mean():.1f}')
        ax.axvline(np.median(risks), color='green', linestyle='--', linewidth=2, label=f'Медиана: {np.median(risks):.1f}')
        ax.set_xlabel('Балл риска', fontsize=12, fontweight='bold')
        ax.set_ylabel('Количество пациентов', fontsize=12, fontweight='bold')
        ax.set_title('Общее распределение интегрального риска', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Распределение по категориям риска
        ax = axes[0, 1]
        risk_categories = pd.cut(risks, 
                                bins=[0, 20, 45, 65, 100], 
                                labels=['Незначительный', 'Средний', 'Высокий', 'Очень высокий'])
        category_counts = risk_categories.value_counts().sort_index()
        
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax.bar(range(len(category_counts)), category_counts.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=0)
        ax.set_ylabel('Количество пациентов', fontsize=12, fontweight='bold')
        ax.set_title('Распределение по категориям риска', fontsize=13, fontweight='bold')
        
        # Добавляем проценты на столбцы
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / len(risks)) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Сравнение рисков: здоровые vs больные
        ax = axes[1, 0]
        
        risks_healthy = risks[has_disease == 0]
        risks_sick = risks[has_disease == 1]
        
        ax.hist(risks_healthy, bins=30, alpha=0.6, color='green', label='Без заболеваний', edgecolor='black')
        ax.hist(risks_sick, bins=30, alpha=0.6, color='red', label='С заболеваниями', edgecolor='black')
        
        ax.axvline(risks_healthy.mean(), color='darkgreen', linestyle='--', linewidth=2, 
                  label=f'Среднее (здоровые): {risks_healthy.mean():.1f}')
        ax.axvline(risks_sick.mean(), color='darkred', linestyle='--', linewidth=2,
                  label=f'Среднее (больные): {risks_sick.mean():.1f}')
        
        ax.set_xlabel('Балл риска', fontsize=12, fontweight='bold')
        ax.set_ylabel('Количество пациентов', fontsize=12, fontweight='bold')
        ax.set_title('Риск: пациенты с заболеваниями vs без', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Box plot по группам
        ax = axes[1, 1]
        
        data_to_plot = [risks_healthy, risks_sick]
        bp = ax.boxplot(data_to_plot, labels=['Без заболеваний', 'С заболеваниями'],
                       patch_artist=True, notch=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        colors_box = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Балл риска', fontsize=12, fontweight='bold')
        ax.set_title('Сравнение распределений (Box Plot)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Добавляем статистику
        stats_text = f"n (здоровые) = {len(risks_healthy)}\n"
        stats_text += f"n (больные) = {len(risks_sick)}\n"
        stats_text += f"Разница средних = {risks_sick.mean() - risks_healthy.mean():.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        plt.suptitle('Анализ интегрального риска', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'risk_distribution.png'}")
        plt.close()
    
    def plot_roc_curves(self, y_true: pd.DataFrame, y_proba: pd.DataFrame, max_diseases: int = 6):
        """
        ROC кривые для нескольких заболеваний
        """
        print("Создание ROC-кривых...")
        
        # Выбираем заболевания с лучшими метриками
        auc_scores = {}
        for disease in self.diseases:
            try:
                auc_scores[disease] = roc_auc_score(y_true[disease], y_proba[disease])
            except:
                auc_scores[disease] = 0
        
        top_diseases = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)[:max_diseases]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_diseases)))
        
        for idx, (disease, auc) in enumerate(top_diseases):
            fpr, tpr, _ = roc_curve(y_true[disease], y_proba[disease])
            ax.plot(fpr, tpr, label=f'{disease} (AUC = {auc:.3f})', 
                   color=colors[idx], linewidth=2)
        
        # Диагональ (случайный классификатор)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Случайный классификатор (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_title('ROC-кривые для заболеваний с лучшими показателями', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'roc_curves.png'}")
        plt.close()
    
    def plot_precision_recall_curves(self, y_true: pd.DataFrame, y_proba: pd.DataFrame, max_diseases: int = 6):
        """
        Precision-Recall кривые
        """
        print("Создание Precision-Recall кривых...")
        
        # Выбираем заболевания с лучшими PR-AUC
        pr_auc_scores = {}
        for disease in self.diseases:
            try:
                pr_auc_scores[disease] = average_precision_score(y_true[disease], y_proba[disease])
            except:
                pr_auc_scores[disease] = 0
        
        top_diseases = sorted(pr_auc_scores.items(), key=lambda x: x[1], reverse=True)[:max_diseases]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_diseases)))
        
        for idx, (disease, pr_auc) in enumerate(top_diseases):
            precision, recall, _ = precision_recall_curve(y_true[disease], y_proba[disease])
            ax.plot(recall, precision, label=f'{disease} (PR-AUC = {pr_auc:.3f})', 
                   color=colors[idx], linewidth=2)
        
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall кривые', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'precision_recall_curves.png'}")
        plt.close()
    
    def plot_calibration_curves(self, y_true: pd.DataFrame, y_proba: pd.DataFrame, max_diseases: int = 4):
        """
        Калибровочные кривые (насколько предсказанные вероятности соответствуют реальным)
        """
        print("Создание калибровочных кривых...")
        
        top_diseases = self.diseases[:max_diseases]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, disease in enumerate(top_diseases):
            ax = axes[idx]
            
            # Разбиваем на бины
            n_bins = 10
            prob_true, prob_pred = [], []
            
            for i in range(n_bins):
                lower = i / n_bins
                upper = (i + 1) / n_bins
                
                mask = (y_proba[disease] >= lower) & (y_proba[disease] < upper)
                if mask.sum() > 0:
                    prob_pred.append((lower + upper) / 2)
                    prob_true.append(y_true[disease][mask].mean())
            
            # График
            ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label=disease)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Идеальная калибровка')
            
            ax.set_xlabel('Предсказанная вероятность', fontsize=11, fontweight='bold')
            ax.set_ylabel('Реальная доля положительных', fontsize=11, fontweight='bold')
            ax.set_title(f'Калибровочная кривая: {disease}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.suptitle('Калибровка модели', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'calibration_curves.png'}")
        plt.close()
    
    def plot_threshold_analysis(self, y_true: pd.DataFrame, y_proba: pd.DataFrame, max_diseases: int = 4):
        """
        Анализ влияния порога на метрики
        """
        print("Создание анализа порогов...")
        
        top_diseases = self.diseases[:max_diseases]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, disease in enumerate(top_diseases):
            ax = axes[idx]
            
            thresholds = np.linspace(0, 1, 100)
            f1_scores = []
            precisions = []
            recalls = []
            
            for thresh in thresholds:
                y_pred_thresh = (y_proba[disease] >= thresh).astype(int)
                
                f1 = f1_score(y_true[disease], y_pred_thresh, zero_division=0)
                f1_scores.append(f1)
                
                tn, fp, fn, tp = confusion_matrix(y_true[disease], y_pred_thresh).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # График
            ax.plot(thresholds, f1_scores, label='F1-score', linewidth=2)
            ax.plot(thresholds, precisions, label='Precision', linewidth=2)
            ax.plot(thresholds, recalls, label='Recall', linewidth=2)
            
            # Оптимальный порог
            optimal_thresh = self.optimal_thresholds.get(disease, 0.5)
            ax.axvline(optimal_thresh, color='red', linestyle='--', linewidth=2, 
                      label=f'Оптимальный порог ({optimal_thresh:.2f})')
            
            ax.set_xlabel('Порог вероятности', fontsize=11, fontweight='bold')
            ax.set_ylabel('Значение метрики', fontsize=11, fontweight='bold')
            ax.set_title(f'{disease}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.suptitle('Влияние порога на метрики качества', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {self.output_dir / 'threshold_analysis.png'}")
        plt.close()
    
    def save_detailed_report(self, metrics_df: pd.DataFrame, y_true: pd.DataFrame, 
                            y_pred: pd.DataFrame, risks: np.ndarray):
        """
        Сохранение детального текстового отчёта
        """
        print("Создание текстового отчёта...")
        
        report_path = self.output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЁТ О ТЕСТИРОВАНИИ МОДЕЛИ\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая информация
            f.write("ОБЩАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"  Всего пациентов: {len(y_true)}\n")
            f.write(f"  Количество заболеваний: {len(self.diseases)}\n")
            f.write(f"  Модель: {self.model_path}\n\n")
            
            # Статистика по заболеваниям
            f.write("=" * 80 + "\n")
            f.write("СТАТИСТИКА ПО ЗАБОЛЕВАНИЯМ:\n")
            f.write("=" * 80 + "\n\n")
            
            for disease in self.diseases:
                n_positive = y_true[disease].sum()
                prevalence = (n_positive / len(y_true)) * 100
                f.write(f"{disease}:\n")
                f.write(f"  Распространённость: {n_positive} ({prevalence:.1f}%)\n")
                
                if disease in metrics_df['Заболевание'].values:
                    row = metrics_df[metrics_df['Заболевание'] == disease].iloc[0]
                    f.write(f"  AUC-ROC: {row['AUC-ROC']:.3f}\n")
                    f.write(f"  PR-AUC: {row['PR-AUC']:.3f}\n")
                    f.write(f"  F1@optimal: {row['F1@optimal']:.3f}\n")
                    f.write(f"  Precision: {row['Precision']:.3f}\n")
                    f.write(f"  Recall: {row['Recall']:.3f}\n")
                    f.write(f"  Specificity: {row['Specificity']:.3f}\n")
                    f.write(f"  Оптимальный порог: {row['Порог']:.3f}\n")
                f.write("\n")
            
            # Средние метрики
            f.write("=" * 80 + "\n")
            f.write("СРЕДНИЕ МЕТРИКИ:\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Средний AUC-ROC: {metrics_df['AUC-ROC'].mean():.3f}\n")
            f.write(f"Средний PR-AUC: {metrics_df['PR-AUC'].mean():.3f}\n")
            f.write(f"Средний F1: {metrics_df['F1@optimal'].mean():.3f}\n")
            f.write(f"Средняя Precision: {metrics_df['Precision'].mean():.3f}\n")
            f.write(f"Средний Recall: {metrics_df['Recall'].mean():.3f}\n")
            f.write(f"Средняя Specificity: {metrics_df['Specificity'].mean():.3f}\n\n")
            
            # Лучшие и худшие заболевания
            f.write("=" * 80 + "\n")
            f.write("ТОП-3 ЛУЧШИХ ПРЕДСКАЗАНИЙ (по AUC-ROC):\n")
            f.write("=" * 80 + "\n\n")
            
            top_3 = metrics_df.nlargest(3, 'AUC-ROC')
            for idx, row in top_3.iterrows():
                f.write(f"{row['Заболевание']}: AUC-ROC = {row['AUC-ROC']:.3f}, F1 = {row['F1@optimal']:.3f}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("ТОП-3 ХУДШИХ ПРЕДСКАЗАНИЙ (по AUC-ROC):\n")
            f.write("=" * 80 + "\n\n")
            
            bottom_3 = metrics_df.nsmallest(3, 'AUC-ROC')
            for idx, row in bottom_3.iterrows():
                f.write(f"{row['Заболевание']}: AUC-ROC = {row['AUC-ROC']:.3f}, F1 = {row['F1@optimal']:.3f}\n")
            
            # Статистика по рискам
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("СТАТИСТИКА ИНТЕГРАЛЬНОГО РИСКА:\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Среднее значение: {risks.mean():.1f}\n")
            f.write(f"Медиана: {np.median(risks):.1f}\n")
            f.write(f"Стандартное отклонение: {risks.std():.1f}\n")
            f.write(f"Минимум: {risks.min():.1f}\n")
            f.write(f"Максимум: {risks.max():.1f}\n\n")
            
            # Распределение по категориям
            risk_categories = pd.cut(risks, 
                                    bins=[0, 20, 45, 65, 100], 
                                    labels=['Незначительный', 'Средний', 'Высокий', 'Очень высокий'])
            category_counts = risk_categories.value_counts().sort_index()
            
            f.write("Распределение по категориям риска:\n")
            for category, count in category_counts.items():
                percentage = (count / len(risks)) * 100
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
            
            # Сравнение здоровые vs больные
            has_disease = (y_true.sum(axis=1) > 0).astype(int)
            risks_healthy = risks[has_disease == 0]
            risks_sick = risks[has_disease == 1]
            
            f.write("\n")
            f.write("Сравнение рисков:\n")
            f.write(f"  Средний риск (без заболеваний): {risks_healthy.mean():.1f}\n")
            f.write(f"  Средний риск (с заболеваниями): {risks_sick.mean():.1f}\n")
            f.write(f"  Разница: {risks_sick.mean() - risks_healthy.mean():.1f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✅ Сохранено: {report_path}")
    
    def run_full_evaluation(self, dataset_path: str):
        """
        Полная оценка модели
        """
        print("\n" + "=" * 80)
        print("НАЧАЛО КОМПЛЕКСНОЙ ОЦЕНКИ МОДЕЛИ")
        print("=" * 80 + "\n")
        
        # 1. Загрузка данных
        X, y_true, df = self.load_dataset(dataset_path)
        
        # 2. Предсказания
        y_proba, y_pred = self.predict_all(X)
        
        # 3. Расчёт метрик
        metrics_df = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Сохраняем метрики в CSV
        metrics_df.to_csv(self.output_dir / 'metrics.csv', index=False, encoding='utf-8')
        print(f"✅ Сохранено: {self.output_dir / 'metrics.csv'}")
        
        # Выводим таблицу метрик
        print("\n" + "=" * 80)
        print("ОСНОВНЫЕ МЕТРИКИ:")
        print("=" * 80)
        print(metrics_df[['Заболевание', 'AUC-ROC', 'PR-AUC', 'F1@optimal']].to_string(index=False))
        print("")
        
        # 4. Визуализации
        print("\n" + "=" * 80)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 80 + "\n")
        
        self.plot_metrics_table(metrics_df)
        self.plot_confusion_matrices(y_true, y_pred)
        
        risks = self.calculate_integrated_risk(y_proba)
        self.plot_risk_distribution(y_proba, y_true)
        
        self.plot_roc_curves(y_true, y_proba)
        self.plot_precision_recall_curves(y_true, y_proba)
        self.plot_calibration_curves(y_true, y_proba)
        self.plot_threshold_analysis(y_true, y_proba)
        
        # 5. Детальный отчёт
        self.save_detailed_report(metrics_df, y_true, y_pred, risks)
        
        # 6. Итоговая сводка
        print("\n" + "=" * 80)
        print("ИТОГОВАЯ СВОДКА")
        print("=" * 80 + "\n")
        
        print(f"✅ Протестировано пациентов: {len(y_true)}")
        print(f"✅ Средний AUC-ROC: {metrics_df['AUC-ROC'].mean():.3f}")
        print(f"✅ Средний F1-score: {metrics_df['F1@optimal'].mean():.3f}")
        print(f"✅ Средний интегральный риск: {risks.mean():.1f}")
        
        # Находим заболевания с отличными результатами (AUC > 0.9)
        excellent = metrics_df[metrics_df['AUC-ROC'] >= 0.9]
        if len(excellent) > 0:
            print(f"\n🏆 Отличные результаты (AUC ≥ 0.9): {len(excellent)} заболеваний")
            for _, row in excellent.iterrows():
                print(f"   • {row['Заболевание']}: {row['AUC-ROC']:.3f}")
        
        # Находим заболевания с плохими результатами (AUC < 0.7)
        poor = metrics_df[metrics_df['AUC-ROC'] < 0.7]
        if len(poor) > 0:
            print(f"\n⚠️  Требуют улучшения (AUC < 0.7): {len(poor)} заболеваний")
            for _, row in poor.iterrows():
                print(f"   • {row['Заболевание']}: {row['AUC-ROC']:.3f}")
        
        print(f"\n📁 Все результаты сохранены в папке: {self.output_dir}/")
        print("\n" + "=" * 80)
        print("ОЦЕНКА ЗАВЕРШЕНА")
        print("=" * 80 + "\n")


def main():
    """
    Главная функция
    """
    # Настройки
    MODEL_PATH = 'cardio_classifier.pkl'
    DATASET_PATH = 'cardio_test_dataset.csv'
    OUTPUT_DIR = 'tests'
    
    # Создаём оценщика и запускаем полную оценку
    evaluator = ModelEvaluator(model_path=MODEL_PATH, output_dir=OUTPUT_DIR)
    evaluator.run_full_evaluation(dataset_path=DATASET_PATH)


if __name__ == "__main__":
    main()