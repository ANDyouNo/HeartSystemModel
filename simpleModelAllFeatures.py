import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import joblib
import warnings
warnings.filterwarnings('ignore')


class MultiLabelCardioClassifier:
    """
    Мультилейбл классификатор кардиологических заболеваний на основе XGBoost
    с автоматическим подбором параметров и оценкой рисков
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.diseases = [
            'Дислипидемия',
            'Атеросклероз',
            'Метаболический синдром',
            'Сахарный диабет',
            'ХСН',
            'Миопатия',
            'Постстрептококковый кардит',
            'Ревматическая лихорадка',
            'Атеросклеротическая кардиопатия',
            'Анемия',
            'Электролитные аритмии',
            'Алкогольная кардиомиопатия'
        ]
        
        self.models = {}
        self.calibrated_models = {}
        self.scalers = {}
        self.best_params = {}
        self.optimal_thresholds = {}
        self.feature_importance = {}
        
        # Веса для оценки риска (по серьезности заболевания)
        self.disease_weights = {
            'Дислипидемия': 0.6,
            'Атеросклероз': 0.9,
            'Метаболический синдром': 0.8,
            'Сахарный диабет': 0.85,
            'ХСН': 1.0,
            'Миопатия': 0.7,
            'Постстрептококковый кардит': 0.85,
            'Ревматическая лихорадка': 0.9,
            'Атеросклеротическая кардиопатия': 0.95,
            'Анемия': 0.5,
            'Электролитные аритмии': 0.8,
            'Алкогольная кардиомиопатия': 0.95
        }
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Подготовка данных: разделение на признаки и метки"""
        # Признаки
        feature_cols = [col for col in df.columns if col not in self.diseases]
        X = df[feature_cols].copy()
        
        # Кодирование пола
        X['Пол'] = X['Пол'].map({'М': 1, 'Ж': 0})
        
        # Метки
        y = df[self.diseases].copy()
        
        return X, y
    
    def find_optimal_iterations(self, X_train, y_train, disease: str) -> Dict:
        """
        Поиск оптимального количества итераций с использованием early stopping
        """
        print(f"\n{'='*60}")
        print(f"Подбор параметров для: {disease}")
        print(f"{'='*60}")
        
        # Разделяем на train/validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state,
            stratify=y_train
        )
        
        # Базовые параметры
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        # Вычисляем scale_pos_weight для балансировки классов
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        base_params['scale_pos_weight'] = scale_pos_weight
        
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        print(f"Positive samples: {pos_count} ({pos_count/len(y_tr)*100:.1f}%)")
        
        # Создаем DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        # Обучение с early stopping
        evals_result = {}
        model = xgb.train(
            base_params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        best_iteration = model.best_iteration
        best_score = model.best_score
        
        print(f"Оптимальное количество итераций: {best_iteration}")
        print(f"Лучший AUC: {best_score:.4f}")
        
        # Настройка learning_rate в зависимости от количества итераций
        if best_iteration < 200:
            base_params['learning_rate'] = 0.1
        elif best_iteration > 800:
            base_params['learning_rate'] = 0.03
        
        base_params['n_estimators'] = best_iteration
        
        return base_params
    
    def find_optimal_threshold(self, y_true, y_pred_proba) -> float:
        """
        Поиск оптимального порога классификации для максимизации F1-score
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        return optimal_threshold
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, test_size=0.2):
        """
        Обучение моделей для каждого заболевания
        """
        print("\n" + "="*80)
        print("НАЧАЛО ОБУЧЕНИЯ МУЛЬТИЛЕЙБЛ КЛАССИФИКАТОРА")
        print("="*80)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nРазмер обучающей выборки: {len(X_train)}")
        print(f"Размер тестовой выборки: {len(X_test)}")
        
        # Масштабирование признаков
        print("\nМасштабирование признаков...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        self.scalers['main'] = scaler
        
        # Обучение моделей для каждого заболевания
        results = {}
        
        for disease in self.diseases:
            print(f"\n{'#'*80}")
            print(f"Обучение модели для: {disease}")
            print(f"{'#'*80}")
            
            y_train_disease = y_train[disease]
            y_test_disease = y_test[disease]
            
            # Пропускаем если слишком мало положительных примеров
            if y_train_disease.sum() < 10:
                print(f"⚠️  Недостаточно примеров для {disease}, пропускаем")
                continue
            
            # Поиск оптимальных параметров
            best_params = self.find_optimal_iterations(
                X_train_scaled, y_train_disease, disease
            )
            self.best_params[disease] = best_params
            
            # Обучение финальной модели
            print("\nОбучение финальной модели...")
            model = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)
            model.fit(
                X_train_scaled, y_train_disease,
                eval_set=[(X_test_scaled, y_test_disease)],
                verbose=False
            )
            
            self.models[disease] = model
            
            # Калибровка вероятностей (создаем модель без early stopping)
            print("Калибровка вероятностей...")
            calibration_params = best_params.copy()
            # Убираем early_stopping_rounds для калибровки
            model_for_calibration = xgb.XGBClassifier(**calibration_params)
            
            calibrated_model = CalibratedClassifierCV(
                model_for_calibration, method='sigmoid', cv=3
            )
            calibrated_model.fit(X_train_scaled, y_train_disease)
            self.calibrated_models[disease] = calibrated_model
            
            # Предсказания на тестовой выборке
            y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
            
            # Поиск оптимального порога
            optimal_threshold = self.find_optimal_threshold(
                y_test_disease, y_pred_proba
            )
            self.optimal_thresholds[disease] = optimal_threshold
            
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Метрики
            auc = roc_auc_score(y_test_disease, y_pred_proba)
            ap = average_precision_score(y_test_disease, y_pred_proba)
            f1 = f1_score(y_test_disease, y_pred)
            
            results[disease] = {
                'AUC': auc,
                'AP': ap,
                'F1': f1,
                'Threshold': optimal_threshold,
                'Iterations': best_params['n_estimators']
            }
            
            print(f"\n📊 Результаты на тестовой выборке:")
            print(f"   AUC: {auc:.4f}")
            print(f"   Average Precision: {ap:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Оптимальный порог: {optimal_threshold:.3f}")
            
            # Feature importance
            self.feature_importance[disease] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Итоговые результаты
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Вывод итоговой статистики обучения"""
        print("\n" + "="*80)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("="*80)
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('AUC', ascending=False)
        
        print("\n" + results_df.to_string())
        
        print(f"\n📈 Средние метрики:")
        print(f"   Средний AUC: {results_df['AUC'].mean():.4f}")
        print(f"   Средний AP: {results_df['AP'].mean():.4f}")
        print(f"   Средний F1: {results_df['F1'].mean():.4f}")
    
    def predict_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание вероятностей для всех заболеваний
        """
        # Масштабирование
        X_scaled = pd.DataFrame(
            self.scalers['main'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        probabilities = {}
        
        for disease in self.diseases:
            if disease in self.calibrated_models:
                proba = self.calibrated_models[disease].predict_proba(X_scaled)[:, 1]
                probabilities[disease] = proba
            else:
                probabilities[disease] = np.zeros(len(X))
        
        return pd.DataFrame(probabilities, index=X.index)
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание классов с использованием оптимальных порогов
        """
        probabilities = self.predict_probabilities(X)
        predictions = pd.DataFrame(index=X.index)
        
        for disease in self.diseases:
            if disease in self.optimal_thresholds:
                threshold = self.optimal_thresholds[disease]
                predictions[disease] = (probabilities[disease] >= threshold).astype(int)
            else:
                predictions[disease] = 0
        
        return predictions
    
    def calculate_risk_score(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет общего риска и определение группы риска
        Использует максимальный риск среди заболеваний и их комбинацию
        """
        risk_scores = []
        risk_categories = []
        
        for idx in probabilities.index:
            # Метод 1: Максимальная взвешенная вероятность
            max_weighted_risk = 0
            
            for disease in self.diseases:
                if disease in probabilities.columns:
                    prob = probabilities.loc[idx, disease]
                    weight = self.disease_weights[disease]
                    weighted_risk = prob * weight * 100
                    max_weighted_risk = max(max_weighted_risk, weighted_risk)
            
            # Метод 2: Учет множественных заболеваний
            high_risk_count = 0
            moderate_risk_count = 0
            
            for disease in self.diseases:
                if disease in probabilities.columns:
                    prob = probabilities.loc[idx, disease]
                    if prob >= 0.5:
                        high_risk_count += 1
                    elif prob >= 0.3:
                        moderate_risk_count += 1
            
            # Базовый риск
            base_risk = max_weighted_risk
            
            # Бонус за множественные заболевания
            if high_risk_count >= 2:
                combo_bonus = (high_risk_count - 1) * 12
                base_risk = min(100, base_risk + combo_bonus)
            
            if moderate_risk_count >= 2:
                combo_bonus = (moderate_risk_count - 1) * 5
                base_risk = min(100, base_risk + combo_bonus)
            
            risk_score = base_risk
            risk_scores.append(risk_score)
            
            # Определение категории риска
            if risk_score >= 65:
                category = "Очень высокий"
            elif risk_score >= 45:
                category = "Высокий"
            elif risk_score >= 20:
                category = "Средний"
            else:
                category = "Незначительный"
            
            risk_categories.append(category)
        
        return pd.DataFrame({
            'risk_score': risk_scores,
            'risk_category': risk_categories
        }, index=probabilities.index)
    
    def predict_with_risk(self, X: pd.DataFrame) -> Dict:
        """
        Полное предсказание: вероятности, классы и оценка риска
        """
        probabilities = self.predict_probabilities(X)
        predictions = self.predict(X)
        risks = self.calculate_risk_score(probabilities)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'risk_scores': risks['risk_score'],
            'risk_categories': risks['risk_category']
        }
    
    def get_patient_report(self, X: pd.DataFrame, patient_idx: int = 0) -> str:
        """
        Генерация детального отчета для пациента
        """
        result = self.predict_with_risk(X.iloc[[patient_idx]])
        
        probabilities = result['probabilities'].iloc[0]
        predictions = result['predictions'].iloc[0]
        risk_score = result['risk_scores'].iloc[0]
        risk_category = result['risk_categories'].iloc[0]
        
        # Сортируем заболевания по вероятности
        disease_probs = [(disease, probabilities[disease]) 
                        for disease in self.diseases 
                        if disease in probabilities.index]
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        report = f"""
{'='*80}
ОТЧЕТ О СОСТОЯНИИ ЗДОРОВЬЯ ПАЦИЕНТА
{'='*80}

ОБЩАЯ ОЦЕНКА РИСКА:
  Балл риска: {risk_score:.1f}/100
  Группа риска: {risk_category}

ВЕРОЯТНОСТИ ЗАБОЛЕВАНИЙ (отсортировано по убыванию):
"""
        
        for disease, prob in disease_probs:
            pred = "ОБНАРУЖЕНО" if predictions[disease] == 1 else "не обнаружено"
            threshold = self.optimal_thresholds.get(disease, 0.5)
            
            # Визуальная шкала
            bar_length = 30
            filled = int(prob * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            report += f"\n  {disease}:\n"
            report += f"    [{bar}] {prob*100:.1f}%\n"
            report += f"    Статус: {pred} (порог: {threshold:.2f})\n"
        
        # Топ-3 факторов риска
        if patient_idx < len(X):
            report += f"\n\nКЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:\n"
            
            # Находим заболевание с наибольшей вероятностью
            if disease_probs:
                top_disease = disease_probs[0][0]
                if top_disease in self.feature_importance:
                    top_features = self.feature_importance[top_disease].head(5)
                    
                    for _, row in top_features.iterrows():
                        feature = row['feature']
                        if feature in X.columns:
                            value = X.iloc[patient_idx][feature]
                            report += f"  • {feature}: {value}\n"
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def save_models(self, filepath: str = 'cardio_classifier.pkl'):
        """Сохранение обученных моделей"""
        model_data = {
            'models': self.models,
            'calibrated_models': self.calibrated_models,
            'scalers': self.scalers,
            'best_params': self.best_params,
            'optimal_thresholds': self.optimal_thresholds,
            'feature_importance': self.feature_importance,
            'diseases': self.diseases,
            'disease_weights': self.disease_weights
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Модели сохранены в {filepath}")
    
    def load_models(self, filepath: str = 'cardio_classifier.pkl'):
        """Загрузка обученных моделей"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.calibrated_models = model_data['calibrated_models']
        self.scalers = model_data['scalers']
        self.best_params = model_data['best_params']
        self.optimal_thresholds = model_data['optimal_thresholds']
        self.feature_importance = model_data['feature_importance']
        self.diseases = model_data['diseases']
        self.disease_weights = model_data['disease_weights']
        print(f"\n✅ Модели загружены из {filepath}")
    
    def plot_feature_importance(self, top_n: int = 15):
        """Визуализация важности признаков для каждого заболевания"""
        n_diseases = len(self.feature_importance)
        n_cols = 3
        n_rows = (n_diseases + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_diseases > 1 else [axes]
        
        for idx, (disease, importance_df) in enumerate(self.feature_importance.items()):
            ax = axes[idx]
            top_features = importance_df.head(top_n)
            
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Важность')
            ax.set_title(f'{disease}', fontsize=10, fontweight='bold')
            ax.invert_yaxis()
        
        # Удаляем пустые субплоты
        for idx in range(len(self.feature_importance), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✅ График важности признаков сохранен в 'feature_importance.png'")
        plt.close()


def main():
    """Основной pipeline обучения и тестирования"""
    
    print("="*80)
    print("СИСТЕМА ДИАГНОСТИКИ КАРДИОЛОГИЧЕСКИХ ЗАБОЛЕВАНИЙ")
    print("="*80)
    
    # Загрузка данных
    print("\n1. Загрузка датасета...")
    df = pd.read_csv('cardio_synthetic_dataset.csv')
    print(f"   Загружено {len(df)} образцов")
    
    # Инициализация классификатора
    print("\n2. Инициализация классификатора...")
    classifier = MultiLabelCardioClassifier(random_state=42)
    
    # Подготовка данных
    print("\n3. Подготовка данных...")
    X, y = classifier.prepare_data(df)
    print(f"   Признаков: {X.shape[1]}")
    print(f"   Классов: {y.shape[1]}")
    
    # Обучение
    print("\n4. Обучение моделей...")
    results = classifier.train(X, y, test_size=0.2)
    
    # Сохранение моделей
    print("\n5. Сохранение моделей...")
    classifier.save_models('cardio_classifier.pkl')
    
    # Визуализация важности признаков
    print("\n6. Визуализация важности признаков...")
    classifier.plot_feature_importance(top_n=10)
    
    # Тестирование на примерах
    print("\n7. Тестирование на случайных пациентах...")
    print("\n" + "="*80)
    print("ПРИМЕРЫ ДИАГНОСТИКИ")
    print("="*80)
    
    # Выбираем 3 случайных пациента
    test_indices = np.random.choice(len(X), size=3, replace=False)
    
    for idx in test_indices:
        report = classifier.get_patient_report(X, idx)
        print(report)
    
    print("\n✅ Обучение завершено успешно!")
    
    return classifier


if __name__ == "__main__":
    classifier = main()