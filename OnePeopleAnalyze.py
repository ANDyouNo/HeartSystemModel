import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional
import sys


class PatientAnalyzer:
    """
    Анализатор для диагностики пациентов с использованием обученной модели
    """
    
    def __init__(self, model_path: str = 'cardio_classifier.pkl'):
        """
        Инициализация анализатора
        
        Args:
            model_path: путь к файлу с сохраненной моделью
        """
        self.model_path = model_path
        self.models = None
        self.calibrated_models = None
        self.scalers = None
        self.optimal_thresholds = None
        self.feature_importance = None
        self.diseases = None
        self.disease_weights = None
        
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        print(f"Загрузка модели из {self.model_path}...")
        
        try:
            model_data = joblib.load(self.model_path)
            
            self.models = model_data['models']
            self.calibrated_models = model_data['calibrated_models']
            self.scalers = model_data['scalers']
            self.optimal_thresholds = model_data['optimal_thresholds']
            self.feature_importance = model_data['feature_importance']
            self.diseases = model_data['diseases']
            self.disease_weights = model_data['disease_weights']
            
            print(f"✅ Модель успешно загружена")
            print(f"   Количество заболеваний: {len(self.diseases)}")
            print(f"   Заболевания: {', '.join(self.diseases)}")
            
        except FileNotFoundError:
            print(f"❌ Ошибка: файл {self.model_path} не найден!")
            print("   Сначала обучите модель, запустив основной скрипт обучения.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            sys.exit(1)
    
    def load_patient_data(self, dataset_path: str, patient_id: int) -> pd.DataFrame:
        """
        Загрузка данных конкретного пациента из датасета
        
        Args:
            dataset_path: путь к файлу с датасетом
            patient_id: номер пациента (индекс в датасете)
            
        Returns:
            DataFrame с данными пациента
        """
        print(f"\nЗагрузка данных пациента #{patient_id}...")
        
        try:
            df = pd.read_csv(dataset_path)
            
            if patient_id < 0 or patient_id >= len(df):
                print(f"❌ Ошибка: пациент #{patient_id} не найден в датасете!")
                print(f"   Доступные индексы: 0-{len(df)-1}")
                sys.exit(1)
            
            # Получаем данные пациента
            patient_data = df.iloc[[patient_id]].copy()
            
            # Разделяем на признаки и истинные метки
            feature_cols = [col for col in df.columns if col not in self.diseases]
            X = patient_data[feature_cols].copy()
            
            # Кодируем пол
            X['Пол'] = X['Пол'].map({'М': 1, 'Ж': 0})
            
            # Истинные метки (если есть)
            true_labels = None
            if all(disease in patient_data.columns for disease in self.diseases):
                true_labels = patient_data[self.diseases]
            
            print(f"✅ Данные пациента загружены")
            
            return X, true_labels, patient_data
            
        except FileNotFoundError:
            print(f"❌ Ошибка: файл {dataset_path} не найден!")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Ошибка при загрузке данных: {e}")
            sys.exit(1)
    
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
    
    def calculate_risk_score(self, probabilities: pd.DataFrame) -> Dict:
        """
        Расчет общего риска и определение группы риска
        Использует максимальный риск среди заболеваний и их комбинацию
        """
        # Метод 1: Максимальная взвешенная вероятность
        max_weighted_risk = 0
        max_risk_disease = None
        
        for disease in self.diseases:
            if disease in probabilities.columns:
                prob = probabilities.iloc[0][disease]
                weight = self.disease_weights[disease]
                weighted_risk = prob * weight * 100
                
                if weighted_risk > max_weighted_risk:
                    max_weighted_risk = weighted_risk
                    max_risk_disease = disease
        
        # Метод 2: Учет множественных заболеваний (бонус за комбинации)
        high_risk_diseases = []
        moderate_risk_diseases = []
        
        for disease in self.diseases:
            if disease in probabilities.columns:
                prob = probabilities.iloc[0][disease]
                if prob >= 0.5:  # Высокая вероятность
                    high_risk_diseases.append((disease, prob))
                elif prob >= 0.3:  # Средняя вероятность
                    moderate_risk_diseases.append((disease, prob))
        
        # Базовый риск = максимальный риск
        base_risk = max_weighted_risk
        
        # Добавляем бонус за множественные заболевания
        if len(high_risk_diseases) >= 2:
            # За каждое дополнительное заболевание добавляем 10-15% к риску
            combo_bonus = (len(high_risk_diseases) - 1) * 12
            base_risk = min(100, base_risk + combo_bonus)
        
        if len(moderate_risk_diseases) >= 2:
            # За умеренные риски добавляем меньше
            combo_bonus = (len(moderate_risk_diseases) - 1) * 5
            base_risk = min(100, base_risk + combo_bonus)
        
        risk_score = base_risk
        
        # Определение категории риска с более чувствительными порогами
        if risk_score >= 65:
            category = "Очень высокий"
            emoji = "🔴"
        elif risk_score >= 45:
            category = "Высокий"
            emoji = "🟠"
        elif risk_score >= 20:
            category = "Средний"
            emoji = "🟡"
        else:
            category = "Незначительный"
            emoji = "🟢"
        
        return {
            'risk_score': risk_score,
            'risk_category': category,
            'risk_emoji': emoji,
            'max_risk_disease': max_risk_disease,
            'n_high_risk': len(high_risk_diseases),
            'n_moderate_risk': len(moderate_risk_diseases)
        }
    
    def generate_patient_report(
        self, 
        X: pd.DataFrame, 
        patient_data: pd.DataFrame,
        true_labels: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Генерация детального отчета для пациента
        """
        # Предсказания
        probabilities = self.predict_probabilities(X)
        predictions = self.predict(X)
        risk_info = self.calculate_risk_score(probabilities)
        
        probabilities_series = probabilities.iloc[0]
        predictions_series = predictions.iloc[0]
        risk_score = risk_info['risk_score']
        risk_category = risk_info['risk_category']
        risk_emoji = risk_info['risk_emoji']
        
        # Демографические данные
        age = patient_data.iloc[0]['Возраст']
        sex = patient_data.iloc[0]['Пол']
        bmi = patient_data.iloc[0]['ИМТ']
        
        # Сортируем заболевания по вероятности
        disease_probs = [(disease, probabilities_series[disease]) 
                        for disease in self.diseases 
                        if disease in probabilities_series.index]
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Формируем отчет
        report = f"""
{'='*80}
ОТЧЕТ О СОСТОЯНИИ ЗДОРОВЬЯ ПАЦИЕНТА
{'='*80}

ДЕМОГРАФИЧЕСКИЕ ДАННЫЕ:
  Возраст: {age} лет
  Пол: {sex}
  ИМТ: {bmi:.1f}

ОБЩАЯ ОЦЕНКА РИСКА:
  {risk_emoji} Балл риска: {risk_score:.1f}/100
  {risk_emoji} Группа риска: {risk_category}
  
  Основной фактор риска: {risk_info.get('max_risk_disease', 'нет')}
  Заболеваний с высокой вероятностью (≥50%): {risk_info.get('n_high_risk', 0)}
  Заболеваний со средней вероятностью (≥30%): {risk_info.get('n_moderate_risk', 0)}

{'='*80}
ВЕРОЯТНОСТИ ЗАБОЛЕВАНИЙ (отсортировано по убыванию):
{'='*80}
"""
        
        for disease, prob in disease_probs:
            pred = predictions_series[disease]
            pred_text = "✓ ОБНАРУЖЕНО" if pred == 1 else "✗ не обнаружено"
            threshold = self.optimal_thresholds.get(disease, 0.5)
            
            # Визуальная шкала
            bar_length = 40
            filled = int(prob * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Цветовая индикация
            if prob >= 0.7:
                status_emoji = "🔴"
            elif prob >= 0.5:
                status_emoji = "🟠"
            elif prob >= 0.3:
                status_emoji = "🟡"
            else:
                status_emoji = "🟢"
            
            report += f"\n{status_emoji} {disease}:\n"
            report += f"  [{bar}] {prob*100:.1f}%\n"
            report += f"  Статус: {pred_text} (порог: {threshold:.2f})\n"
            
            # Если есть истинные метки, показываем их
            if true_labels is not None:
                true_value = true_labels.iloc[0][disease]
                true_text = "Истинное значение: ДА" if true_value == 1 else "Истинное значение: НЕТ"
                correct = "✓" if pred == true_value else "✗"
                report += f"  {correct} {true_text}\n"
        
        # Ключевые показатели
        report += f"\n{'='*80}\n"
        report += "КЛЮЧЕВЫЕ БИОМАРКЕРЫ:\n"
        report += f"{'='*80}\n"
        
        # Находим заболевание с наибольшей вероятностью
        if disease_probs:
            top_disease = disease_probs[0][0]
            top_prob = disease_probs[0][1]
            
            report += f"\nНаиболее вероятное: {top_disease} ({top_prob*100:.1f}%)\n"
            report += f"Важнейшие показатели для этого диагноза:\n\n"
            
            if top_disease in self.feature_importance:
                top_features = self.feature_importance[top_disease].head(10)
                
                for idx, row in top_features.iterrows():
                    feature = row['feature']
                    importance = row['importance']
                    
                    if feature in X.columns:
                        value = X.iloc[0][feature]
                        report += f"  • {feature}: {value:.2f} (важность: {importance:.3f})\n"
        
        report += f"\n{'='*80}\n"
        
        # Сводка по обнаруженным заболеваниям
        detected_diseases = [disease for disease, pred in zip(self.diseases, predictions_series) if pred == 1]
        
        if detected_diseases:
            report += f"\n⚠️  ОБНАРУЖЕНО ЗАБОЛЕВАНИЙ: {len(detected_diseases)}\n"
            for disease in detected_diseases:
                prob = probabilities_series[disease]
                report += f"   • {disease} (вероятность: {prob*100:.1f}%)\n"
        else:
            report += f"\n✅ Заболевания не обнаружены\n"
        
        # Точность предсказания (если есть истинные метки)
        if true_labels is not None:
            true_series = true_labels.iloc[0]
            correct_predictions = (predictions_series == true_series).sum()
            total_diseases = len(self.diseases)
            accuracy = (correct_predictions / total_diseases) * 100
            
            report += f"\n{'='*80}\n"
            report += f"ТОЧНОСТЬ ПРЕДСКАЗАНИЯ: {accuracy:.1f}% ({correct_predictions}/{total_diseases})\n"
            report += f"{'='*80}\n"
        
        return report
    
    def analyze_patient(
        self, 
        dataset_path: str, 
        patient_id: int,
        show_true_labels: bool = True
    ):
        """
        Полный анализ пациента
        
        Args:
            dataset_path: путь к датасету
            patient_id: номер пациента
            show_true_labels: показывать ли истинные метки (для валидации)
        """
        # Загружаем данные пациента
        X, true_labels, patient_data = self.load_patient_data(dataset_path, patient_id)
        
        # Если не хотим показывать истинные метки
        if not show_true_labels:
            true_labels = None
        
        # Генерируем отчет
        report = self.generate_patient_report(X, patient_data, true_labels)
        
        # Выводим отчет
        print(report)
        
        return report
    
    def analyze_multiple_patients(
        self,
        dataset_path: str,
        patient_ids: list,
        show_true_labels: bool = True
    ):
        """
        Анализ нескольких пациентов
        
        Args:
            dataset_path: путь к датасету
            patient_ids: список номеров пациентов
            show_true_labels: показывать ли истинные метки
        """
        for i, patient_id in enumerate(patient_ids):
            print(f"\n{'#'*80}")
            print(f"ПАЦИЕНТ {i+1} из {len(patient_ids)}")
            print(f"{'#'*80}")
            
            self.analyze_patient(dataset_path, patient_id, show_true_labels)
            
            if i < len(patient_ids) - 1:
                print("\n\n")


def main():
    """
    Основная функция для анализа пациентов
    """
    print("="*80)
    print("СИСТЕМА АНАЛИЗА ПАЦИЕНТОВ")
    print("="*80)
    
    # Настройки
    MODEL_PATH = 'cardio_classifier.pkl'
    DATASET_PATH = 'cardio_synthetic_dataset.csv'
    
    # Создаем анализатор
    analyzer = PatientAnalyzer(model_path=MODEL_PATH)
    
    # Варианты использования:
    
    # 1. Анализ одного пациента
    # print("\n" + "="*80)
    # print("АНАЛИЗ ОДНОГО ПАЦИЕНТА")
    # print("="*80)
    
    # PATIENT_ID = 42  # Измените на нужный номер
    # analyzer.analyze_patient(
    #     dataset_path=DATASET_PATH,
    #     patient_id=PATIENT_ID,
    #     show_true_labels=True  # True - показывать истинные значения для проверки
    # )
    
    # 2. Анализ нескольких пациентов
    print("\n\n" + "="*80)
    print("АНАЛИЗ НЕСКОЛЬКИХ ПАЦИЕНТОВ")
    print("="*80)
    
    # Случайные пациенты или конкретные номера
    # import random
    # random.seed(42)
    
    # Загрузим датасет чтобы узнать размер
    df = pd.read_csv(DATASET_PATH)
    max_id = len(df) - 1
    
    # Выбираем 3 случайных пациента
    # random_patients = random.sample(range(max_id), 3)
    
    # analyzer.analyze_multiple_patients(
    #     dataset_path=DATASET_PATH,
    #     patient_ids=random_patients,
    #     show_true_labels=True
    # )
    
    # 3. Пример анализа конкретных пациентов
    analyzer.analyze_multiple_patients(
        dataset_path=DATASET_PATH,
        patient_ids=[0, 100, 256],
        show_true_labels=True
    )


if __name__ == "__main__":
    main()