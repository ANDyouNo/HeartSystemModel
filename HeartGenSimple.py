import numpy as np
import pandas as pd
from scipy.stats import truncnorm

class MedicalDatasetGenerator:
    """
    Генератор синтетического датасета для кардиологических заболеваний
    с учетом медицинских взаимосвязей между биомаркерами
    """
    
    def __init__(self, n_samples=50000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Определение заболеваний
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
        
        # Референсные диапазоны (норма)
        self.reference_ranges = {
            'Общий белок': (64, 83),
            'АСЛО': (0, 200),
            'Креатинин': (62, 115),  # мкмоль/л
            'Мочевина': (2.5, 8.3),  # ммоль/л
            'Мочевая кислота': (200, 420),  # мкмоль/л
            'Альбумин': (35, 52),
            'СРБ': (0, 5),
            'РФ': (0, 14),
            'Глюкоза': (3.9, 6.1),
            'ЛПОНП': (0.26, 1.04),
            'Триглицериды': (0.5, 2.0),
            'АЛТ': (0, 41),
            'АСТ': (0, 40),
            'Амилаза': (25, 125),
            'Холестерин общий': (3.0, 5.2),
            'ЛПВП': (1.0, 2.2),
            'ЛПНП': (0, 3.3),
            'Гамма-ГТ': (0, 55),
            'КФК': (24, 195),
            'HbA1c': (4.0, 6.0),
            'ЛДГ': (125, 220),
            'Щелочная фосфатаза': (40, 150),
            'Билирубин общий': (3.4, 20.5),
            'Билирубин прямой': (0, 5.1),
            'Кальций ионизированный': (1.15, 1.29),
            'Калий': (3.5, 5.1),
            'Натрий': (136, 145),
            'Хлориды': (98, 107),
            'Магний': (0.66, 1.07),
            'Железо': (10.7, 30.4),
            'Фосфор': (0.87, 1.45),
            'Фолаты': (7, 45),
            'Цинк': (11, 18)
        }
    
    def _truncated_normal(self, mean, std, low, high, size=1):
        """Генерация усеченного нормального распределения"""
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    
    def _generate_base_profile(self):
        """Генерация базовых демографических данных"""
        age = np.random.randint(18, 85, self.n_samples)
        sex = np.random.choice(['М', 'Ж'], self.n_samples)
        
        # ИМТ с учетом возраста
        bmi_mean = 24 + (age - 40) * 0.05
        bmi = self._truncated_normal(bmi_mean, 4, 16, 45, self.n_samples)
        
        return pd.DataFrame({
            'Возраст': age,
            'Пол': sex,
            'ИМТ': bmi
        })
    
    def _generate_disease_profile(self):
        """Генерация профиля заболеваний с учетом взаимосвязей"""
        disease_matrix = np.zeros((self.n_samples, len(self.diseases)))
        
        # Вероятность здорового человека
        healthy_prob = 0.3
        healthy_mask = np.random.random(self.n_samples) < healthy_prob
        
        for i in range(self.n_samples):
            if healthy_mask[i]:
                continue
                
            # Количество заболеваний (максимум 2)
            n_diseases = np.random.choice([1, 2], p=[0.6, 0.4])
            
            # Кластеры взаимосвязанных заболеваний
            metabolic_cluster = [0, 1, 2, 3]  # Дислипидемия, Атеросклероз, МС, СД
            cardiac_cluster = [4, 5, 10]  # ХСН, Миопатия, Аритмии
            inflammatory_cluster = [6, 7]  # Кардит, Ревматизм
            hepatic_cluster = [8, 11]  # Кардиопатия, Алкоголь
            
            # Выбираем основной кластер
            if np.random.random() < 0.5:
                # Метаболический кластер (самый частый)
                primary_diseases = np.random.choice(metabolic_cluster, 
                                                    min(n_diseases, len(metabolic_cluster)), 
                                                    replace=False)
            elif np.random.random() < 0.7:
                # Сердечный кластер
                primary_diseases = np.random.choice(cardiac_cluster, 
                                                    min(n_diseases, len(cardiac_cluster)), 
                                                    replace=False)
            elif np.random.random() < 0.85:
                # Воспалительный кластер
                primary_diseases = np.random.choice(inflammatory_cluster, 
                                                    min(n_diseases, len(inflammatory_cluster)), 
                                                    replace=False)
            else:
                # Печеночный кластер
                primary_diseases = np.random.choice(hepatic_cluster, 
                                                    min(n_diseases, len(hepatic_cluster)), 
                                                    replace=False)
            
            disease_matrix[i, primary_diseases] = 1
            
            # Добавляем анемию как сопутствующее (15% случаев)
            if np.random.random() < 0.15:
                disease_matrix[i, 9] = 1
        
        return disease_matrix
    
    def _generate_biomarkers(self, demographics, disease_matrix):
        """Генерация биомаркеров с учетом заболеваний и взаимосвязей"""
        biomarkers = {}
        
        for i in range(self.n_samples):
            age = demographics.iloc[i]['Возраст']
            bmi = demographics.iloc[i]['ИМТ']
            diseases = disease_matrix[i]
            
            # Базовые значения (норма с небольшими отклонениями)
            base_values = {}
            for marker, (low, high) in self.reference_ranges.items():
                mean = (low + high) / 2
                std = (high - low) / 6
                base_values[marker] = self._truncated_normal(mean, std, low * 0.8, high * 1.2, 1)[0]
            
            # Модификация на основе возраста и ИМТ
            age_factor = 1 + (age - 50) * 0.003
            bmi_factor = 1 + (bmi - 24) * 0.01
            
            # ДИСЛИПИДЕМИЯ (индекс 0)
            if diseases[0] == 1:
                base_values['ЛПНП'] *= np.random.uniform(1.4, 2.5)
                base_values['ЛПОНП'] *= np.random.uniform(1.3, 2.0)
                base_values['Триглицериды'] *= np.random.uniform(1.4, 3.0)
                base_values['Холестерин общий'] *= np.random.uniform(1.3, 1.8)
                base_values['ЛПВП'] *= np.random.uniform(0.6, 0.9)
            
            # АТЕРОСКЛЕРОЗ (индекс 1)
            if diseases[1] == 1:
                base_values['ЛПНП'] *= np.random.uniform(1.5, 2.8)
                base_values['ЛПОНП'] *= np.random.uniform(1.2, 1.8)
                base_values['Триглицериды'] *= np.random.uniform(1.3, 2.5)
                base_values['ЛПВП'] *= np.random.uniform(0.5, 0.8)
                base_values['Глюкоза'] *= np.random.uniform(1.1, 1.4)
            
            # МЕТАБОЛИЧЕСКИЙ СИНДРОМ (индекс 2)
            if diseases[2] == 1:
                base_values['Глюкоза'] *= np.random.uniform(1.2, 1.8)
                base_values['HbA1c'] *= np.random.uniform(1.2, 1.6)
                base_values['ЛПНП'] *= np.random.uniform(1.3, 2.2)
                base_values['Триглицериды'] *= np.random.uniform(1.5, 3.0)
                base_values['ЛПВП'] *= np.random.uniform(0.6, 0.8)
                base_values['Гамма-ГТ'] *= np.random.uniform(1.2, 2.0)
            
            # САХАРНЫЙ ДИАБЕТ (индекс 3)
            if diseases[3] == 1:
                base_values['Глюкоза'] *= np.random.uniform(1.5, 2.5)
                base_values['HbA1c'] *= np.random.uniform(1.4, 2.0)
            
            # ХСН (индекс 4)
            if diseases[4] == 1:
                base_values['Креатинин'] *= np.random.uniform(1.2, 2.0)
                base_values['Мочевина'] *= np.random.uniform(1.3, 2.2)
                base_values['Альбумин'] *= np.random.uniform(0.7, 0.9)
                base_values['Натрий'] *= np.random.uniform(0.92, 0.98)
                base_values['Калий'] *= np.random.uniform(1.05, 1.25)
                base_values['Магний'] *= np.random.uniform(0.8, 0.95)
            
            # МИОПАТИЯ (индекс 5)
            if diseases[5] == 1:
                base_values['КФК'] *= np.random.uniform(2.0, 5.0)
                base_values['ЛДГ'] *= np.random.uniform(1.5, 2.8)
                base_values['АСТ'] *= np.random.uniform(1.3, 2.2)
                base_values['АЛТ'] *= np.random.uniform(1.2, 1.8)
            
            # ПОСТСТРЕПТОКОККОВЫЙ КАРДИТ (индекс 6)
            if diseases[6] == 1:
                base_values['АСЛО'] *= np.random.uniform(3.0, 8.0)
                base_values['СРБ'] *= np.random.uniform(5.0, 15.0)
                base_values['РФ'] *= np.random.uniform(0.8, 2.0)
            
            # РЕВМАТИЧЕСКАЯ ЛИХОРАДКА (индекс 7)
            if diseases[7] == 1:
                base_values['АСЛО'] *= np.random.uniform(4.0, 10.0)
                base_values['СРБ'] *= np.random.uniform(8.0, 20.0)
                base_values['РФ'] *= np.random.uniform(1.2, 3.0)
                base_values['Общий белок'] *= np.random.uniform(0.9, 1.1)
                base_values['Альбумин'] *= np.random.uniform(0.8, 0.95)
            
            # АТЕРОСКЛЕРОТИЧЕСКАЯ КАРДИОПАТИЯ (индекс 8)
            if diseases[8] == 1:
                base_values['Гамма-ГТ'] *= np.random.uniform(1.5, 3.5)
                base_values['АСТ'] *= np.random.uniform(1.2, 2.0)
                base_values['АЛТ'] *= np.random.uniform(1.3, 2.2)
                base_values['Альбумин'] *= np.random.uniform(0.85, 0.95)
            
            # АНЕМИЯ (индекс 9)
            if diseases[9] == 1:
                base_values['Железо'] *= np.random.uniform(0.4, 0.7)
                base_values['Фолаты'] *= np.random.uniform(0.5, 0.8)
            
            # ЭЛЕКТРОЛИТНЫЕ АРИТМИИ (индекс 10)
            if diseases[10] == 1:
                # Случайный дисбаланс электролитов
                if np.random.random() < 0.5:
                    base_values['Калий'] *= np.random.uniform(0.7, 0.85)  # Гипокалиемия
                else:
                    base_values['Калий'] *= np.random.uniform(1.15, 1.35)  # Гиперкалиемия
                base_values['Магний'] *= np.random.uniform(0.7, 0.9)
                base_values['Кальций ионизированный'] *= np.random.uniform(0.85, 1.15)
            
            # АЛКОГОЛЬНАЯ КАРДИОМИОПАТИЯ (индекс 11)
            if diseases[11] == 1:
                base_values['Гамма-ГТ'] *= np.random.uniform(3.0, 8.0)
                base_values['АСТ'] *= np.random.uniform(2.0, 4.0)
                base_values['АЛТ'] *= np.random.uniform(1.5, 2.5)
                base_values['КФК'] *= np.random.uniform(1.3, 2.5)
                base_values['ЛДГ'] *= np.random.uniform(1.4, 2.2)
            
            # Применяем возрастной и ИМТ факторы к некоторым маркерам
            base_values['Глюкоза'] *= age_factor * bmi_factor
            base_values['Холестерин общий'] *= age_factor
            base_values['Креатинин'] *= age_factor
            
            # Добавляем корреляции между связанными маркерами
            # Индекс атерогенности
            if base_values['ЛПВП'] > 0:
                index_ater = (base_values['Холестерин общий'] - base_values['ЛПВП']) / base_values['ЛПВП']
            else:
                index_ater = 5.0
            base_values['Индекс атерогенности'] = np.clip(index_ater, 0, 10)
            
            # Билирубин непрямой
            base_values['Билирубин непрямой'] = base_values['Билирубин общий'] - base_values['Билирубин прямой']
            
            # Сохраняем значения для текущего образца
            for marker, value in base_values.items():
                if marker not in biomarkers:
                    biomarkers[marker] = []
                biomarkers[marker].append(value)
        
        return pd.DataFrame(biomarkers)
    
    def generate(self):
        """Генерация полного датасета"""
        print("Генерация демографических данных...")
        demographics = self._generate_base_profile()
        
        print("Генерация профиля заболеваний...")
        disease_matrix = self._generate_disease_profile()
        disease_df = pd.DataFrame(disease_matrix, columns=self.diseases)
        
        print("Генерация биомаркеров...")
        biomarkers = self._generate_biomarkers(demographics, disease_matrix)
        
        # Объединяем все данные
        dataset = pd.concat([demographics, biomarkers, disease_df], axis=1)
        
        # Округляем значения
        for col in biomarkers.columns:
            dataset[col] = dataset[col].round(2)
        
        print(f"\nДатасет сгенерирован: {len(dataset)} образцов")
        print(f"Распределение заболеваний:")
        for disease in self.diseases:
            count = disease_df[disease].sum()
            pct = (count / len(dataset)) * 100
            print(f"  {disease}: {count} ({pct:.1f}%)")
        
        return dataset

# Использование
if __name__ == "__main__":
    generator = MedicalDatasetGenerator(n_samples=50000, random_state=42)
    dataset = generator.generate()
    
    # Сохранение датасета
    dataset.to_csv('cardio_synthetic_dataset.csv', index=False, encoding='utf-8-sig')
    print("\nДатасет сохранен в 'cardio_synthetic_dataset.csv'")
    
    # Статистика
    print("\n" + "="*60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("="*60)
    print(f"Всего образцов: {len(dataset)}")
    print(f"\nСредний возраст: {dataset['Возраст'].mean():.1f} лет")
    print(f"Средний ИМТ: {dataset['ИМТ'].mean():.1f}")
    print(f"Соотношение М/Ж: {(dataset['Пол'] == 'М').sum()}/{(dataset['Пол'] == 'Ж').sum()}")
    
    # Анализ мультилейбл
    disease_cols = generator.diseases
    dataset['Количество заболеваний'] = dataset[disease_cols].sum(axis=1)
    print(f"\nРаспределение по количеству заболеваний:")
    print(dataset['Количество заболеваний'].value_counts().sort_index())
    
    print("\nПервые 5 строк датасета:")
    print(dataset.head())