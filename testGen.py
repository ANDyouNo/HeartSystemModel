import numpy as np
import pandas as pd
from scipy.stats import truncnorm

class NoisyMedicalDatasetGenerator:
    """
    Генератор тестового датасета с шумом, вариативностью и пограничными случаями
    для более реалистичной оценки модели
    """
    
    def __init__(self, n_samples=10000, random_state=2024, noise_level=0.12):
        """
        Args:
            n_samples: количество образцов
            random_state: для воспроизводимости
            noise_level: уровень шума (0.12 = ±12% случайных вариаций)
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.noise_level = noise_level
        np.random.seed(random_state)
        
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
        
        # Референсные диапазоны (те же, что и в оригинале)
        self.reference_ranges = {
            'Общий белок': (64, 83),
            'АСЛО': (0, 200),
            'Креатинин': (62, 115),
            'Мочевина': (2.5, 8.3),
            'Мочевая кислота': (200, 420),
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
    
    def _add_noise(self, value, noise_level=None):
        """Добавление случайного шума к значению"""
        if noise_level is None:
            noise_level = self.noise_level
        
        noise = np.random.normal(1.0, noise_level)
        return value * noise
    
    def _truncated_normal(self, mean, std, low, high, size=1):
        """Генерация усеченного нормального распределения"""
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    
    def _generate_base_profile(self):
        """Генерация базовых демографических данных с вариативностью"""
        # Более широкое распределение возраста
        age = np.random.randint(18, 90, self.n_samples)
        sex = np.random.choice(['М', 'Ж'], self.n_samples)
        
        # ИМТ с большей вариативностью
        bmi_mean = 24 + (age - 40) * 0.06  # Немного другой коэффициент
        bmi_std = 4.5 + np.random.uniform(0, 1, self.n_samples)  # Переменная дисперсия
        bmi = self._truncated_normal(bmi_mean, bmi_std, 15, 50, self.n_samples)
        
        return pd.DataFrame({
            'Возраст': age,
            'Пол': sex,
            'ИМТ': bmi
        })
    
    def _generate_disease_profile(self):
        """
        Генерация профиля заболеваний с:
        - Другими вероятностями
        - Пограничными случаями
        - Неожиданными комбинациями
        """
        disease_matrix = np.zeros((self.n_samples, len(self.diseases)))
        
        # Меньше здоровых людей (более сложная выборка)
        healthy_prob = 0.25  # Было 0.3
        healthy_mask = np.random.random(self.n_samples) < healthy_prob
        
        # Добавляем пограничные случаи (5%)
        borderline_mask = np.random.random(self.n_samples) < 0.05
        
        for i in range(self.n_samples):
            if healthy_mask[i]:
                continue
            
            # Пограничный случай: легкие симптомы без четкого диагноза
            if borderline_mask[i]:
                # Выбираем 1-2 заболевания с низкой вероятностью
                n_borderline = np.random.choice([1, 2])
                borderline_diseases = np.random.choice(
                    len(self.diseases), 
                    n_borderline, 
                    replace=False
                )
                # Вероятность проявления 30-60% (слабые признаки)
                for disease_idx in borderline_diseases:
                    if np.random.random() < 0.5:
                        disease_matrix[i, disease_idx] = 1
                continue
            
            # Больше возможных заболеваний одновременно
            n_diseases = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])  # Добавили тройки
            
            # Кластеры с немного измененными вероятностями
            metabolic_cluster = [0, 1, 2, 3]
            cardiac_cluster = [4, 5, 10]
            inflammatory_cluster = [6, 7]
            hepatic_cluster = [8, 11]
            anemia_idx = 9
            
            # Измененное распределение кластеров
            cluster_choice = np.random.random()
            if cluster_choice < 0.45:  # Было 0.5
                primary_diseases = np.random.choice(
                    metabolic_cluster, 
                    min(n_diseases, len(metabolic_cluster)), 
                    replace=False
                )
            elif cluster_choice < 0.75:  # Было 0.7
                primary_diseases = np.random.choice(
                    cardiac_cluster, 
                    min(n_diseases, len(cardiac_cluster)), 
                    replace=False
                )
            elif cluster_choice < 0.88:  # Было 0.85
                primary_diseases = np.random.choice(
                    inflammatory_cluster, 
                    min(n_diseases, len(inflammatory_cluster)), 
                    replace=False
                )
            else:
                primary_diseases = np.random.choice(
                    hepatic_cluster, 
                    min(n_diseases, len(hepatic_cluster)), 
                    replace=False
                )
            
            disease_matrix[i, primary_diseases] = 1
            
            # Анемия как сопутствующее (увеличили до 20%)
            if np.random.random() < 0.20:
                disease_matrix[i, anemia_idx] = 1
            
            # Добавляем "неожиданные" комбинации (8% случаев)
            if np.random.random() < 0.08:
                random_disease = np.random.choice(len(self.diseases))
                disease_matrix[i, random_disease] = 1
        
        return disease_matrix
    
    def _generate_biomarkers(self, demographics, disease_matrix):
        """
        Генерация биомаркеров с:
        - Значительным шумом
        - Измененными весами влияния
        - Атипичными случаями
        """
        biomarkers = {}
        
        for i in range(self.n_samples):
            age = demographics.iloc[i]['Возраст']
            bmi = demographics.iloc[i]['ИМТ']
            diseases = disease_matrix[i]
            
            # Базовые значения с большим разбросом
            base_values = {}
            for marker, (low, high) in self.reference_ranges.items():
                mean = (low + high) / 2
                std = (high - low) / 5  # Было /6, теперь больше разброс
                
                # Добавляем случайные "выбросы" (2% случаев)
                if np.random.random() < 0.02:
                    base_values[marker] = self._truncated_normal(
                        mean, std * 2, low * 0.5, high * 1.8, 1
                    )[0]
                else:
                    base_values[marker] = self._truncated_normal(
                        mean, std, low * 0.8, high * 1.2, 1
                    )[0]
            
            # Возрастной и ИМТ факторы с шумом
            age_factor = 1 + (age - 50) * 0.0035 * self._add_noise(1.0, 0.05)
            bmi_factor = 1 + (bmi - 24) * 0.012 * self._add_noise(1.0, 0.05)
            
            # === МОДИФИКАЦИЯ ПО ЗАБОЛЕВАНИЯМ С ИЗМЕНЕННЫМИ ВЕСАМИ ===
            
            # ДИСЛИПИДЕМИЯ (немного другие коэффициенты)
            if diseases[0] == 1:
                base_values['ЛПНП'] *= self._add_noise(np.random.uniform(1.3, 2.6))
                base_values['ЛПОНП'] *= self._add_noise(np.random.uniform(1.2, 2.1))
                base_values['Триглицериды'] *= self._add_noise(np.random.uniform(1.3, 3.2))
                base_values['Холестерин общий'] *= self._add_noise(np.random.uniform(1.25, 1.85))
                base_values['ЛПВП'] *= self._add_noise(np.random.uniform(0.55, 0.92))
            
            # АТЕРОСКЛЕРОЗ
            if diseases[1] == 1:
                base_values['ЛПНП'] *= self._add_noise(np.random.uniform(1.4, 2.9))
                base_values['ЛПОНП'] *= self._add_noise(np.random.uniform(1.15, 1.9))
                base_values['Триглицериды'] *= self._add_noise(np.random.uniform(1.25, 2.6))
                base_values['ЛПВП'] *= self._add_noise(np.random.uniform(0.48, 0.82))
                base_values['Глюкоза'] *= self._add_noise(np.random.uniform(1.08, 1.45))
            
            # МЕТАБОЛИЧЕСКИЙ СИНДРОМ
            if diseases[2] == 1:
                base_values['Глюкоза'] *= self._add_noise(np.random.uniform(1.15, 1.9))
                base_values['HbA1c'] *= self._add_noise(np.random.uniform(1.15, 1.7))
                base_values['ЛПНП'] *= self._add_noise(np.random.uniform(1.25, 2.3))
                base_values['Триглицериды'] *= self._add_noise(np.random.uniform(1.4, 3.2))
                base_values['ЛПВП'] *= self._add_noise(np.random.uniform(0.58, 0.83))
                base_values['Гамма-ГТ'] *= self._add_noise(np.random.uniform(1.15, 2.2))
            
            # САХАРНЫЙ ДИАБЕТ
            if diseases[3] == 1:
                base_values['Глюкоза'] *= self._add_noise(np.random.uniform(1.4, 2.7))
                base_values['HbA1c'] *= self._add_noise(np.random.uniform(1.3, 2.2))
                
                # Добавляем вариативность: не всегда оба маркера высокие
                if np.random.random() < 0.15:
                    base_values['Глюкоза'] *= 0.7  # "Компенсированный" диабет
            
            # ХСН
            if diseases[4] == 1:
                base_values['Креатинин'] *= self._add_noise(np.random.uniform(1.15, 2.1))
                base_values['Мочевина'] *= self._add_noise(np.random.uniform(1.25, 2.3))
                base_values['Альбумин'] *= self._add_noise(np.random.uniform(0.68, 0.92))
                base_values['Натрий'] *= self._add_noise(np.random.uniform(0.91, 0.985))
                base_values['Калий'] *= self._add_noise(np.random.uniform(1.03, 1.28))
                base_values['Магний'] *= self._add_noise(np.random.uniform(0.78, 0.96))
            
            # МИОПАТИЯ
            if diseases[5] == 1:
                base_values['КФК'] *= self._add_noise(np.random.uniform(1.8, 5.5))
                base_values['ЛДГ'] *= self._add_noise(np.random.uniform(1.4, 3.0))
                base_values['АСТ'] *= self._add_noise(np.random.uniform(1.25, 2.3))
                base_values['АЛТ'] *= self._add_noise(np.random.uniform(1.15, 1.9))
            
            # ПОСТСТРЕПТОКОККОВЫЙ КАРДИТ
            if diseases[6] == 1:
                base_values['АСЛО'] *= self._add_noise(np.random.uniform(2.8, 9.0))
                base_values['СРБ'] *= self._add_noise(np.random.uniform(4.5, 16.0))
                base_values['РФ'] *= self._add_noise(np.random.uniform(0.75, 2.2))
            
            # РЕВМАТИЧЕСКАЯ ЛИХОРАДКА
            if diseases[7] == 1:
                base_values['АСЛО'] *= self._add_noise(np.random.uniform(3.5, 11.0))
                base_values['СРБ'] *= self._add_noise(np.random.uniform(7.0, 22.0))
                base_values['РФ'] *= self._add_noise(np.random.uniform(1.1, 3.2))
                base_values['Общий белок'] *= self._add_noise(np.random.uniform(0.88, 1.12))
                base_values['Альбумин'] *= self._add_noise(np.random.uniform(0.78, 0.96))
            
            # АТЕРОСКЛЕРОТИЧЕСКАЯ КАРДИОПАТИЯ
            if diseases[8] == 1:
                base_values['Гамма-ГТ'] *= self._add_noise(np.random.uniform(1.4, 3.8))
                base_values['АСТ'] *= self._add_noise(np.random.uniform(1.15, 2.2))
                base_values['АЛТ'] *= self._add_noise(np.random.uniform(1.25, 2.4))
                base_values['Альбумин'] *= self._add_noise(np.random.uniform(0.83, 0.96))
            
            # АНЕМИЯ
            if diseases[9] == 1:
                base_values['Железо'] *= self._add_noise(np.random.uniform(0.35, 0.75))
                base_values['Фолаты'] *= self._add_noise(np.random.uniform(0.45, 0.82))
            
            # ЭЛЕКТРОЛИТНЫЕ АРИТМИИ
            if diseases[10] == 1:
                electrolyte_type = np.random.random()
                if electrolyte_type < 0.4:
                    base_values['Калий'] *= self._add_noise(np.random.uniform(0.65, 0.87))
                elif electrolyte_type < 0.8:
                    base_values['Калий'] *= self._add_noise(np.random.uniform(1.13, 1.38))
                else:
                    # Нормокалиемия с другими нарушениями
                    pass
                
                base_values['Магний'] *= self._add_noise(np.random.uniform(0.68, 0.92))
                base_values['Кальций ионизированный'] *= self._add_noise(np.random.uniform(0.83, 1.18))
            
            # АЛКОГОЛЬНАЯ КАРДИОМИОПАТИЯ
            if diseases[11] == 1:
                base_values['Гамма-ГТ'] *= self._add_noise(np.random.uniform(2.5, 9.0))
                base_values['АСТ'] *= self._add_noise(np.random.uniform(1.8, 4.5))
                base_values['АЛТ'] *= self._add_noise(np.random.uniform(1.4, 2.7))
                base_values['КФК'] *= self._add_noise(np.random.uniform(1.2, 2.7))
                base_values['ЛДГ'] *= self._add_noise(np.random.uniform(1.3, 2.4))
            
            # Применяем возрастной и ИМТ факторы
            base_values['Глюкоза'] *= age_factor * bmi_factor
            base_values['Холестерин общий'] *= age_factor * self._add_noise(1.0, 0.08)
            base_values['Креатинин'] *= age_factor * self._add_noise(1.0, 0.08)
            
            # Добавляем "атипичные" случаи (5%)
            if np.random.random() < 0.05:
                # Случайный маркер получает неожиданное значение
                random_marker = np.random.choice(list(base_values.keys()))
                if np.random.random() < 0.5:
                    base_values[random_marker] *= np.random.uniform(0.5, 0.7)
                else:
                    base_values[random_marker] *= np.random.uniform(1.5, 2.0)
            
            # Корреляции с шумом
            if base_values['ЛПВП'] > 0:
                index_ater = (base_values['Холестерин общий'] - base_values['ЛПВП']) / base_values['ЛПВП']
                index_ater *= self._add_noise(1.0, 0.10)
            else:
                index_ater = 5.0
            base_values['Индекс атерогенности'] = np.clip(index_ater, 0, 10)
            
            # Билирубин непрямой с погрешностью
            base_values['Билирубин непрямой'] = max(0, 
                base_values['Билирубин общий'] - base_values['Билирубин прямой'] + 
                np.random.normal(0, 0.5)
            )
            
            # Сохраняем значения
            for marker, value in base_values.items():
                if marker not in biomarkers:
                    biomarkers[marker] = []
                biomarkers[marker].append(value)
        
        return pd.DataFrame(biomarkers)
    
    def generate(self):
        """Генерация полного тестового датасета"""
        print("="*70)
        print("ГЕНЕРАЦИЯ ТЕСТОВОГО ДАТАСЕТА С ШУМОМ")
        print("="*70)
        print(f"Параметры:")
        print(f"  Размер выборки: {self.n_samples}")
        print(f"  Random seed: {self.random_state}")
        print(f"  Уровень шума: ±{self.noise_level*100:.0f}%")
        print()
        
        print("Генерация демографических данных...")
        demographics = self._generate_base_profile()
        
        print("Генерация профиля заболеваний (с пограничными случаями)...")
        disease_matrix = self._generate_disease_profile()
        disease_df = pd.DataFrame(disease_matrix, columns=self.diseases)
        
        print("Генерация биомаркеров (с шумом и атипичными случаями)...")
        biomarkers = self._generate_biomarkers(demographics, disease_matrix)
        
        # Объединяем все данные
        dataset = pd.concat([demographics, biomarkers, disease_df], axis=1)
        
        # Округляем значения
        for col in biomarkers.columns:
            dataset[col] = dataset[col].round(2)
        
        print(f"\n{'='*70}")
        print(f"ТЕСТОВЫЙ ДАТАСЕТ СГЕНЕРИРОВАН: {len(dataset)} образцов")
        print(f"{'='*70}")
        
        print(f"\nРаспределение заболеваний:")
        for disease in self.diseases:
            count = disease_df[disease].sum()
            pct = (count / len(dataset)) * 100
            print(f"  {disease}: {count} ({pct:.1f}%)")
        
        # Статистика
        print(f"\n{'='*70}")
        print("СТАТИСТИКА:")
        print(f"{'='*70}")
        print(f"Средний возраст: {dataset['Возраст'].mean():.1f} лет")
        print(f"Средний ИМТ: {dataset['ИМТ'].mean():.1f}")
        print(f"Соотношение М/Ж: {(dataset['Пол'] == 'М').sum()}/{(dataset['Пол'] == 'Ж').sum()}")
        
        # Показываем статистику по количеству заболеваний (НЕ добавляя в датасет)
        n_diseases_per_patient = dataset[self.diseases].sum(axis=1)
        print(f"\nРаспределение по количеству заболеваний:")
        print(n_diseases_per_patient.value_counts().sort_index())
        
        return dataset


def main():
    """
    Генерация тестового датасета
    """
    # Параметры
    N_SAMPLES = 10000  # Тестовая выборка меньше
    RANDOM_STATE = 2024  # Другой seed
    NOISE_LEVEL = 0.12  # ±12% шума
    OUTPUT_FILE = 'cardio_test_dataset.csv'
    
    # Генерация
    generator = NoisyMedicalDatasetGenerator(
        n_samples=N_SAMPLES,
        random_state=RANDOM_STATE,
        noise_level=NOISE_LEVEL
    )
    
    test_dataset = generator.generate()
    
    # Сохранение
    test_dataset.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ Тестовый датасет сохранен в '{OUTPUT_FILE}'")
    
    print("\n" + "="*70)
    print("ОСОБЕННОСТИ ТЕСТОВОГО ДАТАСЕТА:")
    print("="*70)
    print("✓ Добавлен случайный шум ±12% к биомаркерам")
    print("✓ Изменены веса влияния заболеваний на маркеры")
    print("✓ Добавлены пограничные случаи (~5%)")
    print("✓ Добавлены атипичные паттерны (~5%)")
    print("✓ Добавлены неожиданные комбинации заболеваний (~8%)")
    print("✓ Увеличена вариативность электролитных нарушений")
    print("✓ Больше случаев с множественными заболеваниями")
    print("\n⚠️  Этот датасет сложнее для модели, чем обучающий!")
    print("="*70)


if __name__ == "__main__":
    main()