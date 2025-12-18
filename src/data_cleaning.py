import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HeartDiseaseDataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        """Charge les données"""
        print("=== CHARGEMENT DES DONNÉES ===")
        self.df = pd.read_csv(self.file_path)
        print(f"✅ Dataset chargé: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        return True
    
    def check_missing_values(self):
        """Vérifie les valeurs manquantes"""
        print("\n=== VALEURS MANQUANTES ===")
        missing = self.df.isnull().sum()
        
        for col, count in missing.items():
            if count > 0:
                print(f"{col}: {count} valeurs manquantes")
                # Remplir avec la médiane
                self.df[col].fillna(self.df[col].median(), inplace=True)
                print(f"   → Remplacées par la médiane")
            else:
                print(f"{col}: Aucune valeur manquante")
    
    def check_duplicates(self):
        """Vérifie les doublons"""
        print("\n=== DOUBLONS ===")
        duplicates = self.df.duplicated().sum()
        
        if duplicates > 0:
            print(f"{duplicates} doublons trouvés")
            self.df = self.df.drop_duplicates()
            print(f"Doublons supprimés")
        else:
            print("Aucun doublon trouvé")
    
    def analyze_target(self):
        """Analyse la variable cible"""
        print("\n=== VARIABLE CIBLE ===")
        target_counts = self.df['target'].value_counts()
        
        print("Distribution de la maladie cardiaque:")
        print(f"✅ Sain (0): {target_counts[0]} patients")
        print(f"❌ Malade (1): {target_counts[1]} patients")
        
        plt.figure(figsize=(8, 6))
        plt.pie(target_counts.values, labels=['Sain', 'Malade'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title('Distribution des Cas de Maladie Cardiaque')
        plt.show()
    
    def basic_statistics(self):
        """Affiche les statistiques de base"""
        print("\n=== STATISTIQUES DE BASE ===")
        print(self.df.describe())
    
    def save_cleaned_data(self):
        """Sauvegarde les données nettoyées"""
        output_path = 'dataset/heart_cleaned.csv'
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Données sauvegardées: {output_path}")
        return output_path
    
    def run_simple_cleaning(self):
        """Exécute le nettoyage complet"""
        print("🚀 DÉMARRAGE DU NETTOYAGE")
        
        steps = [
            self.load_data,
            self.check_missing_values,
            self.check_duplicates,
            self.analyze_target,
            self.basic_statistics,
            self.save_cleaned_data
        ]
        
        for step in steps:
            try:
                step()
            except Exception as e:
                print(f"Erreur: {e}")
                continue
        
        print(f"\n✨ NETTOYAGE TERMINÉ!")
        print(f"📊 Données finales: {self.df.shape}")
        return self.df

if __name__ == "__main__":
    cleaner = HeartDiseaseDataCleaner('dataset/heart.csv')
    cleaned_data = cleaner.run_simple_cleaning()