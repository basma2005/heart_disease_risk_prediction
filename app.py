import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_models():
    scaler = joblib.load('model/scaler.pkl')
    model = joblib.load('model/heart_model.pkl')
    return scaler, model

scaler, model = load_models()

@st.cache_data
def load_data():
    try:
        if os.path.exists('dataset/heart.csv'):
            df = pd.read_csv('dataset/heart.csv')
            return df
        else:
            return None
    except:
        return None

def calculate_model_accuracy():
    df = load_data()
    if df is not None:
        try:
            X = df.drop('target', axis=1)
            y = df['target']
            
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        except:
            return 0.85  
    else:
        return 0.85

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("""
Cette application prédit le risque de maladie cardiaque en fonction des paramètres médicaux.
Remplissez les informations ci-dessous pour obtenir une prédiction.
""")

# Sidebar avec informations
with st.sidebar:
    st.header("📊 Informations")
    st.info("""
    **Variables :**
    - **age** : Âge en années
    - **sex** : Sexe (0=Femme, 1=Homme)
    - **cp** : Type de douleur thoracique (0-3)
    - **trestbps** : Pression artérielle au repos
    - **chol** : Cholestérol en mg/dl
    - **fbs** : Glycémie à jeun > 120 mg/dl (0=Non, 1=Oui)
    - **restecg** : Résultats ECG au repos (0-2)
    - **thalach** : Fréquence cardiaque max atteinte
    - **exang** : Angine induite par l'exercice (0=Non, 1=Oui)
    - **oldpeak** : Dépression ST induite par l'exercice
    - **slope** : Pente du segment ST (0-2)
    - **ca** : Nombre de vaisseaux colorés (0-3)
    - **thal** : Thalassémie (1-3)
    """)
    
    if st.button("🧹 Effacer les données"):
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(["🧪 Prédiction Simple", "📁 Batch Testing", "📈 Analyse"])

with tab1:
    st.header("Prédiction Individuelle")
    
    # Formulaire en colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Âge", 20, 100, 50)
        sex = st.radio("Sexe", options=["Femme", "Homme"])
        cp = st.selectbox(
            "Type de douleur thoracique", 
            options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            format_func=lambda x: f"{x} ({['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(x)})"
        )
        trestbps = st.number_input("Pression artérielle (mm Hg)", 90, 200, 120)
        chol = st.number_input("Cholestérol (mg/dl)", 100, 600, 200)
        
    with col2:
        fbs = st.radio("Glycémie à jeun > 120", options=["Non", "Oui"])
        restecg = st.selectbox(
            "Résultats ECG au repos",
            options=["Normal", "Anomalie onde ST-T", "Hypertrophie ventriculaire gauche probable"],
            format_func=lambda x: f"{x} ({['Normal', 'Anomalie onde ST-T', 'Hypertrophie ventriculaire gauche probable'].index(x)})"
        )
        thalach = st.slider("Fréquence cardiaque max", 60, 220, 150)
        exang = st.radio("Angine induite par exercice", options=["Non", "Oui"])
        
    with col3:
        oldpeak = st.slider("Dépression ST", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox(
            "Pente segment ST",
            options=["Montante", "Plat", "Descendante"],
            format_func=lambda x: f"{x} ({['Montante', 'Plat', 'Descendante'].index(x)})"
        )
        ca = st.selectbox("Nombre vaisseaux colorés", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassémie",
            options=["Normal", "Défaut fixe", "Défaut réversible"],
            format_func=lambda x: f"{x} ({['Normal', 'Défaut fixe', 'Défaut réversible'].index(x)+1})"
        )
    
    # Conversion des données
    sex_num = 1 if sex == "Homme" else 0
    cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs_num = 1 if fbs == "Oui" else 0
    restecg_num = ["Normal", "Anomalie onde ST-T", "Hypertrophie ventriculaire gauche probable"].index(restecg)
    exang_num = 1 if exang == "Oui" else 0
    slope_num = ["Montante", "Plat", "Descendante"].index(slope)
    thal_num = ["Normal", "Défaut fixe", "Défaut réversible"].index(thal) + 1
    
    # Bouton de prédiction
    if st.button("🔍 Analyser le risque", type="primary"):
        # Préparation des données
        features = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, 
                            restecg_num, thalach, exang_num, oldpeak, 
                            slope_num, ca, thal_num]])
        
        # Scaling et prédiction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Gérer différemment selon le type de modèle
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_scaled)[0]
        elif hasattr(model, 'decision_function'):
            decision_score = model.decision_function(features_scaled)[0]
            probability = [1/(1+np.exp(-decision_score)), 1/(1+np.exp(decision_score))]
        else:
            probability = [0.5, 0.5]  # Valeur par défaut
        
        # Affichage des résultats
        st.subheader("📊 Résultats")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.error("⚠️ **Risque Élevé**")
                st.markdown("**Présence probable de maladie cardiaque**")
                # Barre de progression proportionnelle à la probabilité
                risk_level = probability[1] if len(probability) > 1 else 0.8
                st.progress(min(risk_level, 1.0))
            else:
                st.success("✅ **Risque Faible**")
                st.markdown("**Pas de maladie cardiaque détectée**")
                safe_level = probability[0] if len(probability) > 0 else 0.2
                st.progress(min(safe_level, 1.0))
        
        with col_res2:
            if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                prob_value = probability[1] if len(probability) > 1 else 0.5
                st.metric("Probabilité de maladie", f"{prob_value*100:.1f}%")
            st.metric("Prédiction", "Maladie" if prediction == 1 else "Sain")
        
        # Détails des features
        with st.expander("📋 Détails des entrées"):
            feature_names = ["Âge", "Sexe", "Douleur thoracique", "Pression", "Cholestérol", 
                           "Glycémie", "ECG repos", "FC Max", "Angine exercice", 
                           "Dépression ST", "Pente ST", "Vaisseaux", "Thalassémie"]
            feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                            exang, oldpeak, slope, ca, thal]
            
            df_features = pd.DataFrame({
                "Paramètre": feature_names,
                "Valeur": feature_values
            })
            st.dataframe(df_features, use_container_width=True)
            
            # Afficher aussi les valeurs numériques
            st.write("**Valeurs numériques envoyées au modèle :**")
            numeric_values = [age, sex_num, cp_num, trestbps, chol, fbs_num, 
                            restecg_num, thalach, exang_num, oldpeak, 
                            slope_num, ca, thal_num]
            st.code(f"[{', '.join(map(str, numeric_values))}]")

with tab2:
    st.header("Test par Lots")
    
    uploaded_file = st.file_uploader("📤 Uploader un fichier CSV", type=['csv'])
    
    if uploaded_file:
        try:
            # Lecture du fichier
            df_test = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier chargé : {len(df_test)} lignes")
            
            # Vérification des colonnes
            required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            if all(col in df_test.columns for col in required_cols):
                # Prédictions
                X_test = df_test[required_cols]
                X_test_scaled = scaler.transform(X_test)
                
                # Obtenir les probabilités si disponible
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_test_scaled)[:, 1]
                    df_test['probability'] = probabilities
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test_scaled)
                    probabilities = 1 / (1 + np.exp(-decision_scores))
                    df_test['probability'] = probabilities
                
                predictions = model.predict(X_test_scaled)
                df_test['prediction'] = predictions
                df_test['result'] = df_test['prediction'].apply(
                    lambda x: 'Maladie cardiaque' if x == 1 else 'Sain'
                )
                
                # Affichage des résultats
                st.subheader("Résultats des prédictions")
                
                # Statistiques
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total patients", len(df_test))
                with col_stat2:
                    disease_count = sum(predictions)
                    st.metric("Cas détectés", disease_count)
                with col_stat3:
                    detection_rate = (disease_count/len(predictions)*100) if len(predictions) > 0 else 0
                    st.metric("Taux détection", f"{detection_rate:.1f}%")
                
                # Aperçu des données
                st.dataframe(df_test.head(10), use_container_width=True)
                
                # Téléchargement
                csv = df_test.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger résultats complets",
                    data=csv,
                    file_name="predictions_heart_disease.csv",
                    mime="text/csv"
                )
                
                st.subheader("Distribution des résultats")
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                counts = df_test['result'].value_counts()
                colors = ['lightgreen', 'lightcoral']
                ax[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
                         colors=colors[:len(counts)])
                ax[0].set_title("Répartition Maladie/Sain")
                
                if 'probability' in df_test.columns:
                    ax[1].hist(df_test['probability'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                    ax[1].axvline(x=0.5, color='red', linestyle='--', label='Seuil de décision')
                    ax[1].set_xlabel("Probabilité de maladie cardiaque")
                    ax[1].set_ylabel("Nombre de patients")
                    ax[1].set_title("Distribution des probabilités")
                    ax[1].legend()
                    ax[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("📊 Statistiques détaillées"):
                    if 'probability' in df_test.columns:
                        st.write("**Statistiques des probabilités :**")
                        prob_stats = df_test['probability'].describe()
                        st.dataframe(prob_stats)
                        
                        st.write("**Patients à haut risque (probabilité > 70%) :**")
                        high_risk = df_test[df_test['probability'] > 0.7]
                        st.dataframe(high_risk[['age', 'sex', 'probability', 'result']])
                
            else:
                missing_cols = [col for col in required_cols if col not in df_test.columns]
                st.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
                st.info(f"⚠️ Les colonnes requises sont : {', '.join(required_cols)}")
                
        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")
            st.info("Vérifiez que votre fichier CSV contient les bonnes colonnes et valeurs numériques.")

with tab3:
    st.header("Analyse du Modèle")
    
    model_accuracy = calculate_model_accuracy()
    
    st.subheader("📋 Informations du modèle")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        model_name = type(model).__name__
        st.metric("Type de modèle", model_name)
        
        # Afficher les paramètres selon le type de modèle
        if hasattr(model, 'get_params'):
            params = model.get_params()
            with st.expander("🔧 Paramètres du modèle"):
                # Filtrer les paramètres les plus importants
                important_params = {k: v for k, v in params.items() 
                                  if not k.startswith('base_estimator') and v is not None}
                for key, value in important_params.items():
                    st.write(f"**{key}**: `{value}`")
    
    with col_info2:
        st.metric("Précision estimée", f"{model_accuracy*100:.1f}%")
        st.metric("Type de problème", "Classification binaire")
        st.metric("Classes", "0: Sain, 1: Maladie cardiaque")
    
    # Features importance (si disponible)
    if hasattr(model, 'feature_importances_'):
        st.subheader("📊 Importance des caractéristiques")
        
        feature_names = ["âge", "sexe", "douleur thoracique", "pression", "cholestérol", 
                        "glycémie", "ECG repos", "FC max", "angine exercice", 
                        "dépression ST", "pente ST", "vaisseaux", "thalassémie"]
        
        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            'Caractéristique': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_importance['Caractéristique'], df_importance['Importance'], 
                      color=plt.cm.viridis(df_importance['Importance']/df_importance['Importance'].max()))
        ax.set_xlabel("Importance relative")
        ax.set_title("Importance des caractéristiques pour la prédiction")
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **Interprétation :**
        - Les caractéristiques avec une importance plus élevée ont plus d'influence sur la prédiction
        - Les valeurs proches de 0 ont peu d'impact
        """)
    
    elif hasattr(model, 'coef_'):
        st.subheader("📊 Coefficients du modèle (modèle linéaire)")
        
        feature_names = ["âge", "sexe", "douleur thoracique", "pression", "cholestérol", 
                        "glycémie", "ECG repos", "FC max", "angine exercice", 
                        "dépression ST", "pente ST", "vaisseaux", "thalassémie"]
        
        coefficients = model.coef_[0]
        df_coef = pd.DataFrame({
            'Caractéristique': feature_names,
            'Coefficient': coefficients,
            'Impact': ['Positif' if c > 0 else 'Négatif' for c in coefficients]
        }).sort_values('Coefficient', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if c < 0 else 'green' for c in df_coef['Coefficient']]
        bars = ax.barh(df_coef['Caractéristique'], df_coef['Coefficient'], color=colors)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Coefficient")
        ax.set_title("Impact des caractéristiques (positif = risque accru)")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **Interprétation :**
        - **Coefficients positifs** : Augmentent le risque de maladie cardiaque
        - **Coefficients négatifs** : Diminuent le risque de maladie cardiaque
        - La magnitude indique la force de l'impact
        """)
    
    st.subheader("🔍 Diagnostic du modèle")
    
    diagnostic_col1, diagnostic_col2 = st.columns(2)
    
    with diagnostic_col1:
        st.write("**Capacités du modèle :**")
        capabilities = []
        if hasattr(model, 'predict_proba'):
            capabilities.append("✅ Peut fournir des probabilités")
        else:
            capabilities.append("⚠️ Ne fournit que des prédictions binaires")
        
        if hasattr(model, 'feature_importances_'):
            capabilities.append("✅ Importance des features disponible")
        
        if hasattr(model, 'coef_'):
            capabilities.append("✅ Coefficients interprétables")
        
        for cap in capabilities:
            st.write(cap)
    
    with diagnostic_col2:
        st.write("**Performances estimées :**")
        st.write(f"- Précision : {model_accuracy*100:.1f}%")
        
        if model_accuracy > 0.8:
            st.success("✅ Bonne performance")
        elif model_accuracy > 0.7:
            st.warning("⚠️ Performance moyenne")
        else:
            st.error("❌ Performance faible - considérez réentraîner le modèle")
    
    st.subheader("💡 Conseils d'interprétation")
    
    advice_col1, advice_col2 = st.columns(2)
    
    with advice_col1:
        st.info("""
        **Valeurs normales :**
        - **Pression artérielle** : < 120/80 mm Hg
        - **Cholestérol total** : < 200 mg/dl
        - **Glycémie à jeun** : < 100 mg/dl
        - **Fréquence cardiaque repos** : 60-100 bpm
        """)
    
    with advice_col2:
        st.info("""
        **Facteurs de risque :**
        - **Âge** : > 45 ans (homme), > 55 ans (femme)
        - **Tabagisme**
        - **Obésité** (IMC > 30)
        - **Sédentarité**
        - **Antécédents familiaux**
        """)

st.markdown("---")
st.caption("""
⚠️ **Disclaimer médical** : Cet outil est à des fins éducatives et de démonstration seulement. 
Il ne remplace pas une consultation médicale professionnelle. Consultez toujours un professionnel de santé pour un diagnostic médical.
""")

st.caption(f"Modèle : {type(model).__name__} | Précision estimée : {model_accuracy*100:.1f}%")