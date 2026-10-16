# Classification de signaux de trading BTC/USD (BUY / HOLD / SELL)

Projet de certification Data Scientist, Liora (ex-DataScientest), 2026.

Prédire si le Bitcoin va nettement monter (**BUY**), baisser (**SELL**) ou rester stable (**HOLD**) dans les **5 prochains jours**, à partir d'indicateurs techniques. Le projet couvre toute la chaîne : collecte via API, nettoyage, feature engineering, modélisation et application Streamlit.

## Résultats

Test hors échantillon sur les 995 derniers jours (2023-2025) :

| Modèle | Accuracy | F1 macro |
|---|---|---|
| **Gradient Boosting** (seuil 0,35) | 35,9 % | **0,35** |
| Tirage aléatoire | 34,6 % | 0,33 |

**Le modèle ne fait pas significativement mieux que le hasard** : avec ces seuls indicateurs techniques, la direction du BTC à 5 jours reste très difficile à prédire. L'intérêt du projet tient à la démarche et à l'analyse critique des résultats.

## Démarche

- **Données** : cotations journalières BTCUSD de 2012 à 2025 (API Financial Modeling Prep)
- **Cible ajustée à la volatilité** : BUY si le rendement à 5 jours dépasse la volatilité sur 20 jours, SELL s'il passe sous son opposé, HOLD sinon. Un seuil fixe n'aurait pas le même sens en marché calme et en marché agité.
- **21 variables** : momentum, volatilité, tendance, bandes de Bollinger
- **Modèles** : Régression Logistique, Random Forest, Gradient Boosting, avec validation temporelle (`TimeSeriesSplit`), GridSearch et ajustement des seuils de décision
- **Interprétation** : le modèle s'appuie surtout sur les bandes de Bollinger (42 %) et la volatilité (25 %)

## Limites et pistes

- Le seuil de décision a été choisi sur le jeu de test : il faudrait un jeu de validation distinct.
- Le test de robustesse (notebook 09) recouvre en partie la période d'entraînement : ce n'est pas une validation indépendante.
- Certaines variables sont des niveaux de prix, mal extrapolés par les arbres : des variables relatives seraient plus robustes.
- Prochaine étape : un backtest avec coûts de transaction, comparé à la simple détention du BTC.

## Lancer le projet

    pip install -r requirements.txt
    streamlit run streamlit/app.py

Pour réexécuter les notebooks (01 à 09, dans l'ordre), copier `.env.example` en `.env` et y renseigner une clé API Financial Modeling Prep.

**Auteur** : Romaric TCHAKOUNTE HANDOU, Data Scientist Junior 