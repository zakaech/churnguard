# ChurnGuard

Prédiction et explication du risque de départ (churn) des clients d'un opérateur télécom.

> Projet en cours de construction — README complet prévu en phase 5.

## Données

- Jeu de données : **Telco Customer Churn** (IBM), disponible sur Kaggle :
  `blastchar/telco-customer-churn`.
- Licence : **à compléter** (à vérifier sur la page Kaggle avant publication du dépôt).
- Les données ne sont pas versionnées. Les tests utilisent un échantillon **synthétique**
  (`tests/data/sample.csv`) généré par `scripts/make_synthetic_sample.py`.

## Installation

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Linux/macOS : source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Commandes

```bash
python tasks.py download        # télécharge les données (API Kaggle, ou instructions manuelles)
python tasks.py data            # nettoie et découpe (train/val/test) vers data/processed/
python tasks.py notebook        # ré-exécute les notebooks (exploration, résultats)
python tasks.py train-baseline  # entraîne les baselines, métriques sur la validation
python tasks.py train           # compare les modèles, règle, choisit modèle + seuil, sauvegarde
python tasks.py evaluate-test   # évaluation UNIQUE sur le test (refuse de se relancer)
python tasks.py explain-global  # importance globale SHAP (validation) -> reports/
python tasks.py explain --customer-id <ID>  # facteurs de risque d'un client (affichage)
python tasks.py api             # API FastAPI (http://127.0.0.1:8000/docs)
python tasks.py mlflow-ui       # interface MLflow (http://127.0.0.1:5000)
python tasks.py test            # tests pytest
python tasks.py lint            # ruff
```

## Démarche de modélisation

Tous les chiffres ci-dessous proviennent de l'exécution des commandes listées plus haut
(`reports/` contient les fichiers bruts correspondants).

1. **Découpage stratifié 60 / 20 / 20** (train 4 225, validation 1 409, test 1 409 clients ;
   26,5 % de départs dans chaque partie), fait avant toute analyse.
2. **Prétraitement dans un pipeline** (imputation, normalisation, one-hot), appris sur le
   train uniquement.
3. **Comparaison en validation croisée stratifiée (5 plis, train uniquement)** de 3 modèles
   (régression logistique, Random Forest, XGBoost) × 3 stratégies de déséquilibre
   (aucune, poids des classes, SMOTENC), puis recherche aléatoire d'hyperparamètres
   (25 combinaisons) pour chaque modèle. Critère : PR-AUC. Toutes les expériences sont
   journalisées dans MLflow.
4. **Choix du modèle et du seuil sur la validation**, selon une règle fixée **avant** de voir
   les résultats : seuil = précision maximale parmi les seuils donnant un rappel ≥ 80 % ;
   modèle = meilleure précision à ce seuil.
5. **Évaluation unique sur le test** du modèle et du seuil retenus.

### Baselines (validation, seuil 0,5)

| Modèle | Exactitude | Précision | Rappel | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Classe majoritaire | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| Régression logistique | 0.802 | 0.657 | 0.532 | 0.588 | 0.836 | 0.642 |

La classe majoritaire atteint 73,5 % d'exactitude sans détecter un seul départ :
l'exactitude est trompeuse quand les classes sont déséquilibrées.

### Comparaison en validation croisée (PR-AUC, moyenne ± écart-type sur 5 plis)

| Modèle | Aucune | Poids des classes | SMOTENC | Après recherche |
|---|---|---|---|---|
| Régression logistique | 0.666 ± 0.025 | 0.665 ± 0.026 | 0.651 ± 0.030 | 0.668 ± 0.024 (aucune) |
| Random Forest | 0.622 ± 0.005 | 0.616 ± 0.010 | 0.601 ± 0.009 | 0.675 ± 0.012 (aucune) |
| XGBoost | 0.619 ± 0.016 | 0.624 ± 0.008 | 0.620 ± 0.013 | 0.677 ± 0.015 (poids) |

- Le rééquilibrage (poids des classes, SMOTENC) n'améliore pas le classement des clients :
  il décale surtout les probabilités, ce que le choix du seuil fait déjà.
- Random Forest et XGBoost gagnent surtout à être régularisés (arbres moins profonds,
  feuilles plus grandes). Random Forest final : 400 arbres, `max_depth=12`,
  `min_samples_leaf=20`, `max_features=0.3`.

### Choix du modèle et du seuil (validation)

| Modèle | Seuil | Précision | Rappel | F1 | Départs détectés | Fausses alertes | Départs manqués |
|---|---|---|---|---|---|---|---|
| Régression logistique | 0.243 | 0.499 | 0.802 | 0.615 | 300 | 301 | 74 |
| **Random Forest** | **0.271** | **0.508** | 0.802 | 0.622 | 300 | 291 | 74 |
| XGBoost | 0.489 | 0.508 | 0.805 | 0.623 | 301 | 292 | 73 |

**Les trois modèles sont équivalents à un client près.** Random Forest (300 / 591 = 0,50761)
devance XGBoost (301 / 593 = 0,50759) d'un client ; la régression logistique fait environ
10 fausses alertes de plus sur 1 409 clients, et les écarts de PR-AUC en validation croisée
sont inférieurs à l'écart-type entre plis.

**Pourquoi avoir tout de même appliqué la règle (Random Forest) ?** La règle de sélection a
été fixée avant de voir les résultats. La modifier après coup pour préférer un autre modèle
reviendrait à ajuster nos choix sur la validation, et à présenter une estimation trop
optimiste. Respecter la règle garde la démarche honnête ; l'équivalence des modèles est
documentée ici. La régression logistique, plus simple et plus lisible, aurait été un choix
tout aussi défendable s'il avait été décidé avant les résultats.

![Choix du seuil sur la validation](reports/figures/threshold_validation.png)

**Seuil retenu : 0,2712.** Le seuil par défaut de 0,5 ne détecte que 45 % des départs
(169 sur 374 en validation) ; l'objectif métier est d'en détecter au moins 80 %, en limitant
au maximum les offres de fidélisation envoyées à tort.

### Résultat final sur le test (évaluation unique)

Random Forest, seuil 0,2712, fixé sur la validation puis non modifié :

| Précision | Rappel | F1 | ROC-AUC | PR-AUC | Exactitude |
|---|---|---|---|---|---|
| 0.514 | 0.802 | 0.626 | 0.844 | 0.653 | 0.746 |

Matrice de confusion sur 1 409 clients : 300 départs détectés, 74 départs manqués,
284 fausses alertes, 751 clients fidèles correctement non signalés. Le rappel visé (80 %)
est atteint sur le test (80,2 %) ; environ une alerte sur deux correspond à un vrai départ.
Les résultats du test sont proches de ceux de la validation : pas de signe de surajustement
au choix du seuil.

### Analyse de coût (illustrative)

> ⚠️ **Les montants sont des hypothèses, pas des données de l'opérateur.** Ils servent
> uniquement à illustrer l'intérêt d'un seuil adapté. Le seuil retenu reste celui fixé par
> l'objectif de rappel, pas par ce calcul.

Hypothèses (en unités monétaires du jeu de données, à partir de la facture mensuelle
médiane du train, 70,40) : un client perdu coûte 12 mois (844,80) ; une offre de
fidélisation coûte 1 mois (70,40) ; 30 % des clients ciblés restent grâce à l'offre.

```
coût = FN × coût_client_perdu
     + TP × (1 − taux_succès) × coût_client_perdu
     + (TP + FP) × coût_offre
```

| Politique | Coût (validation) | vs aucune action | Coût (test) | vs aucune action |
|---|---|---|---|---|
| Aucune action | 315 955 | — | 315 955 | — |
| Seuil 0,5 | 291 287 | −7,8 % | 289 161 | −8,5 % |
| **Seuil retenu (0,271)** | **281 530** | **−10,9 %** | **281 037** | **−11,1 %** |
| Seuil de coût minimal* | 281 065 (0,35) | −11,0 % | 280 009 (0,30) | −11,4 % |

\* Indicatif seulement : sur le test, ce seuil est calculé a posteriori et n'est pas utilisé.

![Coût selon le seuil](reports/figures/cost_validation.png)

Avec ces hypothèses, la courbe de coût est presque plate entre les seuils 0,25 et 0,40 : le
seuil fixé par le rappel cible est à 0,2 % (validation) du coût minimal, alors que le seuil
0,5 coûte nettement plus. Avec d'autres hypothèses (coût de l'offre, taux de succès), le
seuil optimal changerait : c'est un paramètre à fixer avec le métier.

## Explicabilité (SHAP)

Pour chaque client, **SHAP** répartit l'écart entre sa probabilité de départ et la probabilité
moyenne (0,265) entre les variables. Avec ce Random Forest, les contributions sont exprimées en
**points de probabilité** et s'additionnent exactement (valeur de base + somme des contributions
= probabilité prédite ; écart maximal mesuré de l'ordre de 10⁻¹⁵). Les colonnes one-hot d'une
même variable sont regroupées pour revenir aux 19 variables d'origine.

### Importance globale (validation)

![Importance globale SHAP](reports/figures/shap_global_importance.png)

- **Le contrat domine** (9,4 points d'impact moyen) : le contrat au mois ajoute en moyenne
  9,8 points au risque, le contrat de deux ans en retire 10,6.
- Suivent le type d'internet (5,2), l'ancienneté (5,2), les options sécurité en ligne (4,2) et
  support technique (3,0), puis les montants facturés (2,6 chacun).
- L'ancienneté augmente fortement le risque pendant les 6 premiers mois (+9,2 points en
  moyenne) et le diminue au-delà de 18 mois ; une facture mensuelle supérieure à 80 l'augmente.
- Les variables démographiques (genre, couple, senior, personnes à charge) pèsent moins de
  0,3 point en moyenne.

Le détail (beeswarm, courbes de dépendance, graphiques waterfall) est dans
[notebooks/02_resultats.ipynb](notebooks/02_resultats.ipynb).

### Explication d'un client

`ChurnExplainer.top_factors` renvoie les **5 facteurs les plus importants, dans les deux sens**,
avec l'impact en points de probabilité. Exemple sur un **client fictif** (contrat au mois,
2 mois d'ancienneté, fibre, chèque électronique) :

```
FICTIF-A — probabilité 75,7 % -> signalé (seuil 27,1 %)
  1. Contrat : au mois — augmente le risque (+11,6 points)
  2. Ancienneté : 2 mois — augmente le risque (+9,2 points)
  3. Type d'internet : fibre optique — augmente le risque (+7,2 points)
  4. Option sécurité en ligne : non — augmente le risque (+5,8 points)
  5. Total facturé depuis le début : 171,00 — augmente le risque (+4,2 points)
```

Pour un vrai client du jeu de données : `python tasks.py explain --customer-id <ID> --split test`
(sortie affichée seulement, rien n'est enregistré).

### Limites

- SHAP explique **le modèle**, pas la réalité : aucune lecture causale.
- `tenure` et `TotalCharges` sont très corrélées (Spearman 0,90) : le modèle peut répartir leur
  effet entre les deux de façon arbitraire ; il faut les lire ensemble.
- Les contributions en points de probabilité sont exactes pour ce Random Forest ; un autre type
  de modèle (par exemple XGBoost, dont les sorties sont en log-odds) demanderait une autre
  approche.

## Niveaux de risque

| Niveau | Règle | Validation : clients | Départs réels |
|---|---|---|---|
| **élevé** | probabilité ≥ 0,2712 (seuil du modèle) | 591 | 300 (50,8 %) |
| **moyen** | 0,05 ≤ probabilité < 0,2712 | 461 | 66 (14,3 %) |
| **faible** | probabilité < 0,05 | 357 | 8 (2,2 %) |

- « élevé » correspond **exactement** aux clients signalés par le seuil : c'est la même
  comparaison (probabilité ≥ seuil), et le seuil est lu dans le modèle sauvegardé.
- La limite 0,05 entre « faible » et « moyen » a été choisie sur la **validation uniquement**
  (le test n'a pas servi) : sous 0,05, le taux de départ observé est très bas et stable
  (0,8 à 3,1 % selon la tranche) ; juste au-dessus, il passe à environ 10 % puis reste entre
  10 et 14 % jusqu'à 0,20. Le groupe « moyen » contient 89 % des départs non signalés par le
  seuil (66 sur 74) : c'est une cible naturelle pour des actions plus légères.
- **Limites** : les effectifs par tranche sont petits (63 clients entre 0,05 et 0,08), donc la
  position exacte de la rupture est approximative ; et les probabilités du Random Forest ne
  sont pas calibrées (0,10 ne veut pas dire « 10 % de chances de partir »). C'est pourquoi la
  limite est justifiée par les taux de départ observés, pas par la valeur de la probabilité.

## API (FastAPI)

```bash
python tasks.py api      # http://127.0.0.1:8000 — documentation interactive : /docs
```

Le modèle est chargé une seule fois au démarrage (chemin : variable d'environnement
`CHURNGUARD_MODEL_PATH`, sinon celui de `config.yaml`).

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du service, modèle, version, seuil, limites |
| POST | `/predict` | un client → probabilité, décision, niveau de risque, 5 facteurs (SHAP) |
| POST | `/predict/batch` | plusieurs clients (JSON) ; `include_factors=false` par défaut |
| POST | `/predict/batch/csv` | fichier CSV au format brut ; `include_factors=false` par défaut |

**Entrée** : un client au **format brut du fichier Telco** (mêmes colonnes ; `SeniorCitizen` en
0/1 ; `TotalCharges` peut être vide pour un client d'ancienneté 0). L'API applique le même
nettoyage que l'entraînement (`clean_features`). Sont refusés (erreur 422) : champ manquant ou
en trop, modalité inconnue, valeurs hors bornes, `TotalCharges` vide avec une ancienneté > 0,
et **combinaisons de services incohérentes** (par exemple des options internet sans internet).
Pour un CSV, les colonnes en trop (comme `Churn`) sont ignorées et listées dans la réponse ;
une ligne invalide est signalée avec son numéro.

Exemple (client fictif, réponse réelle de `POST /predict`) :

```json
{
  "customerID": "FICTIF-A",
  "churn_probability": 0.757,
  "churn_predicted": true,
  "risk_level": "élevé",
  "threshold": 0.2712,
  "top_factors": [
    {"label": "Contrat", "value_label": "au mois", "impact_points": 11.6, "direction": "hausse",
     "text": "Contrat : au mois — augmente le risque (+11,6 points)"},
    "… 4 autres facteurs"
  ]
}
```

**Temps de réponse mesurés** (HTTP local, poste de développement) : `/predict` ≈ 0,1 s ;
lot de 2 000 clients ≈ 0,3 s sans facteurs, ≈ 17-18 s avec facteurs ; lot de 500 clients avec
facteurs ≈ 4,6 s. D'où deux limites par lot : **2 000 clients sans facteurs, 500 avec**
(au-delà : erreur 413).
