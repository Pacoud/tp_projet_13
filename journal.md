## Session 1 — Objectif : Initialiser l'architecture et le pipeline d'extraction de base

**Prompt utilisé :**
"Agis comme un développeur Python expert. > Je dois créer le MVP d'un "Extracteur de données de formulaires". L'objectif est de lire un fichier PDF, d'en extraire le texte brut, puis d'utiliser un LLM pour extraire les données spécifiques dans un format JSON strict.

Voici mes choix techniques obligatoires :

PyMuPDF (fitz) pour l'extraction du texte brut du PDF.

Pydantic pour définir le schéma de données attendu et valider les types.

Azure OpenAI (modèle gpt-4o-mini) en utilisant la fonctionnalité native "Structured Outputs" (via client.beta.chat.completions.parse avec Pydantic).

Ce que tu dois générer :

Un fichier requirements.txt avec les bonnes bibliothèques.

Un fichier .env.example pour les variables Azure OpenAI (Endpoint, Clé, Version, Nom du déploiement).

Un script main.py qui fait ceci :

Charge les variables du .env.

Définit un modèle Pydantic simple (par exemple pour une facture : nom_client, email_client, montant_total (float) et date). Important : Configure le schéma pour que les valeurs soient optionnelles (ex: None) si l'IA ne trouve pas l'information, afin d'éviter les hallucinations.

Une fonction pour lire le PDF avec PyMuPDF.

Une fonction qui envoie le texte à l'API Azure OpenAI et retourne l'objet Pydantic validé.

Un bloc try/except robuste pour gérer les cas où le fichier n'existe pas ou si l'API renvoie une erreur.

Génère un code propre, commenté et modulaire.
"

**Problèmes anticipés & Solutions :** Je voulais m'assurer que l'IA n'invente pas de données (hallucinations) si un champ est manquant dans le PDF.
-> _Solution appliquée :_ J'ai explicitement demandé à l'IA générative de configurer le modèle Pydantic avec des champs optionnels (pouvant être `None`). Ainsi, si le LLM ne trouve pas le montant, le programme ne crashe pas et renvoie simplement une valeur nulle.

**Apprentissage :**
J'ai compris l'importance d'utiliser `client.beta.chat.completions.parse` d'OpenAI avec Pydantic. Cela force le LLM à répondre dans un format JSON strict et évite d'avoir à faire du "prompt engineering" hasardeux ou de parser péniblement du texte libre.


