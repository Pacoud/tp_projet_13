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

## Session 2 - Objectif : Paufiner le code, embellir le visuel

**Prompt utilisé :**
Premier prompt-
est ce que l'on peut travailler l'esthétique pour l'utilisateur lorsqu'il lance l'execution du script python ? qu'il voie un écran de chargement de quelque sorte

Deuxième prompt-
en fait pour le spinner je voulais plutot un rectangle de ligne dicontinue en mouvement, également dans le tableau de l'affichage des valeurs, l'email ne s'affiche pas correctement, les dimensions de sa case ne sont pas conformes avec les cases des autres données

Troisième prompt-
Je voulais comprendre ou était géré le spinner, je lui ai demandé de me l'expliquer.

**Problèmes anticipés & Solutions :**
Tout en demandant à l'IA d'embellir le code, j'ai fait attention à la cohérence de ce qu'elle me rendait, cela dit elle m'a donné en premier lieu un affichage en tableau, ce que je n'avais pas demandé, il est somme toute bien en globalité mais le mail est mal affiché je vais régler ce problème. Le spinner était pas à mon gout je lui ai demandé de le changer pour etre plus esthétique.
_Solution appliquée :_ l'intelligence artificielle a bien compris ma demande et a généré un spinner propre et esthétique meme si j'ai du repasser derrière en re-modifiant le spinner par exemple.

**Apprentissage :**
j'ai appris à mieux communiquer avec l'IA, mieux collaborer avec, notamment lorsque j'ai du modifier le fichier main.py, l'IA m'a précisément communiqué ce qui était à changer et d'ailleurs c'était déjà bien commenté, cela m'a aidé et a renforcé l'importance que j'accorde aux commentaires de code.

## Session 3 - Objectif : Gérer différents types de pdf autres que factures

Premier Prompt-
(Je ne vais pas dire à l'IA de tout implémenter toute seule, je vais déjà lui demander de me guider)

"peux tu mettre un if qui détecte si le pdf en question est une facture sinon, il fait une analyse générique"

Deuxième prompt-
"Capture du résultat erroné"
"actuellement l'IA gère mal les accents dans ses réponses et les remplaces par des chiffres qui rendent le résultat illisible"

**Problèmes anticipés & Solutions :**
Initiallement je pars d'un code ou l'IA ne prends pas vraiment en compte d'autres types de PDF que des factures, je souhaite élargir la portée du code pour qu'il puisse prendre en compte d'autres types de documents, je lui demande donc de me guider pour cela. On va donc créer une fonction de test qui vérifie la nature du pdf, si c'est une facture, alors on la traite comme telle sinon, c'est un traitement générique.
L'IA renvoie des résultats avec des coquilles car elle ne prend pas en compte les accents et caractères spéciaux, je dois régler cela.
Problème avec les caractères spéciaux: Je remarque que l'IA traite mal les caracères spéciaux lors de son analyse, un prétraitement du texte devrait etre fait avant de l'envoyer au LLM

_Solutions_
_Problème du type de document_ : le nouveau code permet d'avoir un traitement par situation et par type de document

_Problème des accents et caractères spéciaux_ : Un prétraitement manuel des infos du document permet à l'IA de travailler avec un nouveau document propre et lisible.(suppression des octets nuls et normalisation des accents)

## Session 4 - Objectif : Améliorer la gestion des erreurs et des cas particuliers

Premier Prompt-
maitnenant peux tu analyser le code et y détecter d'éventuelles failles ou situations spécifiques dans lesquelles on ne pourra pas générer un résultat correct ou autre ?

''L'IA me cite des cas de failles critiques''

Deuxième Prompt-
peux tu implémenter dans un premier temps toutes failles et cas limites critiques et moyennes citées au dessus ?
**Problèmes anticipés et solutions**
Si le pdf est une image sans texte, alors le code renvoie un résultat inutile et trompeur car tous les champs sont à None, il faudrait plutôt afficher un message d'erreur et quitter.

_Solution_ :lever une exception ou afficher un message clair et quitter si pas de texte

Un autre problème est que si le pdf est trop volumineux, l'API atteindra un dépassement de tokens et cela renvoie une erreur mais assez cryptique.

_Solution_ :
Il faudrait plutôt afficher un message d'erreur. et ensuite implémenter des API IA de secours

Dans le cas ou le PDF est protégé par mdp, le cas n'est pas géré et cela renvoie une exception ou renvoie un texte vide.

_Solution_ : Implémenter la gestion des pdf protégés par mdp

La détection du type de document est limité à 2000 caractères, si le début du pdf est un long sommaire ou des mentions légales, l'IA ne classifiera pas correctement le type

Pour les cas de devis ou bons de commande, cela ressemble beaucoup à une facture et l'IA peut inconsistément le classer comme tel ou non

La valeur renvoyée du montant est quoi qu'il en soit toujours exprimée en euro, il faut gérer les autres monnaies

L'utilisateur peut passer n'importe quel argument en chemin relatif, ça peut etre risqué, il n'y a pas de validation

_Solution_ implémenter une validation robuste du chemin rentré par l'utilisateur
