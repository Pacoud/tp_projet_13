"""
Extracteur de données de formulaires PDF
=========================================
MVP qui lit un fichier PDF, en extrait le texte brut avec PyMuPDF,
puis utilise Azure OpenAI (gpt-4o-mini) avec les Structured Outputs
pour extraire des données structurées validées par Pydantic.
"""

import os
import sys

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Optional


# =============================================================================
# 1. Chargement des variables d'environnement
# =============================================================================

load_dotenv()  # Charge les variables depuis le fichier .env

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# =============================================================================
# 2. Définition du modèle Pydantic (schéma des données à extraire)
# =============================================================================

class DonneesFacture(BaseModel):
    """
    Schéma Pydantic représentant les données extraites d'une facture.

    Tous les champs sont Optional afin d'éviter les hallucinations :
    si l'IA ne trouve pas une information dans le document, elle doit
    renvoyer None plutôt que d'inventer une valeur.
    """

    nom_client: Optional[str] = Field(
        default=None,
        description="Nom complet du client mentionné sur la facture."
    )
    email_client: Optional[str] = Field(
        default=None,
        description="Adresse e-mail du client, si elle est présente."
    )
    montant_total: Optional[float] = Field(
        default=None,
        description="Montant total TTC de la facture, en nombre décimal."
    )
    date: Optional[str] = Field(
        default=None,
        description="Date de la facture au format texte (ex: '15/01/2025')."
    )


# =============================================================================
# 3. Fonction d'extraction du texte brut depuis un PDF
# =============================================================================

def extraire_texte_pdf(chemin_pdf: str) -> str:
    """
    Lit un fichier PDF avec PyMuPDF et retourne le texte brut de toutes
    les pages concaténées.

    Args:
        chemin_pdf: Chemin absolu ou relatif vers le fichier PDF.

    Returns:
        Le texte brut extrait du PDF.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas au chemin indiqué.
        RuntimeError: Si PyMuPDF ne parvient pas à ouvrir ou lire le fichier.
    """
    # Vérification de l'existence du fichier
    if not os.path.isfile(chemin_pdf):
        raise FileNotFoundError(f"Le fichier PDF est introuvable : '{chemin_pdf}'")

    try:
        document = fitz.open(chemin_pdf)
    except Exception as e:
        raise RuntimeError(f"Impossible d'ouvrir le fichier PDF : {e}")

    texte_complet = ""
    for numero_page, page in enumerate(document, start=1):
        texte_page = page.get_text("text")
        texte_complet += f"\n--- Page {numero_page} ---\n{texte_page}"

    document.close()

    if not texte_complet.strip():
        print("⚠️  Attention : Aucun texte n'a été extrait du PDF. "
              "Le fichier est peut-être un scan (image).")

    return texte_complet


# =============================================================================
# 4. Fonction d'extraction structurée via Azure OpenAI (Structured Outputs)
# =============================================================================

def extraire_donnees_structurees(texte_brut: str) -> DonneesFacture:
    """
    Envoie le texte brut extrait du PDF à l'API Azure OpenAI et utilise
    les Structured Outputs (client.beta.chat.completions.parse) pour
    obtenir un objet Pydantic validé.

    Args:
        texte_brut: Le texte brut extrait du PDF.

    Returns:
        Un objet DonneesFacture avec les champs remplis (ou None si absent).

    Raises:
        ValueError: Si les variables d'environnement Azure ne sont pas configurées.
        RuntimeError: Si l'API renvoie une erreur ou un résultat inattendu.
    """
    # Vérification de la configuration Azure
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME]):
        raise ValueError(
            "Les variables d'environnement Azure OpenAI ne sont pas toutes "
            "configurées. Vérifiez votre fichier .env "
            "(voir .env.example pour la liste des variables requises)."
        )

    # Initialisation du client Azure OpenAI
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Prompt système qui guide l'IA dans l'extraction
    prompt_systeme = (
        "Tu es un assistant spécialisé dans l'extraction de données à partir "
        "de documents textuels. On te fournit le texte brut extrait d'un PDF "
        "de facture. Tu dois extraire les informations demandées.\n\n"
        "RÈGLES IMPORTANTES :\n"
        "- Extrais UNIQUEMENT les informations explicitement présentes dans le texte.\n"
        "- Si une information N'EST PAS trouvée dans le texte, renvoie null.\n"
        "- N'INVENTE JAMAIS de données. Aucune hallucination n'est tolérée.\n"
        "- Pour le montant, renvoie uniquement la valeur numérique (sans symbole €).\n"
    )

    try:
        # Appel à l'API avec Structured Outputs (parse + Pydantic)
        completion = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": (
                    "Voici le texte brut extrait d'une facture PDF. "
                    "Extrais les données demandées :\n\n"
                    f"{texte_brut}"
                )},
            ],
            response_format=DonneesFacture,
            temperature=0,  # Déterminisme maximal pour l'extraction
        )

        # Récupération de l'objet Pydantic parsé
        resultat = completion.choices[0].message.parsed

        if resultat is None:
            raise RuntimeError(
                "L'API a renvoyé une réponse vide. Le modèle n'a pas pu "
                "extraire les données au format attendu."
            )

        return resultat

    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'appel à l'API Azure OpenAI : {e}")


# =============================================================================
# 5. Point d'entrée principal
# =============================================================================

def main():
    """Point d'entrée du script. Orchestre l'extraction de bout en bout."""

    # Récupération du chemin du PDF depuis les arguments de la ligne de commande
    if len(sys.argv) < 2:
        print("Usage : python main.py <chemin_vers_le_fichier.pdf>")
        print("Exemple : python main.py facture_exemple.pdf")
        sys.exit(1)

    chemin_pdf = sys.argv[1]

    # --- Étape 1 : Extraction du texte brut du PDF ---
    print(f"📄 Lecture du fichier PDF : '{chemin_pdf}'...")
    try:
        texte_brut = extraire_texte_pdf(chemin_pdf)
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Erreur lors de la lecture du PDF : {e}")
        sys.exit(1)

    print(f"✅ Texte extrait avec succès ({len(texte_brut)} caractères).\n")

    # Affichage d'un aperçu du texte extrait (pour le débogage)
    print("--- Aperçu du texte extrait (500 premiers caractères) ---")
    print(texte_brut[:500])
    print("--- Fin de l'aperçu ---\n")

    # --- Étape 2 : Extraction structurée via Azure OpenAI ---
    print("🤖 Envoi du texte à Azure OpenAI pour extraction structurée...")
    try:
        donnees = extraire_donnees_structurees(texte_brut)
    except ValueError as e:
        print(f"❌ Erreur de configuration : {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Erreur API : {e}")
        sys.exit(1)

    # --- Étape 3 : Affichage du résultat JSON validé ---
    print("✅ Extraction réussie ! Voici les données structurées :\n")
    print(donnees.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
