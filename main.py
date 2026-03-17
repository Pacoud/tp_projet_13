"""
Extracteur de données de formulaires PDF
=========================================
MVP qui lit un fichier PDF, en extrait le texte brut avec PyMuPDF,
détecte automatiquement le type de document (facture ou autre),
puis utilise Azure OpenAI (gpt-4o-mini) avec les Structured Outputs
pour extraire des données structurées validées par Pydantic.
"""

import os
import sys
import time
import hashlib
import datetime

import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI
from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional
import unicodedata

import threading

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.padding import Padding
from rich.live import Live


# =============================================================================
# Initialisation de la console Rich
# =============================================================================

console = Console()


# =============================================================================
# Animation : rectangle à tirets discontinus avec marqueur tournant
# =============================================================================

def _frame_rect(pos: int, label: str, width: int = 22) -> Text:
    """
    Génère un frame ASCII représentant un rectangle à tirets discontinus
    avec un marqueur ● qui avance le long du périmètre (sens horaire).
    """
    inner = width - 2          # nb de positions internes en haut/bas
    total = 2 * inner + 2     # périmètre total (4 coins exclus des segments)
    pos = pos % total

    top = list('╌' * inner)
    bot = list('╌' * inner)
    left  = '╎'
    right = '╎'

    if pos < inner:               # haut, gauche → droite
        top[pos] = '⮞'
    elif pos == inner:             # côté droit
        right = '⮟'
    elif pos < 2 * inner + 1:     # bas, droite → gauche
        bot[inner - 1 - (pos - inner - 1)] = '⮜'
    else:                          # côté gauche
        left = '⮝'

    top_str = '┌' + ''.join(top) + '┐'
    mid_str = left + ' ' * inner  + right + f'  {label}'
    bot_str = '└' + ''.join(bot) + '┘'

    return Text(f"{top_str}\n{mid_str}\n{bot_str}", style="bold red")


def run_with_rect(label: str, func, *args, **kwargs):
    """
    Exécute `func(*args, **kwargs)` dans un thread secondaire
    tout en affichant l'animation rectangle dans le thread principal.
    Relève l'exception si le worker en a levé une.
    """
    result: list = [None]
    error:  list = [None]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    with Live(console=console, refresh_per_second=14, transient=True) as live:
        pos = 0
        while t.is_alive():
            live.update(_frame_rect(pos, label))
            pos += 1
            time.sleep(1 / 14)

    t.join()
    if error[0]:
        raise error[0]
    return result[0]


# =============================================================================
# 1. Chargement des variables d'environnement
# =============================================================================

load_dotenv()  # Charge les variables depuis le fichier .env

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Configuration Groq (API de secours) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# Limite de caractères envoyés à l'API pour éviter le dépassement de tokens
LIMITE_CARACTERES = 50000
LIMITE_CARACTERES_GROQ = 25000  # Groq a des limites plus basses

# Fichier de cache local (ne pas commiter)
CACHE_FILE = "cache_analyses.json"


# =============================================================================
# Fonctions de mise en cache
# =============================================================================

def calculer_hash_pdf(chemin_pdf: str) -> str:
    """
    Calcule le hash SHA256 du contenu du fichier PDF.
    Sert d'identifiant unique : si le contenu change, le hash change aussi.
    """
    sha256 = hashlib.sha256()
    with open(chemin_pdf, "rb") as f:
        for bloc in iter(lambda: f.read(65536), b""):
            sha256.update(bloc)
    return sha256.hexdigest()


def charger_cache() -> dict:
    """Charge le fichier cache JSON. Retourne un dict vide si inexistant ou corrompu."""
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def sauvegarder_cache(cache: dict) -> None:
    """Sauvegarde le cache dans le fichier JSON."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except IOError as e:
        console.print(f"[bold yellow]⚠  Impossible de sauvegarder le cache : {e}[/bold yellow]")


# =============================================================================
# 2. Définition des modèles Pydantic
# =============================================================================

class TypeDocument(BaseModel):
    """
    Schéma Pydantic utilisé pour la détection du type de document.
    Distingue trois catégories principales : facture, devis/bon de commande, ou autre.
    """
    est_facture: bool = Field(
        description=(
            "True UNIQUEMENT si le document est une facture commerciale émise APRÈS une vente "
            "(contient un numéro de facture, mention 'facture', montant TTC dû, date d'échéance). "
            "False sinon."
        )
    )
    est_devis_ou_bc: bool = Field(
        default=False,
        description=(
            "True si le document est un devis ou un bon de commande : "
            "document AVANT la vente/livraison (contient 'devis', 'bon de commande', "
            "'offre de prix', 'pro forma', 'purchase order', 'PO', montant estimatif). "
            "False sinon. Ne peut pas être True en même temps que est_facture."
        )
    )
    type_detecte: Optional[str] = Field(
        default=None,
        description=(
            "Type précis du document en français : 'facture', 'devis', 'bon de commande', "
            "'contrat', 'rapport', 'déclaration fiscale', 'bulletin de paie', 'autre'."
        )
    )


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
    devise: Optional[str] = Field(
        default=None,
        description="Devise du montant (ex: 'EUR', 'USD', 'GBP'). Si non précisée, renvoyer null."
    )
    date: Optional[str] = Field(
        default=None,
        description="Date de la facture au format texte (ex: '15/01/2025')."
    )


class DonneesDevisBonCommande(BaseModel):
    """
    Schéma Pydantic pour l'extraction des données d'un devis ou bon de commande.
    Ces documents précèdent la vente et contiennent une offre de prix ou une commande.
    """
    type_document: Optional[str] = Field(
        default=None,
        description="Type précis : 'devis', 'bon de commande', 'offre de prix', 'pro forma', etc."
    )
    numero_reference: Optional[str] = Field(
        default=None,
        description="Numéro de référence du devis ou bon de commande (ex: 'DEV-2024-042', 'BC-001')."
    )
    nom_client: Optional[str] = Field(
        default=None,
        description="Nom du client ou de l'acheteur."
    )
    nom_fournisseur: Optional[str] = Field(
        default=None,
        description="Nom du fournisseur ou de l'émetteur du document."
    )
    montant_total: Optional[float] = Field(
        default=None,
        description="Montant total estimé ou commandé (TTC), en valeur numérique."
    )
    devise: Optional[str] = Field(
        default=None,
        description="Devise du montant (ex: 'EUR', 'USD', 'GBP'). Si non précisée, renvoyer null."
    )
    date_emission: Optional[str] = Field(
        default=None,
        description="Date d'émission du document."
    )
    date_validite: Optional[str] = Field(
        default=None,
        description="Date de validité ou d'expiration du devis (si mentionnée)."
    )
    description_prestations: Optional[str] = Field(
        default=None,
        description="Résumé des produits, services ou prestations listés dans le document."
    )


class AnalyseGenerique(BaseModel):
    """
    Schéma Pydantic pour l'analyse de tout document qui n'est ni une facture
    ni un devis/bon de commande : rapport, contrat, déclaration, etc.
    """

    titre_document: Optional[str] = Field(
        default=None,
        description="Titre ou intitulé principal du document."
    )
    auteur_ou_organisme: Optional[str] = Field(
        default=None,
        description=(
            "PRIORITÉ : Nom de l'auteur, du signataire, de l'organisme ou de l'entité émettrice du document. "
            "Cherche des mentions comme 'rédigé par', 'auteur', 'signataire', un en-tête, un logo textuel, "
            "ou une signature en bas de document. Renvoie null uniquement si vraiment introuvable."
        )
    )
    date: Optional[str] = Field(
        default=None,
        description=(
            "PRIORITÉ : Date d'émission ou de création du document. "
            "Cherche des mentions comme 'le', 'date', 'émis le', 'fait à', 'en-tête daté', "
            "ou toute date en début ou fin de document. Format texte (ex: '15/01/2025'). "
            "Renvoie null uniquement si vraiment introuvable."
        )
    )
    resume_contenu: Optional[str] = Field(
        default=None,
        description="Résumé concis (2-4 phrases) du contenu principal du document."
    )
    points_cles: Optional[str] = Field(
        default=None,
        description=(
            "Points clés ou informations importantes extraits du document "
            "(montants, noms, références, obligations, etc.), listés en texte libre."
        )
    )


# =============================================================================
# 3. Fonction d'extraction du texte brut depuis un PDF
# =============================================================================

def extraire_texte_pdf(chemin_pdf: str) -> str:
    """
    Lit un fichier PDF avec PyMuPDF et retourne le texte brut de toutes
    les pages concaténées.
    Gère les cas : fichier introuvable, extension invalide, PDF protégé,
    et PDF scanné (image sans texte).
    """
    # --- Validation du chemin ---
    if not chemin_pdf.lower().endswith('.pdf'):
        raise ValueError(f"Le fichier doit avoir l'extension .pdf : '{chemin_pdf}'")

    if not os.path.isfile(chemin_pdf):
        raise FileNotFoundError(f"Le fichier PDF est introuvable : '{chemin_pdf}'")

    try:
        document = fitz.open(chemin_pdf)
    except Exception as e:
        raise RuntimeError(f"Impossible d'ouvrir le fichier PDF : {e}")

    # --- Détection d'un PDF protégé par mot de passe ---
    if document.needs_pass:
        document.close()
        raise PermissionError(
            "Le fichier PDF est protégé par un mot de passe. "
            "Veuillez fournir un PDF non protégé."
        )

    texte_complet = ""
    for numero_page, page in enumerate(document, start=1):
        texte_page = page.get_text("text")
        texte_complet += f"\n--- Page {numero_page} ---\n{texte_page}"

    document.close()

    # --- Nettoyage du texte extrait ---
    # Supprime les octets nuls parasites (présents dans les PDFs encodés en UTF-16)
    # qui provoquent des séquences illisibles comme '0e9' à la place de 'é'.
    texte_complet = texte_complet.replace('\x00', '')
    # Normalisation Unicode NFC : recompose les caractères décomposés
    texte_complet = unicodedata.normalize('NFC', texte_complet)

    # --- Détection d'un PDF scanné (image sans texte exploitable) ---
    if not texte_complet.strip():
        raise RuntimeError(
            "Aucun texte n'a pu être extrait du PDF. "
            "Le fichier est probablement un scan (image). "
            "Utilisez un outil d'OCR pour convertir les images en texte."
        )

    return texte_complet


# =============================================================================
# 4. Fonction de détection du type de document
# =============================================================================

def detecter_type_document(texte_brut: str) -> TypeDocument:
    """
    Effectue un premier appel léger à l'API Azure OpenAI pour déterminer
    si le document est une facture ou un autre type de document.
    Retourne un objet TypeDocument avec un booléen et le type détecté.
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    prompt_detection = (
        "Tu es un assistant expert en classification de documents commerciaux et administratifs. "
        "On te fournit le texte brut extrait d'un PDF. "
        "Tu dois classifier précisément ce document selon les règles ci-dessous.\n\n"
        "RÈGLES DE CLASSIFICATION :\n"
        "1. FACTURE (est_facture=true, est_devis_ou_bc=false) :\n"
        "   - Document émis APRÈS une vente ou prestation réalisée.\n"
        "   - Contient : 'FACTURE', numéro de facture, date d'échéance ou 'À payer avant le', "
        "     montant TTC exigible.\n"
        "   - Le paiement est DÛ (obligation de règlement).\n\n"
        "2. DEVIS ou BON DE COMMANDE (est_facture=false, est_devis_ou_bc=true) :\n"
        "   - Document émis AVANT la vente/livraison.\n"
        "   - Contient les mots : 'DEVIS', 'BON DE COMMANDE', 'OFFRE DE PRIX', 'PRO FORMA', "
        "     'PURCHASE ORDER', 'PO', 'QUOTATION', ou une date de validité de l'offre.\n"
        "   - Le montant est une ESTIMATION ou une COMMANDE, pas encore dû.\n"
        "   - IMPORTANT : Un devis ressemble à une facture (lignes de produits, montants) "
        "     mais il n'y a PAS de mention 'facture' ni d'obligation de paiement immédiate.\n\n"
        "3. AUTRE (est_facture=false, est_devis_ou_bc=false) :\n"
        "   - Tout autre document : contrat, rapport, déclaration fiscale, bulletin de paie, etc.\n\n"
        "Pour type_detecte, sois précis : 'facture', 'devis', 'bon de commande', "
        "'contrat', 'rapport', 'déclaration fiscale', 'bulletin de paie', 'autre'."
    )

    try:
        completion = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": prompt_detection},
                {"role": "user", "content": (
                    "Voici le texte brut extrait d'un PDF. "
                    "Classifie ce document :\n\n"
                    # On envoie le début ET la fin du texte pour une meilleure détection
                    f"{texte_brut[:3000]}\n[...]\n{texte_brut[-1000:]}"
                )},
            ],
            response_format=TypeDocument,
            temperature=0,
        )

        resultat = completion.choices[0].message.parsed
        if resultat is None:
            # En cas d'échec, on suppose que ce n'est pas une facture
            return TypeDocument(est_facture=False, type_detecte="inconnu")
        return resultat

    except Exception as e:
        # --- Fallback Groq pour la détection ---
        console.print(f"[bold yellow]⚠  Azure échoué pour la détection, basculement vers Groq...[/bold yellow]")
        return _detecter_type_document_groq(texte_brut)


# =============================================================================
# 5. Fonction d'extraction structurée via Azure OpenAI (Structured Outputs)
# =============================================================================

def extraire_donnees_structurees(texte_brut: str, type_doc: TypeDocument):
    """
    Envoie le texte brut extrait du PDF à l'API Azure OpenAI et utilise
    les Structured Outputs pour obtenir un objet Pydantic validé.

    - Si est_facture=True         : retourne un objet DonneesFacture
    - Si est_devis_ou_bc=True     : retourne un objet DonneesDevisBonCommande
    - Sinon                       : retourne un objet AnalyseGenerique
    """
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME]):
        raise ValueError(
            "Les variables d'environnement Azure OpenAI ne sont pas toutes "
            "configurées. Vérifiez votre fichier .env."
        )

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    texte_a_envoyer = texte_brut[:LIMITE_CARACTERES]

    if type_doc.est_facture:
        # --- Chemin FACTURE ---
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'extraction de données à partir "
            "de documents textuels. On te fournit le texte brut extrait d'un PDF "
            "de facture. Tu dois extraire les informations demandées.\n\n"
            "RÈGLES IMPORTANTES :\n"
            "- Extrais UNIQUEMENT les informations explicitement présentes dans le texte.\n"
            "- Si une information N'EST PAS trouvée dans le texte, renvoie null.\n"
            "- N'INVENTE JAMAIS de données. Aucune hallucination n'est tolérée.\n"
            "- Pour le montant, renvoie uniquement la valeur numérique (sans symbole monétaire).\n"
        )
        message_utilisateur = (
            "Voici le texte brut extrait d'une facture PDF. "
            "Extrais les données demandées :\n\n"
            f"{texte_a_envoyer}"
        )
        schema = DonneesFacture

    elif type_doc.est_devis_ou_bc:
        # --- Chemin DEVIS / BON DE COMMANDE ---
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'extraction de données à partir "
            "de devis et bons de commande. On te fournit le texte brut extrait d'un PDF. "
            "Tu dois extraire les informations demandées.\n\n"
            "RÈGLES IMPORTANTES :\n"
            "- Extrais UNIQUEMENT les informations explicitement présentes dans le texte.\n"
            "- Si une information N'EST PAS trouvée dans le texte, renvoie null.\n"
            "- N'INVENTE JAMAIS de données. Aucune hallucination n'est tolérée.\n"
            "- Pour le montant, renvoie uniquement la valeur numérique (sans symbole monétaire).\n"
            "- Pour type_document, indique précisément : 'devis', 'bon de commande', "
            "  'offre de prix', 'pro forma', etc.\n"
        )
        message_utilisateur = (
            "Voici le texte brut extrait d'un devis ou bon de commande PDF. "
            "Extrais les données demandées :\n\n"
            f"{texte_a_envoyer}"
        )
        schema = DonneesDevisBonCommande

    else:
        # --- Chemin GÉNÉRIQUE ---
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'analyse de documents textuels. "
            "On te fournit le texte brut extrait d'un PDF qui n'est pas une facture "
            "ni un devis. Tu dois extraire les informations clés de ce document.\n\n"
            "PRIORITÉS ABSOLUES (cherche dans tout le document) :\n"
            "1. auteur_ou_organisme : cherche l'auteur, le signataire, l'organisme émetteur, "
            "   un en-tête, un logo textuel ou une signature. Renvoie null SEULEMENT si introuvable.\n"
            "2. date : cherche la date d'émission, de création ou de signature du document "
            "   (en-tête, bas de page, formule 'fait à...', 'le...'). Renvoie null SEULEMENT si introuvable.\n\n"
            "RÈGLES IMPORTANTES :\n"
            "- Extrais UNIQUEMENT les informations explicitement présentes dans le texte.\n"
            "- Si une information N'EST PAS trouvée dans le texte, renvoie null.\n"
            "- N'INVENTE JAMAIS de données. Aucune hallucination n'est tolérée.\n"
            "- Pour le résumé et les points clés, sois concis et factuel.\n"
        )
        message_utilisateur = (
            "Voici le texte brut extrait d'un document PDF. "
            "Extrais les informations clés :\n\n"
            f"{texte_a_envoyer}"
        )
        schema = AnalyseGenerique

    try:
        completion = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": message_utilisateur},
            ],
            response_format=schema,
            temperature=0,
        )

        resultat = completion.choices[0].message.parsed

        if resultat is None:
            raise RuntimeError(
                "L'API a renvoyé une réponse vide. Le modèle n'a pas pu "
                "extraire les données au format attendu."
            )

        return resultat

    except Exception as e:
        # --- Fallback Groq pour l'extraction ---
        console.print(f"[bold yellow]⚠  Azure échoué pour l'extraction, basculement vers Groq...[/bold yellow]")
        return _extraire_donnees_groq(texte_brut, type_doc)


# =============================================================================
# 6. Fonctions de secours Groq (Llama 3)
# =============================================================================

def _get_groq_client() -> Groq:
    """Crée et retourne un client Groq. Lève une erreur si la clé n'est pas configurée."""
    if not GROQ_API_KEY or GROQ_API_KEY == "VOTRE_CLE_GROQ_ICI":
        raise RuntimeError(
            "L'API Azure OpenAI a échoué et l'API de secours Groq n'est pas configurée. "
            "Ajoutez votre GROQ_API_KEY dans le fichier .env "
            "(obtenez-la sur https://console.groq.com/keys)."
        )
    return Groq(api_key=GROQ_API_KEY)


def _detecter_type_document_groq(texte_brut: str) -> TypeDocument:
    """
    Fallback : détecte le type de document via Groq / Llama 3.
    Utilise le mode JSON et parse manuellement le résultat.
    """
    client = _get_groq_client()

    prompt_detection = (
        "Tu es un assistant expert en classification de documents commerciaux. "
        "Classifie le document en JSON selon ces règles STRICTES :\n\n"
        "1. FACTURE : est_facture=true, est_devis_ou_bc=false\n"
        "   → Document post-vente avec mention 'FACTURE', obligation de paiement, date d'échéance.\n\n"
        "2. DEVIS/BON DE COMMANDE : est_facture=false, est_devis_ou_bc=true\n"
        "   → Document pré-vente avec mots 'DEVIS', 'BON DE COMMANDE', 'OFFRE', 'PRO FORMA', "
        "     'PURCHASE ORDER', ou date de validité de l'offre. Montant estimé, pas encore dû.\n\n"
        "3. AUTRE : est_facture=false, est_devis_ou_bc=false\n\n"
        "Réponds UNIQUEMENT en JSON : "
        '{"est_facture": bool, "est_devis_ou_bc": bool, "type_detecte": "facture|devis|bon de commande|autre"}'
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_detection},
                {"role": "user", "content": f"Classifie ce document :\n\n{texte_brut[:3000]}\n[...]\n{texte_brut[-1000:]}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        contenu = completion.choices[0].message.content
        data = json.loads(contenu)
        return TypeDocument(
            est_facture=data.get("est_facture", False),
            est_devis_ou_bc=data.get("est_devis_ou_bc", False),
            type_detecte=data.get("type_detecte", "inconnu"),
        )
    except Exception as e:
        raise RuntimeError(
            f"Échec sur Azure ET sur Groq pour la détection du type : {e}"
        )


def _extraire_donnees_groq(texte_brut: str, type_doc: TypeDocument):
    """
    Fallback : extrait les données structurées via Groq / Llama 3.
    Utilise le mode JSON et parse manuellement vers le modèle Pydantic approprié.
    """
    client = _get_groq_client()

    if type_doc.est_facture:
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'extraction de données de factures. "
            "Extrais les informations demandées du texte fourni.\n\n"
            "Réponds UNIQUEMENT en JSON avec ce format exact :\n"
            '{"nom_client": "...", "email_client": "...", "montant_total": 123.45, '
            '"devise": "EUR", "date": "..."}\n\n'
            "RÈGLES : extrais uniquement ce qui est présent, mets null si absent, n'invente rien."
        )
        schema_class = DonneesFacture
    elif type_doc.est_devis_ou_bc:
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'extraction de données de devis et bons de commande. "
            "Extrais les informations demandées du texte fourni.\n\n"
            "Réponds UNIQUEMENT en JSON avec ce format exact :\n"
            '{"type_document": "devis", "numero_reference": "...", "nom_client": "...", '
            '"nom_fournisseur": "...", "montant_total": 123.45, "devise": "EUR", '
            '"date_emission": "...", "date_validite": "...", "description_prestations": "..."}\n\n'
            "RÈGLES : extrais uniquement ce qui est présent, mets null si absent, n'invente rien."
        )
        schema_class = DonneesDevisBonCommande
    else:
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'analyse de documents. "
            "Extrais les informations clés du texte fourni.\n\n"
            "Réponds UNIQUEMENT en JSON avec ce format exact :\n"
            '{"titre_document": "...", "auteur_ou_organisme": "...", "date": "...", '
            '"resume_contenu": "...", "points_cles": "..."}\n\n'
            "PRIORITÉS : cherche en premier l'auteur/organisme émetteur ET la date d'émission "
            "dans tout le document (en-têtes, signatures, bas de page). "
            "Mets null uniquement si vraiment introuvable, n'invente rien."
        )
        schema_class = AnalyseGenerique

    # Troncature adaptée à Groq (contexte plus petit)
    texte_a_envoyer = texte_brut[:LIMITE_CARACTERES_GROQ]

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": f"Voici le texte du document :\n\n{texte_a_envoyer}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        contenu = completion.choices[0].message.content
        data = json.loads(contenu)
        return schema_class(**data)

    except Exception as e:
        raise RuntimeError(
            f"Échec sur Azure ET sur Groq pour l'extraction des données : {e}"
        )


# =============================================================================
# 7. Affichage des résultats dans un tableau Rich
# =============================================================================

def afficher_resultats(donnees, chemin_pdf: str, type_doc: TypeDocument):
    """
    Affiche les données extraites dans un joli tableau Rich.
    L'affichage s'adapte selon le type de document : facture, devis/BC, ou générique.
    """

    console.print()
    console.print(Rule("[bold green]Résultats de l'extraction[/bold green]", style="green"))
    console.print()

    table = Table(
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        show_lines=True,
        expand=False,
        min_width=55,
    )

    table.add_column("Champ", style="bold white", justify="left", no_wrap=True)
    table.add_column("Valeur extraite", style="bright_white", justify="left")

    def val(v):
        return str(v) if v is not None else "[dim italic]Non trouvé[/dim italic]"

    def montant_str(montant, devise):
        if montant is None:
            return "[dim italic]Non trouvé[/dim italic]"
        symbole = {"EUR": "€", "USD": "$", "GBP": "£"}.get(
            (devise or "").upper(), devise or "€"
        )
        return f"[bold green]{montant:,.2f} {symbole}[/bold green]"

    if type_doc.est_facture:
        # --- Affichage spécifique FACTURE ---
        table.add_row("👤  Nom client",   val(donnees.nom_client))
        table.add_row("✉  E-mail",        val(donnees.email_client))
        table.add_row("💶  Montant TTC",  montant_str(donnees.montant_total, donnees.devise))
        table.add_row("📅  Date",         val(donnees.date))

    elif type_doc.est_devis_ou_bc:
        # --- Affichage spécifique DEVIS / BON DE COMMANDE ---
        table.add_row("📋  Type",               val(donnees.type_document))
        table.add_row("🔢  Référence",          val(donnees.numero_reference))
        table.add_row("👤  Client",             val(donnees.nom_client))
        table.add_row("🏢  Fournisseur",        val(donnees.nom_fournisseur))
        table.add_row("💰  Montant estimé",     montant_str(donnees.montant_total, donnees.devise))
        table.add_row("📅  Date d'émission",    val(donnees.date_emission))
        table.add_row("⏳  Validité jusqu'au",  val(donnees.date_validite))
        table.add_row("📝  Prestations",        val(donnees.description_prestations))

    else:
        # --- Affichage générique ---
        table.add_row("📄  Titre",              val(donnees.titre_document))
        table.add_row("🏢  Auteur / Organisme", val(donnees.auteur_ou_organisme))
        table.add_row("📅  Date",               val(donnees.date))
        table.add_row("📝  Résumé",             val(donnees.resume_contenu))
        table.add_row("🔑  Points clés",        val(donnees.points_cles))

    console.print(Padding(table, (0, 4)))
    console.print()

    console.print(
        Panel(
            f"[bold green]✅  Analyse terminée avec succès[/bold green]\n"
            f"[dim]Fichier traité :[/dim] [cyan]{chemin_pdf}[/cyan]",
            border_style="green",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()


# =============================================================================
# 8. Point d'entrée principal
# =============================================================================

def main():
    """Point d'entrée du script. Orchestre l'extraction de bout en bout."""

    # --- Bannière de démarrage ---
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold bright_white]PDF  ➜  Données Structurées[/bold bright_white]\n"
                "[dim]Extraction intelligente via Azure OpenAI · gpt-4o[/dim]",
                justify="center",
            ),
            border_style="bright_blue",
            padding=(1, 6),
        )
    )
    console.print()

    # --- Vérification des arguments ---
    if len(sys.argv) < 2:
        console.print("[bold red]❌  Usage :[/bold red] python main.py <chemin_vers_le_fichier.pdf>")
        console.print("[dim]Exemple  : python main.py FACTURE.pdf[/dim]")
        sys.exit(1)

    chemin_pdf = sys.argv[1]

    # --- Vérification anticipée des variables d'environnement Azure ---
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME]):
        console.print(
            "[bold red]❌  Erreur de configuration :[/bold red] "
            "Les variables d'environnement Azure OpenAI ne sont pas toutes "
            "configurées. Vérifiez votre fichier .env."
        )
        sys.exit(1)

    # --- Vérification préliminaire du fichier (avant calcul hash) ---
    # cache et hash_pdf initialisés ici pour rester en scope jusqu'à la fin de main()
    cache: dict = {}
    hash_pdf: str = ""

    if not chemin_pdf.lower().endswith('.pdf') or not os.path.isfile(chemin_pdf):
        # On laisse extraire_texte_pdf gérer l'erreur détaillée
        pass
    else:
        # --- Vérification dans le cache ---
        cache = charger_cache()
        hash_pdf = calculer_hash_pdf(chemin_pdf)

        if hash_pdf in cache:
            entree = cache[hash_pdf]
            console.print(
                Panel(
                    f"[bold green]📦  Résultat depuis le cache[/bold green]\n"
                    f"[dim]Ce fichier a déjà été analysé le [/dim]"
                    f"[cyan]{entree.get('date_analyse', '?')}[/cyan]\n"
                    f"[dim]Hash SHA256 :[/dim] [dim]{hash_pdf[:16]}...[/dim]\n"
                    f"[dim]Aucun appel API effectué (0 token consommé).[/dim]",
                    border_style="green",
                    expand=False,
                    padding=(0, 2),
                )
            )
            console.print()

            # Reconstruction des objets Pydantic depuis le cache
            type_doc_cache = TypeDocument(**entree["type_doc"])
            if type_doc_cache.est_facture:
                donnees_cache = DonneesFacture(**entree["donnees"])
            elif type_doc_cache.est_devis_ou_bc:
                donnees_cache = DonneesDevisBonCommande(**entree["donnees"])
            else:
                donnees_cache = AnalyseGenerique(**entree["donnees"])

            afficher_resultats(donnees_cache, chemin_pdf, type_doc_cache)
            return  # On quitte main() ici, le travail est fait

    # --- Étape 1 : Extraction du texte brut du PDF ---
    try:
        texte_brut = run_with_rect(
            f"Lecture du PDF  {chemin_pdf}",
            extraire_texte_pdf, chemin_pdf
        )
    except FileNotFoundError as e:
        console.print(f"\n[bold red]❌  Erreur :[/bold red] {e}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"\n[bold red]❌  Erreur :[/bold red] {e}")
        sys.exit(1)
    except PermissionError as e:
        console.print(f"\n[bold red]🔒  Erreur :[/bold red] {e}")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌  Erreur lecture PDF :[/bold red] {e}")
        sys.exit(1)

    console.print(
        f"[bold green]✔[/bold green]  Texte extrait — "
        f"[cyan]{len(texte_brut)}[/cyan] caractères sur [cyan]{chemin_pdf}[/cyan]"
    )

    # --- Avertissement si le texte est très long (sera tronqué) ---
    if len(texte_brut) > LIMITE_CARACTERES:
        console.print(
            f"[bold yellow]⚠  Attention :[/bold yellow] Le texte fait "
            f"[cyan]{len(texte_brut)}[/cyan] caractères, il sera tronqué à "
            f"[cyan]{LIMITE_CARACTERES}[/cyan] pour l'envoi à l'API."
        )

    # --- Aperçu du texte extrait ---
    apercu = texte_brut[:400].strip().replace("\n", " ")
    if apercu:
        console.print(
            Panel(
                f"[dim]{apercu}…[/dim]",
                title="[bold]Aperçu du texte extrait[/bold]",
                border_style="dim",
                padding=(0, 2),
            )
        )
    console.print()

    # --- Étape 2 : Détection du type de document ---
    try:
        type_doc = run_with_rect(
            "Détection du type de document",
            detecter_type_document, texte_brut
        )
    except RuntimeError as e:
        console.print(f"\n[bold red]❌  Erreur détection :[/bold red] {e}")
        sys.exit(1)

    # Affichage du type détecté
    if type_doc.est_facture:
        badge = "[bold green]🧾  Facture[/bold green]"
    elif type_doc.est_devis_ou_bc:
        badge = f"[bold cyan]📋  {type_doc.type_detecte or 'Devis / Bon de commande'}[/bold cyan]"
    else:
        badge = f"[bold yellow]📂  {type_doc.type_detecte or 'Document générique'}[/bold yellow]"

    console.print(
        Panel(
            f"Type de document détecté : {badge}",
            border_style="dim",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    # --- Étape 3 : Extraction structurée via Azure OpenAI ---
    try:
        donnees = run_with_rect(
            "Analyse par Azure OpenAI  (gpt-4o)",
            extraire_donnees_structurees, texte_brut, type_doc
        )
    except ValueError as e:
        console.print(f"\n[bold red]❌  Erreur de configuration :[/bold red] {e}")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌  Erreur API :[/bold red] {e}")
        sys.exit(1)

    console.print("[bold green]✔[/bold green]  Réponse reçue de l'API Azure OpenAI")

    # --- Sauvegarde dans le cache ---
    try:
        cache[hash_pdf] = {
            "chemin_pdf": chemin_pdf,
            "date_analyse": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type_doc": type_doc.model_dump(),
            "donnees": donnees.model_dump(),
        }
        sauvegarder_cache(cache)
        console.print("[dim]💾  Résultat sauvegardé dans le cache (prochaine analyse instantanée).[/dim]")
    except Exception as e:
        console.print(f"[bold yellow]⚠  Cache non sauvegardé : {e}[/bold yellow]")

    # --- Étape 4 : Affichage des résultats ---
    afficher_resultats(donnees, chemin_pdf, type_doc)


if __name__ == "__main__":
    main()
