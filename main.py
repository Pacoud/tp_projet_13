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

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI
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

# Limite de caractères envoyés à l'API pour éviter le dépassement de tokens
LIMITE_CARACTERES = 50000


# =============================================================================
# 2. Définition des modèles Pydantic
# =============================================================================

class TypeDocument(BaseModel):
    """
    Schéma Pydantic utilisé pour la détection du type de document.
    Un premier appel API léger détermine si le PDF est une facture ou non.
    """
    est_facture: bool = Field(
        description="True si le document est une facture commerciale, False sinon."
    )
    type_detecte: Optional[str] = Field(
        default=None,
        description=(
            "Type du document détecté en français, ex: 'facture', 'contrat', "
            "'rapport', 'déclaration fiscale', 'bulletin de paie', 'autre'."
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


class AnalyseGenerique(BaseModel):
    """
    Schéma Pydantic pour l'analyse de tout document qui n'est pas une facture.

    Les champs couvrent les informations généralement présentes dans
    n'importe quel document textuel (rapport, contrat, déclaration, etc.).
    """

    titre_document: Optional[str] = Field(
        default=None,
        description="Titre ou intitulé principal du document."
    )
    auteur_ou_organisme: Optional[str] = Field(
        default=None,
        description="Nom de l'auteur, de l'organisme ou de l'entité émettrice du document."
    )
    date: Optional[str] = Field(
        default=None,
        description="Date mentionnée dans le document (date d'émission, de signature, etc.)."
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
        "Tu es un assistant expert en classification de documents. "
        "On te fournit le texte brut extrait d'un PDF. "
        "Ta seule mission est de déterminer si ce document est une facture commerciale ou non, "
        "et d'indiquer quel type de document il s'agit.\n\n"
        "RÈGLES :\n"
        "- Une facture contient généralement : un numéro de facture, un montant TTC/HT, "
        "un client, une date d'échéance, des lignes d'articles ou de services.\n"
        "- Si tu n'es pas sûr, indique est_facture = false.\n"
        "- Pour type_detecte, sois précis : 'facture', 'contrat', 'rapport annuel', "
        "'déclaration fiscale', 'bulletin de paie', 'relevé bancaire', 'autre', etc."
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
        raise RuntimeError(f"Erreur lors de la détection du type de document : {e}")


# =============================================================================
# 5. Fonction d'extraction structurée via Azure OpenAI (Structured Outputs)
# =============================================================================

def extraire_donnees_structurees(texte_brut: str, est_facture: bool):
    """
    Envoie le texte brut extrait du PDF à l'API Azure OpenAI et utilise
    les Structured Outputs pour obtenir un objet Pydantic validé.

    - Si est_facture=True  : retourne un objet DonneesFacture
    - Si est_facture=False : retourne un objet AnalyseGenerique
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

    if est_facture:
        # --- Chemin FACTURE ---
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
        # Troncature si le texte est trop long pour éviter le dépassement de tokens
        texte_a_envoyer = texte_brut[:LIMITE_CARACTERES]
        message_utilisateur = (
            "Voici le texte brut extrait d'une facture PDF. "
            "Extrais les données demandées :\n\n"
            f"{texte_a_envoyer}"
        )
        schema = DonneesFacture

    else:
        # --- Chemin GÉNÉRIQUE ---
        prompt_systeme = (
            "Tu es un assistant spécialisé dans l'analyse de documents textuels. "
            "On te fournit le texte brut extrait d'un PDF qui n'est pas une facture. "
            "Tu dois extraire les informations clés de ce document.\n\n"
            "RÈGLES IMPORTANTES :\n"
            "- Extrais UNIQUEMENT les informations explicitement présentes dans le texte.\n"
            "- Si une information N'EST PAS trouvée dans le texte, renvoie null.\n"
            "- N'INVENTE JAMAIS de données. Aucune hallucination n'est tolérée.\n"
            "- Pour le résumé et les points clés, sois concis et factuel.\n"
        )
        # Troncature si le texte est trop long pour éviter le dépassement de tokens
        texte_a_envoyer = texte_brut[:LIMITE_CARACTERES]
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
        raise RuntimeError(f"Erreur lors de l'appel à l'API Azure OpenAI : {e}")


# =============================================================================
# 6. Affichage des résultats dans un tableau Rich
# =============================================================================

def afficher_resultats(donnees, chemin_pdf: str, est_facture: bool):
    """
    Affiche les données extraites dans un joli tableau Rich.
    L'affichage s'adapte selon que le document est une facture ou non.
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

    if est_facture:
        # --- Affichage spécifique FACTURE ---
        table.add_row("👤  Nom client",   val(donnees.nom_client))
        table.add_row("✉  E-mail",        val(donnees.email_client))
        # Affichage du montant avec la devise détectée par l'IA
        symbole_devise = {"EUR": "€", "USD": "$", "GBP": "£"}.get(
            (donnees.devise or "").upper(), donnees.devise or "€"
        )
        table.add_row("💶  Montant TTC",
                      f"[bold green]{donnees.montant_total:,.2f} {symbole_devise}[/bold green]"
                      if donnees.montant_total is not None
                      else "[dim italic]Non trouvé[/dim italic]")
        table.add_row("📅  Date",         val(donnees.date))

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
# 7. Point d'entrée principal
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
            extraire_donnees_structurees, texte_brut, type_doc.est_facture
        )
    except ValueError as e:
        console.print(f"\n[bold red]❌  Erreur de configuration :[/bold red] {e}")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌  Erreur API :[/bold red] {e}")
        sys.exit(1)

    console.print("[bold green]✔[/bold green]  Réponse reçue de l'API Azure OpenAI")

    # --- Étape 4 : Affichage des résultats ---
    afficher_resultats(donnees, chemin_pdf, type_doc.est_facture)


if __name__ == "__main__":
    main()
