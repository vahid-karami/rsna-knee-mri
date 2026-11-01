"""Rule-based, multilingual labeller for knee MRI reports.

Turns free-text radiology reports into a soft score in [0, 1] per finding.
It is a fast, transparent baseline: scores are checked against the expert
("gold") labels, and the LLM labeller in knee.report_llm is compared with it.

Covered languages: English, German, Dutch, Spanish, French, Italian,
Portuguese, Turkish, Greek. Reports in other languages simply score as
"not mentioned" (the LLM labeller handles them).

Text and patterns are both normalised (accents stripped, lower case, the
micro sign used in some Greek reports mapped to mu), so the vocabulary below
can be written without worrying about diacritics.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config as C


def normalise(text):
    text = unicodedata.normalize("NFKC", text)  # "µ" (micro sign) -> "μ", full-width forms
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return text.lower().replace("ς", "σ")


# --- vocabulary --------------------------------------------------------------

INJURY = (
    r"tear|torn|rupture|ruptur|disrupt|discontinu|insufficien|sprain|lesion|"
    r"riss|einriss|läsion|zerrung|"
    r"scheur|ruptuur|letsel|"
    r"desgarro|rotura|roto|lesión|"
    r"déchirure|lésion|"
    r"lesione|rottura|"
    r"ruptura|lesão|"
    r"y[iı]rt[iı]k|kopma|lezyon|hasar|"  # Turkish
    r"ρηξ|ρηγμ|διακοπ"  # Greek: rupture, tear, discontinuity
)
MEDIAL = r"medial|mediale|interne?|innen|interno|interna|intern|εσω"
LATERAL = r"lateral|laterale|externe?|außen|aussen|externo|externa|extern|εξω"
MENISCUS = r"menisc|menisk|ménisque|menisco|meniscus|μηνισκ"
OA = (
    r"osteoarthrit|osteoartrit|osteoartros|arthros|artros|artroz|gonarthros|gonartros|gonartroz|"
    r"degenerative joint|cartilage loss|chondral loss|knorpelverlust|"
    r"chondropath|condropat|kondropati|chondromalac|condromalac|kondromalaz|"
    r"k[iı]k[iı]rdak (?:incel|kayb|defekt|hasar)|"
    r"οστεοαρθρ|αρθροπαθ|αρθριτιδ|χονδροπαθ|εκφυλιστικ\w* αλλοιωσ\w* (?:του )?χονδρ"
)
PF = (
    r"patellofemoral|patello-femoral|femoropatell|femoro-patell|retropatell|"
    r"patelofemoral|fémoro-patell|femoro-rotule|femororotul|patella|patell|rotule|rótula|rotula|"
    r"επιγονατιδ|μηριαιοεπιγονατιδ"
)

PATTERNS = {
    "ACL": {
        "any": [r"\bacl\b|\blca\b|\bvkb\b|\bπχσ\b|anterior cruciate|vordere[sn]? kreuzband|"
                r"cruzado anterior|croisé antérieur|voorste kruisband|crociato anteriore|"
                r"ön çapraz|on capraz|"
                r"προσθιο\w* χιαστ"],
        "need_injury": True,
    },
    "MCL": {
        "any": [r"\bmcl\b|\blcm\b|\binnenband|medial collateral|medial(es)? seitenband|"
                r"colateral medial|collatéral médial|collatéral interne|mediale collaterale band|"
                r"collaterale mediale|ligamento colateral interno|"
                r"medial kollateral|"
                r"εσω (?:πλαγιο|παραπλευρο)\w* συνδεσμ"],
        "need_injury": True,
    },
    "Medial Meniscus": {"all": [MENISCUS, MEDIAL], "need_injury": True},
    "Lateral Meniscus": {"all": [MENISCUS, LATERAL], "need_injury": True},
    "Medial OA": {"all": [OA, MEDIAL]},
    "Lateral OA": {"all": [OA, LATERAL]},
    "PF OA": {"all": [OA, PF]},
    "Effusion": {
        "any": [r"effusion|erguss|derrame|épanchement|hydrops|"
                r"gewrichtsvocht|vochtcollectie|versamento|intra-?articular fluid|"
                r"efüzyon|efuzyon|eklem (?:ici )?s[iı]v[iı]|s[iı]v[iı] art[iı]s|"
                r"υδραρθρ|συλλογ\w* υγρ|υγρ\w* .{0,40}αρθρωσ"],
    },
    "Synovitis": {"any": [r"synovit|synovialit|sinovit|synovite|συνοβιτιδ|υμενιτιδ"]},
    "Baker's": {
        "any": [r"baker|popliteal cyst|poplitealzyste|quiste popl|kyste popl|cisti popl|"
                r"cisto popl|bakerzyste|bakercyste|popliteal kist|"
                r"ιγνυακ\w* κυστ|κυστ\w* .{0,80}γαστροκνημ"],
    },
    "Contusion": {
        "any": [r"contusion|contusión|contusão|contusione|kontusion|kontüzyon|bone bruise|"
                r"bone marrow o?edema|knochenmarködem|knochenmarkoedem|knochenmark-ödem|"
                r"beenmergoedeem|botkneuzing|edema óseo|edema de medula|"
                r"œdème osseux|oedeme osseux|œdème médullaire|edema osseo|edema midollare|"
                r"kemik ili[gğ]i? ödem|kemik iliginde ödem|"
                r"θλασ|οστικ\w* οιδημα|οιδημα\w* (?:του )?(?:οστικου )?μυελου"],
    },
    "Fracture": {
        "any": [r"fractur|fraktur|fractuur|frattur|fratura|breuk|k[iı]r[iı][gğk]|καταγμ"],
    },
}

NEGATION = re.compile(
    r"\b(?:no|not|without|absent|absence|negative|intact|normal|unremarkable|"
    r"kein|keine|keinen|ohne|nicht|unauffällig|intakt|regelrecht|"
    r"sin|ausencia|íntegro|íntegra|normales|"
    r"pas de|sans|aucun|aucune|"
    r"geen|zonder|niet|normaal|"
    r"sem|ausência|"
    r"non|senza|assenza|integri|"
    r"yok|yoktur|normaldir|dogaldir|olagandir|"
    r"χωρισ|δεν|ουτε|ουδεμια)\b(?!-)"  # (?!-) keeps "non-displaced fracture" positive
    r"|\b(?:φυσιολογικ|ακεραι)"  # Greek stems ("normal", "intact"); \b keeps "αφυσιολογικ" (abnormal) out
    r"|\w+m[ae]d[iı]\b|\w+m[ae]m[ae]kt[ae]d[iı]r",  # Turkish negative verbs: izlenmedi, saptanmamaktadır
    re.IGNORECASE,
)
# "Cannot be excluded" contains a negation word but means "possible".
NOT_EXCLUDED = re.compile(
    r"cannot be excluded|can not be excluded|cannot be ruled out|nicht auszuschließen|nicht ausgeschlossen|"
    r"no se descarta|ne peut être exclu|non si esclude|niet uit te sluiten|"
    r"ekarte edilemez|dislanamaz|δεν αποκλειεται",
    re.IGNORECASE,
)
UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspect\w*|suspicion|likely|"
    r"equivocal|questionable|verdacht|fraglich|möglich|wahrscheinlich|"
    r"posible|sospecha|éventuel\w*|"
    r"mogelijk|verdenking|waarschijnlijk|"
    r"possibile|sospett\w*|"
    r"provável|suspeita|"
    r"şüpheli|olas[iı]|muhtemel\w*)\b"
    r"|\b(?:πιθαν|ενδεχομεν|υποπτ)",
    re.IGNORECASE,
)
MILD_EFFUSION = re.compile(
    r"physiolog|minimal|trace|gering|mínimo|discreto|klein|hafif|az miktar|minim|ελαχιστ|μικρ",
    re.IGNORECASE,
)

IMPRESSION = re.compile(
    r"(impression|conclusion|summary|beurteilung|zusammenfassung|fazit|"
    r"conclusie|samenvatting|indruk|"
    r"conclusión|impresión|"
    r"conclusione|conclusão|impressão|"
    r"sonuç|yorum|kanaat|"
    r"συμπερασμα|γνωματευσ)\s*:?",
    re.IGNORECASE,
)

CLAUSE_SPLIT = re.compile(
    r"[;:\n]|\.(?!\d)|(?<!\d)\.|,|\b(?:but|however|aber|jedoch|pero|mais|maar|ma|mas|ancak|fakat|ενω|αλλα)\b",
    re.IGNORECASE,
)

NOT_MENTIONED = 0.02


def _compile(pattern):
    return re.compile(normalise(pattern), re.IGNORECASE)


def _renormalise(rx):
    return re.compile(normalise(rx.pattern), rx.flags)


NEGATION, NOT_EXCLUDED, UNCERTAIN, MILD_EFFUSION, IMPRESSION, CLAUSE_SPLIT = map(
    _renormalise, (NEGATION, NOT_EXCLUDED, UNCERTAIN, MILD_EFFUSION, IMPRESSION, CLAUSE_SPLIT)
)
_COMPILED = {
    label: {
        "any": [_compile(p) for p in spec.get("any", [])],
        "all": [_compile(p) for p in spec.get("all", [])],
        "need_injury": spec.get("need_injury", False),
    }
    for label, spec in PATTERNS.items()
}
_INJURY = _compile(INJURY)


def _sentences(text):
    # Split on full stops, but not on decimals such as "4.7 cm".
    return [s for s in re.split(r"[;\n]|\.(?!\d)|(?<!\d)\.", text) if s.strip()]


def _clauses(sentence):
    return [c for c in CLAUSE_SPLIT.split(sentence) if c and c.strip()]


def _mentions(spec, text):
    if spec["any"] and not any(p.search(text) for p in spec["any"]):
        return False
    if spec["all"] and not all(p.search(text) for p in spec["all"]):
        return False
    return bool(spec["any"] or spec["all"])


def _clause_score(label, spec, clause):
    if spec["need_injury"] and not _INJURY.search(clause):
        return 0.0  # e.g. "ACL intact" or "medial meniscus normal"
    if NOT_EXCLUDED.search(clause):
        return 0.6
    if NEGATION.search(clause):
        return 0.0
    if UNCERTAIN.search(clause):
        return 0.6
    if label == "Effusion" and MILD_EFFUSION.search(clause):
        return 0.5
    return 1.0


def _score_text(label, text):
    """Return a score, or None when the finding is never mentioned."""
    spec = _COMPILED[label]
    scores = []
    for sentence in _sentences(text):
        if not _mentions(spec, sentence):
            continue
        # Score the clause(s) that carry the mention; fall back to the whole
        # sentence when the vocabulary is split across clauses
        # ("medial meniscus: horizontal tear of the posterior horn").
        hits = [c for c in _clauses(sentence) if _mentions(spec, c)]
        for chunk in hits or [sentence]:
            scores.append(_clause_score(label, spec, chunk))
    return max(scores) if scores else None


def score_report(text, labels=None):
    labels = labels or C.LABELS
    text = normalise(text) if isinstance(text, str) else ""
    m = list(IMPRESSION.finditer(text))
    impression = text[m[-1].end():] if m else ""
    out = {}
    for label in labels:
        if label not in _COMPILED:
            out[label] = np.nan
            continue
        s_imp = _score_text(label, impression) if impression else None
        s_all = _score_text(label, text)
        # The impression is the radiologist's conclusion, so it wins when it
        # talks about the finding.
        s = s_imp if s_imp is not None else s_all
        out[label] = NOT_MENTIONED if s is None else s
    return out


def score_reports(df, labels=None):
    labels = labels or C.LABELS
    rows = [score_report(t, labels) for t in df[C.REPORT_COL].fillna("")]
    out = pd.DataFrame(rows, columns=labels)
    out.insert(0, C.ID_COL, df[C.ID_COL].values)
    return out
