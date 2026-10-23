"""Rule-based, multilingual labeller for knee MRI reports.

Turns free-text radiology reports into a soft score in [0, 1] per finding.
It is a fast, transparent baseline: scores are checked against the expert
("gold") labels, and the LLM labeller in knee.report_llm is compared with it.

Covered languages: English, German, Dutch, Spanish, French, Italian,
Portuguese. Reports in other languages simply score as "not mentioned".
"""

import re

import numpy as np
import pandas as pd

from . import config as C

# --- vocabulary --------------------------------------------------------------

INJURY = (
    r"tear|torn|rupture|ruptur|disrupt|discontinu|insufficien|sprain|lesion|"
    r"riss|einriss|läsion|lasion|zerrung|"
    r"scheur|ruptuur|letsel|"
    r"desgarro|rotura|roto|lesión|"
    r"déchirure|lésion|rupture|"
    r"lesione|rottura|"
    r"ruptura|lesão"
)
MEDIAL = r"medial|mediale|interne?|innen|interno|interna|intern"
LATERAL = r"lateral|laterale|externe?|außen|aussen|externo|externa|extern"
MENISCUS = r"menisc|menisk|ménisque|menisco|meniscus|innenmeniskus|außenmeniskus|aussenmeniskus"
OA = (
    r"osteoarthrit|osteoartrit|osteoartros|arthros|artros|gonarthros|gonartros|"
    r"degenerative joint|cartilage loss|chondral loss|knorpelverlust|"
    r"chondropath|condropat|chondromalac|condromalac"
)
PF = (
    r"patellofemoral|patello-femoral|femoropatell|femoro-patell|retropatell|"
    r"patelofemoral|fémoro-patell|femoro-rotule|femororotul|patella|rotule|rótula|rotula"
)

PATTERNS = {
    "ACL": {
        "any": [r"\bacl\b|\blca\b|\bvkb\b|anterior cruciate|vordere[sn]? kreuzband|"
                r"cruzado anterior|croisé antérieur|voorste kruisband|crociato anteriore"],
        "need_injury": True,
    },
    "MCL": {
        "any": [r"\bmcl\b|\blcm\b|\bmcl\b|\binnenband|medial collateral|medial(es)? seitenband|"
                r"colateral medial|collatéral médial|collatéral interne|mediale collaterale band|"
                r"collaterale mediale|ligamento colateral interno"],
        "need_injury": True,
    },
    "Medial Meniscus": {"all": [MENISCUS, MEDIAL], "need_injury": True},
    "Lateral Meniscus": {"all": [MENISCUS, LATERAL], "need_injury": True},
    "Medial OA": {"all": [OA, MEDIAL]},
    "Lateral OA": {"all": [OA, LATERAL]},
    "PF OA": {"all": [OA, PF]},
    "Effusion": {
        "any": [r"effusion|erguss|derrame|épanchement|epanchement|hydrops|"
                r"gewrichtsvocht|vochtcollectie|versamento|intra-?articular fluid"],
    },
    "Synovitis": {"any": [r"synovit|synovialit|sinovit|synovite"]},
    "Baker's": {
        "any": [r"baker|popliteal cyst|poplitealzyste|quiste popl|kyste popl|cisti popl|"
                r"cisto popl|bakerzyste|bakercyste"],
    },
    "Contusion": {
        "any": [r"contusion|contusión|contusão|contusione|kontusion|bone bruise|"
                r"bone marrow o?edema|knochenmarködem|knochenmarkoedem|knochenmark-ödem|"
                r"beenmergoedeem|botkneuzing|edema óseo|edema oseo|edema de medula|"
                r"œdème osseux|oedeme osseux|œdème médullaire|edema osseo|edema midollare"],
    },
    "Fracture": {
        "any": [r"fractur|fraktur|fractuur|frattur|fratura|breuk"],
    },
}

NEGATION = re.compile(
    r"\b(no|not|without|absent|absence|negative|intact|normal|unremarkable|"
    r"kein|keine|keinen|ohne|nicht|unauffällig|unauffallig|intakt|regelrecht|"
    r"sin|ausencia|íntegro|integro|íntegra|normales?|"
    r"pas de|sans|aucun|aucune|absence|"
    r"geen|zonder|niet|normaal|"
    r"sem|ausência|"
    r"non|senza|assenza|integri)\b(?!-)",  # (?!-) keeps "non-displaced fracture" positive
    re.IGNORECASE,
)
UNCERTAIN = re.compile(
    r"\b(possible|possibly|probable|probably|suspect\w*|suspicion|likely|cannot be excluded|"
    r"equivocal|questionable|verdacht|fraglich|möglich|wahrscheinlich|"
    r"posible|probable|sospecha|"
    r"suspicion|éventuel\w*|"
    r"mogelijk|verdenking|waarschijnlijk|"
    r"possibile|sospett\w*|"
    r"provável|suspeita)\b",
    re.IGNORECASE,
)
MILD_EFFUSION = re.compile(r"physiolog|minimal|trace|gering|mínimo|minimo|discreto|klein", re.IGNORECASE)

IMPRESSION = re.compile(
    r"(impression|conclusion|summary|beurteilung|zusammenfassung|fazit|"
    r"conclusie|samenvatting|indruk|"
    r"conclusión|impresión|conclusión diagnóstica|"
    r"conclusione|conclusão|impressão)\s*:?",
    re.IGNORECASE,
)

CLAUSE_SPLIT = re.compile(r"[.;:\n]|,|\b(but|however|aber|jedoch|pero|mais|maar|ma|mas)\b", re.IGNORECASE)

NOT_MENTIONED = 0.02

_COMPILED = {
    label: {
        "any": [re.compile(p, re.IGNORECASE) for p in spec.get("any", [])],
        "all": [re.compile(p, re.IGNORECASE) for p in spec.get("all", [])],
        "need_injury": spec.get("need_injury", False),
    }
    for label, spec in PATTERNS.items()
}
_INJURY = re.compile(INJURY, re.IGNORECASE)


def _sentences(text):
    return [s for s in re.split(r"[.;\n]", text) if s.strip()]


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
    text = text if isinstance(text, str) else ""
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
