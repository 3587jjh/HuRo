import re
import json
from typing import Dict, List, Optional, Tuple
from lemminflect import getLemma


# Base verbs kept as they are. lemminflect's first lemma for each is another verb
_KEEP_BASE_VERBS = frozenset({"lay", "saw", "bore"})


def _to_base_verb(word: str) -> str:
    """Convert a word to its base verb form using lemminflect."""
    w = word.lower()
    if w in _KEEP_BASE_VERBS:
        return w
    lemmas = getLemma(w, upos="VERB")
    return lemmas[0] if lemmas else w


_NA_PHRASES = frozenset({
    "n/a", "na", "n.a", "n.a.", "n / a",
    "not applicable", "no action", "no clear action",
    "nothing", "nothing visible", "no interaction", "no contact",
    "idle hands", "none", "no manipulation",
})


def _strip_with_suffix(phrase: str) -> str:
    """Strip ' with ...' instrumental suffix from an action phrase.
    'slice carrot with knife' -> 'slice carrot'"""
    idx = phrase.rfind(" with ")
    return phrase[:idx].rstrip() if idx >= 0 else phrase


_VAGUE_OBJECTS = frozenset({
    "item", "items", "thing", "things", "object", "objects", 
    "stuff", "piece", "pieces", "something", 
    "hand", "hands", "finger", "fingers",
    "action",
})

# Soft-vague nouns: bare → bare_soft_vague, with non-weak modifier → typed_generic (both pass)
_SOFT_VAGUE_OBJECTS = frozenset({
    "food", "container", "package", "appliance", "contents",
    "surface", "material", "ingredient",
    "vegetable", "fruit",
    "tool", "utensil",
    "part",
})

_BAD_PRONOUNS = frozenset({
    "them", "its", "these", "those", "this", "their",
})

# Adjectives that don't make a vague object specific
_WEAK_ADJS = frozenset({
    "small", "large", "big", "little", "tiny",
    "white", "black", "red", "green", "blue", "yellow", "brown", "orange", "pink", "dark", "light",
    "round", "flat", "long", "thin", "cylindrical", "spherical", "rectangular",
    # Past-participle adjectives (don't identify the object)
    "wrapped", "sliced", "chopped", "diced", "peeled", "cooked",
    "dried", "frozen", "fried", "baked", "roasted",
    # Material adjectives
    "metallic", "plastic", "wooden", "silver", "gold", "metal",
    # Ordinal/positional (don't identify the object)
    "next", "other", "same", "certain",
})

_ARTICLES = frozenset({"the", "a", "an"})


def _is_weak_adj(token: str) -> bool:
    """Check if token (possibly hyphenated like 'yellow-capped') is a weak adjective."""
    if token in _WEAK_ADJS:
        return True
    if '-' in token:
        return any(part in _WEAK_ADJS for part in token.split('-'))
    return False


def _is_weak_adj_comparative(w: str) -> bool:
    """Check if w is a comparative/superlative form of a weak adjective."""
    for suffix in ("er", "est"):
        if not w.endswith(suffix) or len(w) <= len(suffix) + 1:
            continue
        stem = w[:-len(suffix)]
        # direct: small+er → small
        if _is_weak_adj(stem):
            return True
        # silent-e: larg+er → large
        if _is_weak_adj(stem + "e"):
            return True
        # doubled consonant: bigg+est → big, thinn+er → thin
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            if _is_weak_adj(stem[:-1]):
                return True
        # y → ier/iest: tini+er → tiny, tini+est → tiny
        if stem.endswith("i") and _is_weak_adj(stem[:-1] + "y"):
            return True
    return False


# Always rejected in any position
_REJECT_VERBS = frozenset({
    "touch", "point", "search", "feel",
    "look", "see", "observe", "notice", "examine", "inspect",
    "operate", "guide", "manipulate", "handle",
    "direct",
    # Non-manipulation pointing/signaling
    "gesture", "wave", "signal", "indicate",
    # Vague meta-verbs
    "assist", "help", "repeat", "interact",
    "use", "work", "manage",
    # Non-manipulation contact/proximity (no discrete state change)
    "contact", "approach",
    # Stative / non-action (hand at rest, not manipulating)
    "rest", "remain", "stay", "maintain", "keep",
    # Presenting / displaying (not discrete manipulation)
    "display", "show", "view", "offer",
    # Continuous/repetitive tool actions (not discrete manipulation events)
    "mow", "trim", "vacuum", "sweep",
    # Continuous locomotion (not discrete manipulation)
    "drive", "steer", "walk",
    # Non-manipulation actions
    "clap", "follow", "play",
})

# Imprecise but valid manipulation verbs. The text gate judges them.
_SOFT_REJECT_VERBS = frozenset({"move", "position", "prepare", "displace", "reposition",
                                 "control", "create", "process", "leave"})

# _STABILIZING_ONLY passes per-phrase validation with allow_stabilizing=True
# (stabilizing-only fields are post-rejected). _TRANSPORT_VERBS always reject.
_STABILIZING_ONLY = frozenset({"hold", "steady", "stabilize", "support", "grip"})
_TRANSPORT_VERBS = frozenset({"carry", "bring", "transport"})
_STABILIZING_VERBS = _STABILIZING_ONLY | _TRANSPORT_VERBS

_PARTICLES = frozenset({
    "up", "down", "on", "off", "out", "in", "away", "over", "back", "apart",
})

# Food nouns lemminflect also tags as verbs. "X and Y" with both in this set
# is a compound noun, not a multi-action separator.
_FOOD_NOUNS = frozenset({
    "bread", "butter", "cream", "flour", "honey",
    "milk", "oil", "pepper", "salt", "sugar", "water",
})

# Words lemminflect wrongly tags as verbs ("tap", "handle", "seal" and "can" stay verbs)
_NOUN_NOT_VERB = frozenset({
    "cup", "plate", "bowl", "lid", "jar", "mug", "tray",
    "bin", "rack", "pot", "pan", "cap", "tin", "box", "bag",
    "forth", "person",
})

# Prepositions that can introduce a destination
_LOCATION_PREPS = frozenset({
    "in", "into", "on", "onto", "to", "from", "within", "across",
    "toward", "towards", "along",
    "near", "beside", "behind", "above", "below", "under", "over",
    "against", "between",
})

# Prepositions that may follow a repeated object in auto-"it" replacement
_OBJECT_PREPS = frozenset({
    "in", "into", "on", "onto", "to", "from", "toward",
    "towards", "under", "over", "through", "across", "along",
    "near", "beside", "behind", "above", "below", "against", "between",
})

# Direction adverbs that may follow an object or "it" in an action phrase
_DIR_ADVERBS = frozenset({
    "left", "right", "forward", "backward", "backwards",
})

# Trailing manner/group adverbs allowed after "it"
_TAIL_ADVERBS = frozenset({
    "together", "gently", "carefully", "slowly", "quickly", "firmly",
    "slightly",
})

# Words redundant in bimanual field (bimanual already implies togetherness)
_BIMANUAL_REDUNDANT_TAIL = frozenset({"together", "jointly"})
_BIMANUAL_REDUNDANT_HEAD = frozenset({"jointly", "together"})

_CROSS_HAND_PATTERNS = (
    " while ", " whilst ", " as the ",
    "the right hand", "the left hand",
    "right hand", "left hand",
    "other hand", "both hands",
    "together", "simultaneously",
)

_BIMANUAL_REJECT_PATTERNS = (
    "while holding", "while gripping", "while supporting", "while steady",
    # stem prefixes: match base and gerund forms
    "cooperat", "coordinat", "collaborat",
    "work together", "assist", "help",
)


def _has_hand_pattern(text: str, patterns) -> bool:
    """Space-padded patterns (" while ") match anywhere. Other patterns match whole words with an
    optional plural s, so "left hand" matches "left hands" but not "left handle"."""
    return any(pat in text if pat.startswith(" ") else re.search(r"\b" + re.escape(pat) + r"s?\b", text)
               for pat in patterns)


def _extract_obj_tokens(words):
    """Extract object tokens from verb phrase, skipping verb + particles."""
    j = 1
    while j < len(words) and words[j].lower() in _PARTICLES:
        j += 1
    return words[j:], j


def _reject_if_prep_object_is_vague(obj_tokens: List[str]) -> bool:
    """Check if the prep-phrase object after an intransitive verb+particle is vague.
    Applies only when direct_obj is empty. True means reject as vague_object."""
    prep_idx = None
    for k, t in enumerate(obj_tokens):
        if t in _LOCATION_PREPS:
            prep_idx = k
            break
    if prep_idx is None:
        return False
    prep_obj = []
    for t in obj_tokens[prep_idx + 1:]:
        if t in _LOCATION_PREPS or t in _PARTICLES or t in _DIR_ADVERBS:
            break
        prep_obj.append(t)
    prep_vague = _VAGUE_OBJECTS & set(prep_obj)
    if prep_vague:
        non_vague = [t for t in prep_obj
                     if t not in _VAGUE_OBJECTS and not _is_weak_adj(t) and not _is_weak_adj_comparative(t) and t not in _ARTICLES]
        if not non_vague:
            return True
    return False


def _classify_direct_obj(direct_obj: List[str]) -> str:
    """Classify direct-object NP specificity from lowercased pre-prep tokens.
    Returns: "concrete", "typed_generic", "appearance_only", "bare_soft_vague", "bare_vague", "none"."""
    content = [t for t in direct_obj if t not in _ARTICLES]
    if not content:
        return "none"
    vague_head = _VAGUE_OBJECTS & set(content)
    soft_vague_head = _SOFT_VAGUE_OBJECTS & set(content)
    if not vague_head and not soft_vague_head:
        return "concrete"
    non_vague_non_adj = [t for t in content
                         if t not in _VAGUE_OBJECTS and t not in _SOFT_VAGUE_OBJECTS
                         and not _is_weak_adj(t) and not _is_weak_adj_comparative(t)]
    if non_vague_non_adj:
        return "typed_generic"  # concrete modifier qualifies vague head
    has_adj = any(_is_weak_adj(t) or _is_weak_adj_comparative(t) for t in content)
    if has_adj:
        return "appearance_only"
    if soft_vague_head and not vague_head:
        return "bare_soft_vague"
    return "bare_vague"


_GENERIC_TASK_VERBS = frozenset({"prepare", "organize", "arrange", "rearrange",
                                  "sort", "clean", "tidy", "share"})
_GENERIC_TASK_NOUNS = frozenset({"task", "activity", "process"})


def _is_generic_task_summary(phrase: str) -> bool:
    """Detect generic task summaries ('prepare food' → True)."""
    words = phrase.split()
    if len(words) < 2:
        return False
    if _GENERIC_TASK_NOUNS & set(w.lower() for w in words):
        return True
    verb = words[0].lower()
    if verb not in _GENERIC_TASK_VERBS:
        return False
    obj_tokens, _ = _extract_obj_tokens(words)
    direct_obj = []
    for t in (w.lower() for w in obj_tokens):
        if t in _PARTICLES or t in _LOCATION_PREPS or t in _DIR_ADVERBS:
            break
        if t == "next":
            break
        direct_obj.append(t)
    obj_class = _classify_direct_obj(direct_obj)
    return obj_class not in ("concrete", "typed_generic")


# ---------------------------------------------------------------------------
# Canonical split / rejoin utilities
# ---------------------------------------------------------------------------

def normalize_punct(s: str) -> str:
    """Normalize comma spacing, remove Oxford commas, and turn 'and then' into 'and'."""
    s = re.sub(r'\s*,\s*', ', ', s)        # "A,B" / "A , B" → "A, B"
    s = re.sub(r',\s*and\s+', ' and ', s)  # Oxford comma: ", and " / ",and " → " and "
    s = re.sub(r'\band then\b', 'and', s)  # "and then" → "and" (temporal adverb artifact)
    return s


def _is_verb_like(token: str) -> bool:
    """Check if a token looks like a verb using lemminflect."""
    lemmas = getLemma(token.lower(), upos="VERB")
    return bool(lemmas)


def _starts_verb_phrase_after_and(words: List[str], first_verb: str) -> bool:
    """Whether the words after " and " start an action. After a base-form first verb, a word
    ending in -ing starts a noun ("chopping board"), not an action."""
    if len(words) < 2:
        return False
    first = first_verb.lower()
    return not (words[0].lower().endswith("ing") and first in getLemma(first, upos="VERB"))


def _split_chunk_by_and(chunk: str) -> Optional[List[str]]:
    """Split a comma-free chunk by ' and ' with all-or-nothing guards.
    If any resulting phrase fails a guard, the whole chunk stays one phrase.
    Returns None only on empty-phrase structural error."""
    if " and " not in chunk:
        return [chunk]
    parts = [p.strip() for p in chunk.split(" and ")]
    if any(not p for p in parts):
        return None
    _ARTICLES = {"a", "an", "the", "another", "some", "each"}
    for i in range(len(parts)):
        words = parts[i].split()
        # part starting with an article/determiner is a noun phrase, not an action
        if words[0].lower() in _ARTICLES:
            return [chunk]
        if not _is_verb_like(words[0]):
            return [chunk]
        if words[0].lower() in _NOUN_NOT_VERB:
            return [chunk]
        if len(words) < 2:
            return [chunk]
        if i > 0:
            if not _starts_verb_phrase_after_and(words, parts[0].split()[0]):
                return [chunk]
            prev_words = parts[i - 1].split()
            if prev_words[-1].lower() in _FOOD_NOUNS and words[0].lower() in _FOOD_NOUNS:
                return [chunk]
            # adjective before "and" → compound modifier, not a separator
            if _is_weak_adj(prev_words[-1].lower()):
                return [chunk]
    return parts


def split_actions_canonical(s: str, max_phrases: int = 3,
                            truncate: bool = False) -> Optional[List[str]]:
    """Split an action string into a phrase list ("A, B and C" -> ["A", "B", "C"]).
    Returns None for >max_phrases (unless truncate=True, which drops the excess) or structural errors.
    Input must be punctuation-normalized via normalize_punct first."""
    s = s.strip()
    if not s:
        return [s]

    if ", " in s:
        chunks = [c.strip() for c in s.split(", ")]
    else:
        chunks = [s]

    phrases = []
    for chunk in chunks:
        sub = _split_chunk_by_and(chunk)
        if sub is None:
            return None
        phrases.extend(sub)

    if any(not p for p in phrases):
        return None

    if len(phrases) > max_phrases:
        if truncate:
            phrases = phrases[:max_phrases]
        else:
            return None

    return phrases


def _rejoin_canonical(phrases: List[str]) -> str:
    """Rejoin phrases: 1 -> "A", 2 -> "A and B", 3 -> "A, B and C"."""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return phrases[0] + " and " + phrases[1]
    return phrases[0] + ", " + phrases[1] + " and " + phrases[2]


# ---------------------------------------------------------------------------
# Stabilizing verb simplification (list-based)
# ---------------------------------------------------------------------------

def _strip_leading_stabilizing(phrases: List[str]) -> List[str]:
    """Remove a leading stabilizing verb. Resolve "it" in the next phrase using the stripped verb's object."""
    if len(phrases) < 2:
        return phrases
    w1 = phrases[0].split()
    if len(w1) < 2:
        return phrases
    verb1 = w1[0].lower()
    if verb1 not in _STABILIZING_ONLY:
        return phrases
    obj1, _ = _extract_obj_tokens(w1)
    remaining = list(phrases[1:])
    if obj1 and verb1 in _STABILIZING_ONLY:
        wp = remaining[0].split()
        wp_lower = [t.lower() for t in wp]
        if "it" in wp_lower:
            k = wp_lower.index("it")
            wp = wp[:k] + obj1 + wp[k+1:]
            remaining[0] = " ".join(wp)
    return remaining


def _strip_trailing_stabilizing(phrases: List[str]) -> List[str]:
    """Remove trailing stabilizing verb from phrase list."""
    if len(phrases) < 2:
        return phrases
    wlast = phrases[-1].split()
    if wlast and wlast[0].lower() in _STABILIZING_ONLY:
        return phrases[:-1]
    return phrases


# ---------------------------------------------------------------------------
# Normalize / validate
# ---------------------------------------------------------------------------

def normalize_caption(s: str) -> str:
    """Normalize a raw action string: strip, lowercase the first letter, lemmatize verbs,
    strip instruments, simplify stabilizing, normalize n/a."""
    s = s.strip()
    if not s:
        return "n/a"
    s = s.splitlines()[0].strip()
    s = s.strip("\"'\u201c\u201d\u2018\u2019")
    if s.endswith("."):
        s = s[:-1].strip()

    low = s.lower().strip()
    if low in _NA_PHRASES:
        return "n/a"

    if s and s[0].isalpha():
        s = s[0].lower() + s[1:]

    s = normalize_punct(s)

    def lemmatize_first_verb(phrase: str) -> str:
        phrase = phrase.strip()
        if not phrase:
            return phrase
        words = phrase.split()
        if words:
            words[0] = _to_base_verb(words[0])
        return " ".join(words)

    # --- Split into canonical phrases ---
    phrases = split_actions_canonical(s)
    if phrases is None:
        if ", " in s or " and " in s:
            return "n/a"
        phrases = [s]

    # --- Per-phrase: lemmatize conjugated verbs after "and" in unsplit compounds ---
    # Runs before the first verb is lemmatized, because it reads that verb as written
    def lemmatize_and_verbs(phrase: str) -> str:
        if " and " not in phrase:
            return phrase
        left, right = phrase.split(" and ", 1)
        rw = right.strip().split()
        if _starts_verb_phrase_after_and(rw, left.split()[0]):
            rw[0] = _to_base_verb(rw[0])
        return left + " and " + " ".join(rw)

    phrases = [lemmatize_and_verbs(p) for p in phrases]

    # --- Per-phrase: lemmatize first verb ---
    phrases = [lemmatize_first_verb(p) for p in phrases]

    # --- Per-phrase: strip "with ..." instrumental suffix ---
    phrases = [_strip_with_suffix(p) for p in phrases]

    # --- Reassemble and check n/a ---
    s = _rejoin_canonical(phrases)
    s = s.strip()
    if not s or s.lower() in _NA_PHRASES:
        return "n/a"

    # Re-split (stripping may have changed structure)
    phrases = split_actions_canonical(s)
    if phrases is None:
        if ", " in s or " and " in s:
            return "n/a"
        phrases = [s]

    # --- Cross-phrase: auto-replace repeated object with "it" ---
    if len(phrases) >= 2:
        w1 = phrases[0].split()
        if len(w1) >= 2:
            obj1_words, _ = _extract_obj_tokens(w1)
            obj1_lower = [w.lower() for w in obj1_words]

            if obj1_lower:
                # Strip location phrase tail + trailing direction adverbs
                obj1_match = []
                for wi_idx, w in enumerate(obj1_lower):
                    if w in _LOCATION_PREPS:
                        break
                    # "next to" bigram: treat as location preposition
                    if w == "next" and wi_idx + 1 < len(obj1_lower) and obj1_lower[wi_idx + 1] == "to":
                        break
                    obj1_match.append(w)
                while obj1_match and obj1_match[-1] in _DIR_ADVERBS:
                    obj1_match = obj1_match[:-1]
                if not obj1_match:
                    obj1_match = obj1_lower

                for idx in range(1, len(phrases)):
                    wi = phrases[idx].split()
                    if len(wi) >= 2:
                        obji_words, obji_start = _extract_obj_tokens(wi)
                        obji_lower = [w.lower() for w in obji_words]
                        if obji_lower:
                            n = len(obj1_match)
                            if obji_lower[:n] == obj1_match:
                                after = obji_lower[n:]
                                if not after or after[0] in _OBJECT_PREPS or after[0] in _PARTICLES or after[0] in _DIR_ADVERBS:
                                    new_obji = ["it"] + obji_words[n:]
                                    wi = wi[:obji_start] + new_obji
                                    phrases[idx] = " ".join(wi)

    # --- Cross-phrase: simplify stabilizing verbs ---
    phrases = _strip_leading_stabilizing(phrases)
    phrases = _strip_trailing_stabilizing(phrases)

    return _rejoin_canonical(phrases)


def caption_format_reason(s: str, allow_stabilizing: bool = False) -> Optional[str]:
    """Return rejection reason for caption format (structural/lexical checks only), or None if valid.
    allow_stabilizing: skip _STABILIZING_ONLY rejection at phrase level."""
    if s == "n/a":
        return None
    if not s or not s[0].islower():
        return "fmt_structural"
    if s.endswith("."):
        return "fmt_structural"

    s = normalize_punct(s)

    # --- Structural: canonical split ---
    phrases = split_actions_canonical(s)
    if phrases is None:
        return "fmt_structural"

    # --- Per-phrase validation ---
    for phrase_idx, phrase in enumerate(phrases):
        words = phrase.strip().split()
        if len(words) < 2:
            return "fmt_structural"
        verb = words[0].lower()

        direct_obj, obj_tokens = _extract_direct_obj(words)
        obj_class = _classify_direct_obj(direct_obj)

        if verb in _REJECT_VERBS:
            return "non_manipulation_verb"
        # _SOFT_REJECT_VERBS: pass through (text gate handles quality)
        if verb in _STABILIZING_VERBS:
            if verb in _TRANSPORT_VERBS:
                return "non_manipulation_verb"
            elif not allow_stabilizing:
                return "non_manipulation_verb"
        if verb in _NOUN_NOT_VERB:
            return "non_manipulation_verb"

        if not obj_tokens and not (phrase.strip().split()[-1].lower() == "it"):
            return "fmt_structural"

        if obj_class == "bare_vague":
            return "vague_object"
        # bare_soft_vague, appearance_only: pass (text gate handles quality)

        if not direct_obj and obj_tokens and _reject_if_prep_object_is_vague(obj_tokens):
            return "vague_object"

    # --- No bad pronouns anywhere ---
    all_words = s.lower().split()
    if _BAD_PRONOUNS & set(all_words):
        return "bad_pronoun"

    # --- "it" rules ---
    n = len(phrases)
    if n == 1:
        if "it" in all_words:
            return "fmt_structural"
    else:  # n == 2 or n == 3
        first_words = phrases[0].strip().lower().split()
        if "it" in first_words:
            return "fmt_structural"
        # "it" in subsequent phrases: must be followed by nothing, particle, prep, direction adverb, or manner adverb
        for idx in range(1, n):
            phrase_words = phrases[idx].strip().lower().split()
            if "it" in phrase_words:
                it_idx = phrase_words.index("it")
                trailing = phrase_words[it_idx + 1:]
                if trailing:
                    first_after = trailing[0]
                    if (first_after not in _PARTICLES
                            and first_after not in _OBJECT_PREPS
                            and first_after not in _DIR_ADVERBS
                            and first_after not in _TAIL_ADVERBS):
                        return "fmt_structural"

    return None


def _extract_last_json_object(raw: str, prefer_keys: Tuple[str, ...] = ("action", "think"),
                              prefer_any: bool = False) -> Optional[str]:
    """Extract the last top-level JSON object from raw text via brace-depth tracking.
    prefer_keys: prefer candidates containing these keys. prefer_any: OR instead of AND."""
    in_str = False
    esc = False
    depth = 0
    start = None
    candidates = []
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start:i + 1])
                    start = None
    check = any if prefer_any else all
    for s in reversed(candidates):
        if check(f'"{k}"' in s for k in prefer_keys):
            return s
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Per-hand extraction, validation & guards
# ---------------------------------------------------------------------------

_OVERLAY_WORDS = frozenset({
    "overlay", "overlays",
    "marker", "markers",
    "dot", "dots",
    "trail", "trails",
    "circle", "circles",
    "indicator", "indicators",
    "trajectory", "trajectories",
})


def _has_overlay_words(text: str) -> bool:
    """Check if text contains overlay hallucination words (token-boundary matching).
    'grab the dot' -> True, 'bookmark' -> False (token 'bookmark' != 'marker')."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & _OVERLAY_WORDS)


_OBJECT_HEAD_PREPS = frozenset({
    "to", "into", "onto", "on", "in", "from", "off", "out",
    "inside", "outside", "over", "under", "at",
    "near", "beside", "behind", "above", "below", "against", "between",
})


# Prepositions that introduce a target object (e.g. "reach for the drawer")
_TARGET_PREPS = frozenset({"for"})

# Combined stop words for chain anchor noun extraction
_ANCHOR_STOP = _LOCATION_PREPS | _OBJECT_HEAD_PREPS | _TARGET_PREPS | _PARTICLES | _DIR_ADVERBS

# Nouns too vague to serve as anchor nouns for cross-field matching
_ANCHOR_EXCLUDE = _VAGUE_OBJECTS | _SOFT_VAGUE_OBJECTS


def _extract_chain_anchor_nouns(phrase: str) -> set:
    """Extract primary object nouns for cross-field noun-mismatch checks.
    Prefers direct-object nouns, falls back to post-prep nouns for intransitive
    patterns. Excludes vague/soft-vague nouns."""
    words = phrase.split()
    if len(words) < 2:
        return set()
    j = 1
    while j < len(words) and words[j].lower() in _PARTICLES:
        j += 1
    nouns = set()
    prep_idx = None
    had_pre_prep_candidates = False
    for idx, t in enumerate(w.lower() for w in words[j:]):
        if t in _ANCHOR_STOP:
            prep_idx = j + idx
            break
        if t == "next":
            prep_idx = j + idx
            break
        if t == "it":
            # Pronoun object: no anchor extraction (avoids destination leak)
            return set()
        if t not in _ARTICLES and not _is_weak_adj(t) and not _is_weak_adj_comparative(t):
            had_pre_prep_candidates = True
            if t not in _ANCHOR_EXCLUDE:
                nouns.add(t)
    # Intransitive fallback: extract post-prep nouns only when there were no
    # pre-prep candidates (avoids matching on destination nouns)
    if not nouns and not had_pre_prep_candidates and prep_idx is not None:
        for t in (w.lower() for w in words[prep_idx + 1:]):
            if t in _ANCHOR_STOP:
                break
            if t == "next":
                break
            if (t not in _ARTICLES and not _is_weak_adj(t) and not _is_weak_adj_comparative(t) and t != "it"
                    and t not in _ANCHOR_EXCLUDE):
                nouns.add(t)
    return nouns


def extract_from_json_perhand(raw: str) -> Tuple[Dict[str, str], str, Optional[str]]:
    """Extract per-hand fields from VLM JSON output.
    Returns (narr_dict, think_str, reject_reason_or_None).
    narr_dict has keys: think, left, right, bimanual."""
    json_str = _extract_last_json_object(raw.strip(), prefer_keys=("left", "right"), prefer_any=True)
    if not json_str:
        return {"think": "", "left": "n/a", "right": "n/a", "bimanual": "n/a"}, "", "no_json"
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return {"think": "", "left": "n/a", "right": "n/a", "bimanual": "n/a"}, "", "json_decode"

    if not isinstance(obj, dict):
        return {"think": "", "left": "n/a", "right": "n/a", "bimanual": "n/a"}, "", "json_decode"

    if "left" not in obj and "right" not in obj:
        return {"think": "", "left": "n/a", "right": "n/a", "bimanual": "n/a"}, "", "missing_left_right"

    raw_think = obj.get("think", "")
    think = str(raw_think).strip() if raw_think is not None else ""

    def _field_to_str(val):
        if val is None or not isinstance(val, str):
            return "n/a"
        val = val.strip()
        if not val or val.lower().strip() in _NA_PHRASES:
            return "n/a"
        return val

    left = _field_to_str(obj.get("left"))
    right = _field_to_str(obj.get("right"))
    bimanual = _field_to_str(obj.get("bimanual"))

    narr_dict = {"think": think, "left": left, "right": right, "bimanual": bimanual}

    if left == "n/a" and right == "n/a" and bimanual == "n/a":
        return narr_dict, think, "all_na"

    return narr_dict, think, None


def _extract_direct_obj(words: List[str]) -> Tuple[List[str], List[str]]:
    """Extract direct-object tokens and all obj_tokens from a phrase's word list.
    Returns (direct_obj, obj_tokens) where direct_obj stops at location preps."""
    j = 1
    while j < len(words) and words[j].lower() in _PARTICLES:
        j += 1
    obj_tokens = [w.lower() for w in words[j:]]
    direct_obj = []
    for k, t in enumerate(obj_tokens):
        if t in _PARTICLES or t in _LOCATION_PREPS or t in _DIR_ADVERBS:
            break
        if t == "next" and k + 1 < len(obj_tokens) and obj_tokens[k + 1] == "to":
            break
        direct_obj.append(t)
    return direct_obj, obj_tokens


def _validate_single_phrase(phrase: str, allow_stabilizing: bool) -> Optional[str]:
    """Validate a single action phrase. Returns rejection reason or None if valid.
    Reasons: fmt_structural, non_manipulation_verb, vague_object, bad_pronoun, overlay_hallucination."""
    words = phrase.strip().split()
    if len(words) < 2:
        return "fmt_structural"
    verb = words[0].lower()

    direct_obj, obj_tokens = _extract_direct_obj(words)
    obj_class = _classify_direct_obj(direct_obj)

    if verb in _REJECT_VERBS:
        return "non_manipulation_verb"
    # _SOFT_REJECT_VERBS: pass through (text gate handles quality)
    if verb in _STABILIZING_VERBS:
        if verb in _TRANSPORT_VERBS:
            return "non_manipulation_verb"
        if not allow_stabilizing:
            return "non_manipulation_verb"
    if verb in _NOUN_NOT_VERB:
        return "non_manipulation_verb"

    if not obj_tokens and words[-1].lower() != "it":
        return "fmt_structural"

    if obj_class == "bare_vague":
        return "vague_object"
    # typed_generic, bare_soft_vague, appearance_only: pass (text gate handles quality)

    if not direct_obj and obj_tokens and _reject_if_prep_object_is_vague(obj_tokens):
        return "vague_object"

    if _BAD_PRONOUNS & set(w.lower() for w in words):
        return "bad_pronoun"
    if _has_overlay_words(phrase):
        return "overlay_hallucination"

    return None


def normalize_and_validate_perhand(narr_dict: Dict[str, str], return_drops: bool = False):
    """Per-hand normalize + validate prefilter (structural/lexical only). Semantic quality goes to the text gate.
    Returns (normalized_narr_dict, reject_reason_or_None). With return_drops=True it also returns
    drops = {field: [(phrase_idx, phrase, reason), ...]} (phrase_idx -1 = field-level entry)."""
    think = narr_dict.get("think", "")
    result = {"think": think}
    all_drops: Dict[str, List[Tuple[int, str, str]]] = {}

    for key in ("left", "right", "bimanual"):
        val = narr_dict.get(key, "n/a")
        if not isinstance(val, str):
            val = "n/a"

        val_clean = val.strip()
        if not val_clean or val_clean.lower() in _NA_PHRASES:
            result[key] = "n/a"
            all_drops[key] = []
            continue

        # Pre-split cleanup (same initial steps as normalize_caption, but no cross-phrase ops)
        val_clean = val_clean.splitlines()[0].strip()
        val_clean = val_clean.strip("\"'\u201c\u201d\u2018\u2019")
        if val_clean.endswith("."):
            val_clean = val_clean[:-1].strip()
        if not val_clean or val_clean.lower() in _NA_PHRASES:
            result[key] = "n/a"
            all_drops[key] = []
            continue
        if val_clean[0].isalpha():
            val_clean = val_clean[0].lower() + val_clean[1:]
        val_clean = normalize_punct(val_clean)

        # Pre-split lexical pattern checks (cheapest filter first)
        val_lower = val_clean.lower()
        if key in ("left", "right"):
            if _has_hand_pattern(val_lower, _CROSS_HAND_PATTERNS):
                result[key] = "n/a"
                all_drops[key] = [(-1, val_clean, "cross_hand_leakage")]
                continue
        elif key == "bimanual":
            if any(pat in val_lower for pat in _BIMANUAL_REJECT_PATTERNS):
                result[key] = "n/a"
                all_drops[key] = [(-1, val_clean, "bimanual_reject_pattern")]
                continue

        # Split into phrases BEFORE per-phrase normalization (avoids auto-"it")
        # Keep the first 3 phrases of a field with 4 or more
        phrases_raw = split_actions_canonical(val_clean, max_phrases=3, truncate=True)
        if phrases_raw is None:
            result[key] = "n/a"
            all_drops[key] = [(-1, val_clean, "fmt_structural")]
            continue

        # Hybrid drop: fmt_structural/normalize_na kill the field. Other reasons allow phrase-level salvage
        _FIELD_HARD_DROP = frozenset({"fmt_structural"})
        allow_stab = key in ("left", "right")
        surviving = []  # [(orig_idx, canon_phrase), ...]
        dropped = []
        field_hard_fail = False
        for phrase_idx, phrase in enumerate(phrases_raw):
            canon_p = normalize_caption(phrase)
            if canon_p == "n/a":
                dropped.append((phrase_idx, phrase, "normalize_na"))
                field_hard_fail = True
                continue
            reason = _validate_single_phrase(canon_p, allow_stabilizing=allow_stab)
            if reason is not None:
                if reason in _FIELD_HARD_DROP:
                    dropped.append((phrase_idx, canon_p, reason))
                    field_hard_fail = True
                else:
                    dropped.append((phrase_idx, canon_p, reason))
            else:
                surviving.append((phrase_idx, canon_p))

        # Sort by original phrase order before further processing
        surviving.sort(key=lambda x: x[0])

        # --- Strict drop policy: restrict phrase-level salvage ---
        if not field_hard_fail:
            surviving_indices = {idx for idx, _ in surviving}
            actually_dropped = [i for i in range(len(phrases_raw))
                                if i not in surviving_indices]
            if actually_dropped:
                # Rule 1: non-leading drop → truncate to longest surviving prefix
                if any(i > 0 for i in actually_dropped):
                    prefix_end = 0
                    for idx in range(len(phrases_raw)):
                        if idx in surviving_indices:
                            prefix_end = idx + 1
                        else:
                            break

                    if prefix_end > 0:
                        surviving = [(idx, text) for idx, text in surviving
                                     if idx < prefix_end]
                        dropped.append((-1, val_clean,
                            f"truncated_nonleading_drop:kept={prefix_end}/{len(phrases_raw)}"))
                    else:
                        # No leading survivors → field hard drop
                        dropped.append((-1, val_clean, "strict_nonleading_drop"))
                        field_hard_fail = True
                elif 0 in actually_dropped:
                    # Rule 2: leading drop allowed only for stabilizing verbs. Bimanual never salvages
                    if key == "bimanual":
                        dropped.append((-1, val_clean, "strict_bimanual_leading_drop"))
                        field_hard_fail = True
                    else:
                        first_canon = normalize_caption(phrases_raw[0])
                        first_words = first_canon.split() if first_canon != "n/a" else []
                        first_verb = first_words[0].lower() if first_words else ""
                        if first_verb in _STABILIZING_ONLY:
                            pass  # stabilizing leading drop is fine
                        else:
                            dropped.append((-1, val_clean, "strict_bad_leading_drop"))
                            field_hard_fail = True

        if field_hard_fail:
            result[key] = "n/a"
            all_drops[key] = dropped
            continue

        # Dangling "it": first surviving phrase with "it" object after a drop → field hard drop
        has_salvage_drop = any(
            not r.startswith("kept:")
            for _, _, r in dropped
        )
        if has_salvage_drop and surviving:
            _, first_text = surviving[0]
            first_words = first_text.split()
            obj_tokens, _ = _extract_obj_tokens(first_words)
            if obj_tokens and obj_tokens[0].lower() == "it":
                dropped.append((-1, first_text, "dangling_it"))
                result[key] = "n/a"
                all_drops[key] = dropped
                continue

        if not surviving:
            result[key] = "n/a"
            all_drops[key] = dropped
            continue

        # Cross-phrase stabilizing simplification on assembled field
        if len(surviving) >= 2 and key in ("left", "right"):
            assembled_phrases = [text for _, text in surviving]
            simplified = _strip_leading_stabilizing(assembled_phrases)
            simplified = _strip_trailing_stabilizing(simplified)
            if len(simplified) < len(assembled_phrases):
                surviving = [(0, text) for text in simplified]
                dropped.append(
                    (-1, _rejoin_canonical(assembled_phrases),
                     f"stabilizing_simplified:{len(assembled_phrases)}->{len(simplified)}"))

        # Bimanual: strip redundant leading/trailing "together"/"jointly"
        if key == "bimanual" and surviving:
            new_surviving = []
            for idx, text in surviving:
                words = text.split()
                while len(words) > 1 and words[0].lower() in _BIMANUAL_REDUNDANT_HEAD:
                    words.pop(0)
                while len(words) > 1 and words[-1].lower() in _BIMANUAL_REDUNDANT_TAIL:
                    words.pop()
                new_text = " ".join(words)
                if len(new_text.split()) < 2:
                    # A phrase collapsed to a single word drops the field
                    dropped.append((idx, text, "bimanual_redundant_collapse"))
                    field_hard_fail = True
                    break
                new_surviving.append((idx, new_text))
            if field_hard_fail:
                result[key] = "n/a"
                all_drops[key] = dropped
                continue
            surviving = new_surviving

        # Rejoin and final cross-phrase check (catches remaining "it" edge cases)
        canon = _rejoin_canonical([text for _, text in surviving])
        reason = caption_format_reason(canon, allow_stabilizing=allow_stab)
        if reason is not None:
            result[key] = "n/a"
            all_drops[key] = dropped + [(-1, canon, reason)]
            continue

        result[key] = canon
        all_drops[key] = dropped

    # Bimanual hand-reference check
    if result["bimanual"] != "n/a":
        bi_lower = result["bimanual"].lower()
        if _has_hand_pattern(bi_lower, _HAND_REFS):
            all_drops.setdefault("bimanual", []).append((-1, result["bimanual"], "bimanual_hand_ref"))
            result["bimanual"] = "n/a"

    # Bimanual generic-task-summary: drop when ALL phrases are generic summaries
    if result["bimanual"] != "n/a":
        bi_phrases = split_actions_canonical(result["bimanual"])
        if bi_phrases and all(_is_generic_task_summary(p) for p in bi_phrases):
            all_drops.setdefault("bimanual", []).append(
                (-1, result["bimanual"], "bimanual_generic_summary"))
            result["bimanual"] = "n/a"

    # Bimanual noun mismatch: bimanual anchor nouns must overlap with left/right
    if result["bimanual"] != "n/a":
        lr_nouns = set()
        for side in ("left", "right"):
            if result[side] != "n/a":
                side_phrases = split_actions_canonical(result[side])
                if side_phrases:
                    for p in side_phrases:
                        lr_nouns |= _extract_chain_anchor_nouns(p)
        bi_nouns = set()
        bi_phrases = split_actions_canonical(result["bimanual"])
        if bi_phrases:
            for p in bi_phrases:
                bi_nouns |= _extract_chain_anchor_nouns(p)
        if lr_nouns and bi_nouns and not (bi_nouns & lr_nouns):
            all_drops.setdefault("bimanual", []).append(
                (-1, result["bimanual"], "bimanual_noun_mismatch"))
            result["bimanual"] = "n/a"

    # Per-hand fields whose phrases are ALL stabilizing verbs drop to n/a
    for field in ("left", "right"):
        if result[field] != "n/a":
            _phrases = split_actions_canonical(result[field])
            if _phrases and all(p.split()[0].lower() in _STABILIZING_ONLY for p in _phrases):
                all_drops.setdefault(field, []).append((-1, result[field], "stabilizing_only"))
                result[field] = "n/a"

    if result["left"] == "n/a" and result["right"] == "n/a" and result["bimanual"] == "n/a":
        if return_drops:
            return result, "all_na", all_drops
        return result, "all_na"

    if return_drops:
        return result, None, all_drops
    return result, None


# ---------------------------------------------------------------------------
# VLM self-verification output parsing
# ---------------------------------------------------------------------------

_VALID_NA_KEYS = frozenset({"global_na", "left_na", "right_na", "bimanual_na"})


def parse_verification_output(
    raw: str,
    shown_fields: Optional[Tuple[str, ...]] = None,
) -> Tuple[Dict[str, bool], Optional[str]]:
    """Parse VLM verification JSON into per-field *_na flags.
    Returns (na_flags, reason). Only global_na and *_na keys matching shown_fields are kept (all if None).
    Fail-open: any parse error returns ({}, "parse_error"), so the caption is kept as-is."""
    # Sanitize shown_fields (guard against "left_na" being passed instead of "left")
    _VALID_FIELDS = ("left", "right", "bimanual")
    if shown_fields is not None:
        orig = shown_fields
        shown_fields = tuple(f for f in shown_fields if f in _VALID_FIELDS)
        if orig and not shown_fields:
            shown_fields = None  # fallback: accept all known *_na keys

    raw_s = raw.strip()
    if raw_s.startswith('```'):
        first_nl = raw_s.find('\n')
        if first_nl >= 0:
            raw_s = raw_s[first_nl + 1:]
        if raw_s.rstrip().endswith('```'):
            raw_s = raw_s.rstrip()[:-3].rstrip()
        raw = raw_s

    json_str = _extract_last_json_object(
        raw,
        prefer_keys=("global_na", "left_na", "right_na", "bimanual_na"),
        prefer_any=True,
    )
    if not json_str:
        return {}, "parse_error"

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return {}, "parse_error"

    if not isinstance(obj, dict):
        return {}, "parse_error"

    if shown_fields is not None:
        allowed = {"global_na"} | {f"{f}_na" for f in shown_fields}
    else:
        allowed = _VALID_NA_KEYS

    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    na_flags: Dict[str, bool] = {}
    for key in allowed:
        if key in obj:
            na_flags[key] = _to_bool(obj[key])

    # global_na=true forces all shown *_na true. This must run BEFORE the missing-field check
    if na_flags.get("global_na", False):
        for key in allowed:
            if key != "global_na":
                na_flags[key] = True

    # Missing shown field flags → parse_error (contract violation)
    if shown_fields is not None:
        required = {f"{f}_na" for f in shown_fields}
        if any(k not in na_flags for k in required):
            return {}, "parse_error"

    reason = str(obj.get("reason", "n/a")).strip() or "n/a"
    return na_flags, reason


def classify_verification_reason(reason: str, global_na: bool = False) -> str:
    """Map free-text verification reason to a tracker category via keyword matching.
    global_na distinguishes window-level from candidate-level rejections for ambiguous codes."""
    r = (reason or "").lower().strip()

    if r == "parse_error":
        return "vlm_verify_parse_error"

    # "saw"/"sand"/"iron" excluded here (substring false positives).
    # The word-boundary regex below matches their -ing forms.
    _TOOL_KEYWORDS_SUBSTR = ("mow", "trim", "vacuum", "sweep", "drill",
                             "grind", "weld", "solder", "polish",
                             "stir", "driv", "steer")
    _TOOL_KEYWORDS_WORD = re.compile(
        r'\b(?:sawing|sanding|ironing)\b', re.IGNORECASE)

    def _has_tool_keyword(text):
        if any(kw in text for kw in _TOOL_KEYWORDS_SUBSTR):
            return True
        if _TOOL_KEYWORDS_WORD.search(text):
            return True
        return False

    # Structured tags (prompt instructs VLM to use "global:<tag>" format)
    _STRUCTURED_TAGS = {
        "other_person": "vlm_verify_other_person",
        "self_contact": "vlm_verify_self_contact",
        "walking": "vlm_verify_walking",
        "dark": "vlm_verify_dark",
        "continuous_tool": "vlm_verify_continuous_tool",
    }

    # Keyword fallbacks for free-form reasons
    _PERSON_KEYWORDS = ("another person", "other person", "second person",
                        "rule 8", "rule8")
    _SELF_CONTACT_KEYWORDS = ("self-contact", "self contact",
                              "own body", "own person", "own clothing",
                              "currently being worn", "currently worn",
                              "worn item", "worn items",
                              "worn clothing", "worn accessory", "worn accessories",
                              "rule 9", "rule9")

    if r.startswith("global:"):
        g = r.split(":", 1)[1].strip()
        for tag, code in _STRUCTURED_TAGS.items():
            if g.startswith(tag):
                return code
        if any(kw in g for kw in _PERSON_KEYWORDS):
            return "vlm_verify_other_person"
        if any(kw in g for kw in _SELF_CONTACT_KEYWORDS):
            return "vlm_verify_self_contact"
        if "walk" in g:
            return "vlm_verify_walking"
        if "dark" in g or "unclear" in g or "blurry" in g:
            return "vlm_verify_dark"
        if _has_tool_keyword(g) or "tool" in g:
            return "vlm_verify_continuous_tool"
        return "vlm_verify_other"

    # Non-global (field-level) reasons
    if any(kw in r for kw in _PERSON_KEYWORDS):
        return "vlm_verify_other_person" if global_na else "vlm_verify_untracked"
    if any(kw in r for kw in _SELF_CONTACT_KEYWORDS):
        return "vlm_verify_self_contact" if global_na else "vlm_verify_other"
    if "walk" in r:
        return "vlm_verify_walking"
    if "dark" in r or "unclear" in r or "blurry" in r:
        return "vlm_verify_dark"
    if _has_tool_keyword(r):
        return "vlm_verify_continuous_tool"
    if any(kw in r for kw in ("continuous", "repetitive")):
        return "vlm_verify_continuous_generic"
    if "stabiliz" in r or "only hold" in r or "only grip" in r:
        return "vlm_verify_stabilizing"
    if any(kw in r for kw in ("not observed", "not visible", "wrong", "mismatch", "inconsistent", "unsupported", "over-specif")):
        return "vlm_verify_inconsistent"
    if any(kw in r for kw in ("unclear object", "ambiguous", "cannot identify", "unidentifiable")):
        return "vlm_verify_unclear_object"
    if any(kw in r for kw in ("no contact", "hover", "idle", "no interaction", "no purposeful")):
        return "vlm_verify_no_contact"
    if any(kw in r for kw in ("untracked", "wrong region", "misaligned")):
        return "vlm_verify_untracked"
    if "bimanual" in r or "not shared" in r:
        return "vlm_verify_bimanual"
    return "vlm_verify_other"


def parse_label_verify_output(raw: str, shown_fields=None):
    """Parse text-only label verifier output into per-field *_drop flags.
    Returns (drop_flags, reason). Fail-open: on parse error returns ({}, "parse_error")."""
    _VALID_FIELDS = ("left", "right", "bimanual")
    if shown_fields is not None:
        shown_fields = tuple(f for f in shown_fields if f in _VALID_FIELDS)
        if not shown_fields:
            shown_fields = None

    raw_s = raw.strip()
    if raw_s.startswith('```'):
        first_nl = raw_s.find('\n')
        if first_nl >= 0:
            raw_s = raw_s[first_nl + 1:]
        if raw_s.rstrip().endswith('```'):
            raw_s = raw_s.rstrip()[:-3].rstrip()
        raw = raw_s

    json_str = _extract_last_json_object(
        raw,
        prefer_keys=("left_drop", "right_drop", "bimanual_drop"),
        prefer_any=True,
    )
    if not json_str:
        return {}, "parse_error"

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return {}, "parse_error"

    if not isinstance(obj, dict):
        return {}, "parse_error"

    if shown_fields is not None:
        allowed = {f"{f}_drop" for f in shown_fields}
    else:
        allowed = {f"{f}_drop" for f in _VALID_FIELDS}

    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    drop_flags: Dict[str, bool] = {}
    for key in allowed:
        if key in obj:
            drop_flags[key] = _to_bool(obj[key])

    # Reject if model omitted a field it was asked about (fail-open = keep candidate)
    if shown_fields is not None:
        required = {f"{f}_drop" for f in shown_fields}
        if any(k not in drop_flags for k in required):
            return {}, "parse_error"

    reason = str(obj.get("reason", "n/a")).strip() or "n/a"
    return drop_flags, reason


# ---------------------------------------------------------------------------
# Post-assembly cross-field consistency check
# ---------------------------------------------------------------------------

_HAND_REFS = ("left hand", "right hand", "both hands", "each hand",
              "one hand", "other hand", "two hands")


def postcheck_assembled_triplet(
    assembled: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Cross-field consistency recheck on the assembled dict (fields may come from different samples).
    Cross-hand leakage and bimanual hand-reference/generic-summary checks hard-drop.
    Bimanual noun-mismatch only appends a soft flag to bimanual_soft_flags.
    Returns (patched_assembled, postcheck_meta) with {field}_dropped/{field}_drop_reason and *_soft_flags."""
    result = dict(assembled)
    meta: Dict[str, str] = {
        "bimanual_dropped": False,
        "bimanual_drop_reason": None,
        "bimanual_soft_flags": [],
        "left_soft_flags": [],
        "right_soft_flags": [],
    }

    # 0. Cross-hand leakage recheck (per-hand)
    for side in ("left", "right"):
        if result.get(side, "n/a") != "n/a":
            side_lower = result[side].lower()
            if _has_hand_pattern(side_lower, _CROSS_HAND_PATTERNS):
                meta[f"{side}_dropped"] = True
                meta[f"{side}_drop_reason"] = "postcheck_cross_hand_leakage"
                result[side] = "n/a"

    # 1-3. Bimanual-specific checks (only when bimanual is active)
    if result.get("bimanual", "n/a") != "n/a":
        # 1. Hand-reference check (hard drop)
        bi_lower = result["bimanual"].lower()
        if _has_hand_pattern(bi_lower, _HAND_REFS):
            meta["bimanual_dropped"] = True
            meta["bimanual_drop_reason"] = "postcheck_bimanual_hand_ref"
            result["bimanual"] = "n/a"

        # 2. Generic-task-summary check (hard drop)
        if result["bimanual"] != "n/a":
            bi_phrases = split_actions_canonical(result["bimanual"])
            if bi_phrases and all(_is_generic_task_summary(p) for p in bi_phrases):
                meta["bimanual_dropped"] = True
                meta["bimanual_drop_reason"] = "postcheck_bimanual_generic_summary"
                result["bimanual"] = "n/a"

        # 3. Noun-mismatch vs assembled left/right → soft flag
        if result["bimanual"] != "n/a":
            lr_nouns = set()
            for side in ("left", "right"):
                if result.get(side, "n/a") != "n/a":
                    side_phrases = split_actions_canonical(result[side])
                    if side_phrases:
                        for p in side_phrases:
                            lr_nouns |= _extract_chain_anchor_nouns(p)
            bi_nouns = set()
            bi_phrases = split_actions_canonical(result["bimanual"])
            if bi_phrases:
                for p in bi_phrases:
                    bi_nouns |= _extract_chain_anchor_nouns(p)
            if lr_nouns and bi_nouns and not (bi_nouns & lr_nouns):
                meta["bimanual_soft_flags"].append("postcheck_bimanual_noun_mismatch")

    # 4. Vague noun recheck: hard-vague always drops. Bare soft-vague (no real modifier) drops
    _HARD_VAGUE = {"something", "stuff"}
    _SOFT_VAGUE_POSTCHECK = {
        "object", "objects", "item", "items", "thing", "things",
        "piece", "pieces", "material", "materials", "contents",
    }
    # Relational/determiner adjectives don't count as appearance modifiers
    _RELATIONAL_ADJS = {"other", "same", "certain", "next"}
    _ALL_VAGUE_POSTCHECK = _HARD_VAGUE | _SOFT_VAGUE_POSTCHECK
    for field in ("left", "right", "bimanual"):
        if result.get(field, "n/a") != "n/a":
            words = re.findall(r"[a-z]+", result[field].lower())
            word_set = set(words)
            matched_hard = word_set & _HARD_VAGUE
            matched_soft = word_set & _SOFT_VAGUE_POSTCHECK
            if not matched_hard and not matched_soft:
                continue
            if matched_hard:
                drop = True
            else:
                # soft vague: needs a real modifier within its NP
                has_real_modifier = False
                ignore = _ARTICLES | _RELATIONAL_ADJS | _ALL_VAGUE_POSTCHECK
                for i, w in enumerate(words):
                    if w in matched_soft:
                        for j in range(i - 1, -1, -1):
                            if words[j] in _ARTICLES:
                                break  # hit NP start, no real modifier found
                            if words[j] in ignore:
                                continue
                            has_real_modifier = True
                            break
                        if has_real_modifier:
                            break
                drop = not has_real_modifier
            if drop:
                if field == "bimanual":
                    meta["bimanual_dropped"] = True
                    meta["bimanual_drop_reason"] = "postcheck_vague_noun"
                else:
                    meta[f"{field}_dropped"] = True
                    meta[f"{field}_drop_reason"] = "postcheck_vague_noun"
                result[field] = "n/a"
            else:
                flag_key = "bimanual_soft_flags" if field == "bimanual" else f"{field}_soft_flags"
                meta[flag_key].append("postcheck_vague_noun_soft")

    # 5. Bimanual without per-hand support (soft flag only)
    if (result.get("bimanual", "n/a") != "n/a"
            and result.get("left", "n/a") == "n/a"
            and result.get("right", "n/a") == "n/a"):
        meta["bimanual_soft_flags"].append("postcheck_bimanual_without_perhand")

    return result, meta
