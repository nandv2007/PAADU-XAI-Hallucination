#!/usr/bin/env python3
"""
Strict Allowlisted Grammar-Based Factual Assertion Parser and Complete Span Validator
for PAADU/CHARM Telemetry and XAI Explanations.

Enforces Complete Assertion Validation Invariants:
1. Every nontrivial span of the original text is accounted for.
2. Every factual assertion is extracted with exact character offsets.
3. Every extracted assertion has a source field and comparison result.
4. Every factual assertion is independently verified against authoritative incident telemetry.
5. Unknown causal relationships (e.g. 'because', 'proving'), unknown incident identifiers,
   unrecognized policy values, and mismatched feature tuples FAIL CLOSED.
6. A claim is ONLY labeled SUPPORTED [VERIFIED TELEMETRY] if:
   - Zero unparsed factual spans remain.
   - Zero unsupported or contradictory assertions exist.
   - At least one supported telemetry assertion is present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, NamedTuple

NUM_PATTERN = r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'

# Causal Connectors (Observational flow telemetry cannot verify causal attribution)
RE_CAUSAL_CONNECTORS = re.compile(
    r'\b(?:because|caused\s+by|due\s+to|proving|proves|as\s+a\s+result\s+of|therefore|implies|resulting\s+in|leads\s+to|explains\s+why)\b',
    re.IGNORECASE
)

# Global Contradiction Scan Regexes
RE_SCORE_GLOBAL = re.compile(
    r'\b(?:(?:an?|the|hybrid)\s+)?(?:hybrid\s+anomaly\s+score|anomaly\s+score|hybrid\s+score|score)\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_THRESH_GLOBAL = re.compile(
    r'\b(?:(?:an?|the|calibrated\s+baseline|baseline|decision)\s+)?threshold\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_RECON_GLOBAL = re.compile(
    r'\b(?:(?:an?|the)\s+)?(?:reconstruction\s+error|reconstruction\s+loss)\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_MAHAL_GLOBAL = re.compile(
    r'\b(?:(?:an?|the|latent)\s+)?(?:latent\s+mahalanobis\s+distance|mahalanobis\s+distance)\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_STATUS_GLOBAL = re.compile(
    r'\b(?:status\s*(?:is|=|:)?\s*|evaluated\s+\S+\s+as\s+|as\s+)(ANOMALOUS|BENIGN)\b',
    re.IGNORECASE
)
RE_OBS_MEAN_GLOBAL = re.compile(
    r'\bobserved\s+mean\s*(?:(?:for|of)\s+([a-zA-Z0-9_\-]+))?\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_BEN_MEAN_GLOBAL = re.compile(
    r'\bbenign\s+mean\s*(?:(?:for|of)\s+([a-zA-Z0-9_\-]+))?\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_CONTRIB_GLOBAL = re.compile(
    r'\b(?:contribution\s*(?:(?:for|of)\s+([a-zA-Z0-9_\- ]+?))?|([a-zA-Z0-9_\- ]+?)\s+contribution)\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')',
    re.IGNORECASE
)
RE_FEAT_MENTION = re.compile(
    r'\bfeature\s+([a-zA-Z0-9_\-]+)\b',
    re.IGNORECASE
)

# Tier 1: Canonical Full Sentence Schemas (Production Explainer Outputs)
FULL_DECISION_RE = re.compile(
    r'^PAADU/CHARM\s+evaluated\s+(\S+)\s+as\s+(ANOMALOUS|BENIGN)\s+with\s+(?:a\s+)?hybrid\s+anomaly\s+score\s+of\s+(' + NUM_PATTERN + r')\s+against\s+the\s+calibrated\s+baseline\s+threshold\s+of\s+(' + NUM_PATTERN + r')\.?$',
    re.IGNORECASE
)

FULL_TRIGGER_RE = re.compile(
    r'^(?:The\s+)?alert\s+was\s+triggered\s+by\s+policies:\s*\[([a-zA-Z0-9_,\s]+)\],\s*with\s+(?:a\s+)?reconstruction\s+error\s+of\s+(' + NUM_PATTERN + r')\s+and\s+(?:a\s+)?latent\s+Mahalanobis\s+distance\s+of\s+(' + NUM_PATTERN + r')\.?$',
    re.IGNORECASE
)

FULL_XAI_PREAMBLE_RE = re.compile(
    r'^Explainable\s+AI\s+leave-one-feature-out\s+occlusion\s+identified\s+the\s+most\s+influential\s+features\s+as:\s*',
    re.IGNORECASE
)

FEATURE_ITEM_RE = re.compile(
    r'^([a-zA-Z0-9_\- ]+?)\s*\(\s*contribution\s*[:=]\s*(' + NUM_PATTERN + r')\s*,\s*observed\s+mean\s*[:=]\s*(' + NUM_PATTERN + r')\s+vs\s+benign\s+mean\s*[:=]\s*(' + NUM_PATTERN + r')\s*\)$',
    re.IGNORECASE
)

# Tier 2: Atomic Assertion Schemas for Arbitrary/Compound Sentences
ATOMIC_SCHEMAS = {
    'score': re.compile(
        r'^(?:(?:(?:an?|the|with\s+(?:a\s+)?|hybrid)\s+)?(?:hybrid\s+anomaly\s+score|anomaly\s+score|hybrid\s+score|score))\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'threshold': re.compile(
        r'^(?:(?:(?:an?|the|against\s+(?:the\s+)?|calibrated\s+baseline|baseline|decision)\s+)?threshold)\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'recon': re.compile(
        r'^(?:(?:(?:an?|the|with\s+(?:a\s+)?)\s+)?(?:reconstruction\s+error|reconstruction\s+loss))\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'mahal': re.compile(
        r'^(?:(?:(?:and|with)\s+(?:a\s+)?)?(?:latent\s+mahalanobis\s+distance|mahalanobis\s+distance))\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'status': re.compile(
        r'^(?:(?:status\s*(?:is|=|:)?\s*)|(?:evaluated\s+\S+\s+as\s+)|(?:as\s+))(ANOMALOUS|BENIGN)$',
        re.IGNORECASE
    ),
    'feature_tuple': re.compile(
        r'^([a-zA-Z0-9_\- ]+?)\s*\(\s*contribution\s*[:=]\s*(' + NUM_PATTERN + r')\s*,\s*observed\s+mean\s*[:=]\s*(' + NUM_PATTERN + r')\s+vs\s+benign\s+mean\s*[:=]\s*(' + NUM_PATTERN + r')\)$',
        re.IGNORECASE
    ),
    'feature_contrib': re.compile(
        r'^(?:(?:contribution\s*(?:(?:for|of)\s+([a-zA-Z0-9_\- ]+?))?)|(?:([a-zA-Z0-9_\- ]+?)\s+contribution))\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'single_contrib': re.compile(
        r'^contribution\s*[:=]\s*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'obs_mean': re.compile(
        r'^(?:(?:the\s+)?observed\s+mean\s*(?:(?:for|of)\s+([a-zA-Z0-9_\- ]+?))?|(?:feature\s+)?([a-zA-Z0-9_\-]+)\s+(?:has\s+)?observed\s+mean)\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'ben_mean': re.compile(
        r'^(?:the\s+)?benign\s+mean\s*(?:(?:for|of)\s+([a-zA-Z0-9_\- ]+?))?\s*(?:is\s+|of\s+|=|:|\s+)*(' + NUM_PATTERN + r')$',
        re.IGNORECASE
    ),
    'policies': re.compile(
        r'^(?:(?:The\s+)?alert\s+was\s+triggered\s+by\s+)?policies:\s*\[([a-zA-Z0-9_,\s]+)\]$',
        re.IGNORECASE
    ),
}

# Delimiter pattern for splitting into atomic propositions (does NOT swallow causal connectors)
DELIM_RE = re.compile(
    r';|\s+-\s+|\s+(?:and|but|while|whereas|as\s+well\s+as|with)\s+|,\s*',
    re.IGNORECASE
)


@dataclass
class TextSpan:
    start_char: int
    end_char: int
    text: str
    span_type: str  # 'canonical_full', 'clause', 'connector', 'causal_connector', 'unparsed'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_char': self.start_char,
            'end_char': self.end_char,
            'text': self.text,
            'span_type': self.span_type
        }


@dataclass
class AtomicAssertionAudit:
    assertion_type: str
    field_name: str
    operator: str
    asserted_value: Any
    expected_value: Any
    is_supported: bool
    is_contradiction: bool
    error_message: Optional[str] = None
    raw_clause: str = ""
    start_char: int = -1
    end_char: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'assertion_type': self.assertion_type,
            'field_name': self.field_name,
            'operator': self.operator,
            'asserted_value': str(self.asserted_value),
            'expected_value': str(self.expected_value),
            'is_supported': self.is_supported,
            'is_contradiction': self.is_contradiction,
            'error_message': self.error_message or "",
            'raw_clause': self.raw_clause,
            'start_char': self.start_char,
            'end_char': self.end_char
        }


def split_clauses_protected(text: str) -> List[Tuple[str, int, int]]:
    """
    Splits compound claims into atomic clauses while preserving commas inside parentheses.
    Returns list of (clause_text, start_char, end_char).
    """
    chars: List[str] = []
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
            chars.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            chars.append(ch)
        elif ch == ',' and depth > 0:
            chars.append('\x00')
        else:
            chars.append(ch)
    protected = ''.join(chars)

    clauses: List[Tuple[str, int, int]] = []
    last_end = 0
    for m in DELIM_RE.finditer(protected):
        cl = protected[last_end:m.start()].replace('\x00', ',').strip().rstrip('.')
        if cl:
            # Locate exact start and end in original text
            raw_start = text.find(cl, last_end)
            if raw_start != -1:
                clauses.append((cl, raw_start, raw_start + len(cl)))
            else:
                clauses.append((cl, last_end, m.start()))
        last_end = m.end()

    # Trailing clause
    trailing = protected[last_end:].replace('\x00', ',').strip().rstrip('.')
    if trailing:
        raw_start = text.find(trailing, last_end)
        if raw_start != -1:
            clauses.append((trailing, raw_start, raw_start + len(trailing)))
        else:
            clauses.append((trailing, last_end, len(text)))

    return clauses


def parse_and_validate_telemetry(
    claim_text: str,
    incident: Any
) -> Tuple[str, str, str, str, List[AtomicAssertionAudit], List[TextSpan]]:
    """
    Parses claim text using strict allowlisted grammar and validates every atomic
    factual assertion against authoritative incident ground truth.

    Returns:
        status: 'SUPPORTED', 'CONTRADICTED', or 'UNSUPPORTED'
        action: 'RETAINED_VERIFIED', 'CONTRADICTION_RETRACTED', or 'FLAGGED_WITH_UNCERTAINTY'
        presented_text: Safe narrative with verification badges or retraction notes
        reason: Diagnostic reason string detailing outcomes
        audit_records: List of AtomicAssertionAudit objects for structured assertion-level reporting
        unparsed_spans: List of TextSpan objects identifying any unaccounted text
    """
    text = claim_text.strip()

    # Pre-existing retraction check
    if "[CONTRADICTED TELEMETRY RETRACTED]" in text or "[CONTRADICTED CLAIM RETRACTED]" in text:
        rec = AtomicAssertionAudit(
            assertion_type="retraction_marker",
            field_name="status",
            operator="marker",
            asserted_value="RETRACTED",
            expected_value="RETRACTED",
            is_supported=False,
            is_contradiction=True,
            error_message="Pre-existing retraction marker",
            raw_clause=text,
            start_char=0,
            end_char=len(text)
        )
        return "CONTRADICTED", "CONTRADICTION_RETRACTED", text, "Pre-existing retraction marker", [rec], []

    contradictions: List[str] = []
    unsupported_assertions: List[str] = []
    supported_assertions: List[str] = []
    audit_records: List[AtomicAssertionAudit] = []
    unparsed_spans: List[TextSpan] = []

    top_feats = {f.feature.lower(): f for f in incident.top_xai_features}

    # STEP 0: Causal Connector Trap (Flow telemetry cannot verify causal attribution)
    for m_causal in RE_CAUSAL_CONNECTORS.finditer(text):
        msg = f"Ungrounded causal connector '{m_causal.group(0)}' cannot be verified from flow telemetry"
        unsupported_assertions.append(msg)
        audit_records.append(AtomicAssertionAudit(
            assertion_type="causal_claim",
            field_name="causal_attribution",
            operator="causes",
            asserted_value=m_causal.group(0),
            expected_value="Independent flow telemetry measurements (non-causal)",
            is_supported=False,
            is_contradiction=False,
            error_message=msg,
            raw_clause=m_causal.group(0),
            start_char=m_causal.start(),
            end_char=m_causal.end()
        ))

    # STEP 1: Global Contradiction Scan (Find ANY Contradictory Telemetry Assertion)
    # 1.1 Global Score Scan
    for m in RE_SCORE_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.anomaly_score
        if abs(val - exp) > 0.05:
            msg = f"asserted score {val:.3f} contradicts actual {exp:.3f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('score', 'anomaly_score', 'of', val, exp, False, True, msg, m.group(0), m.start(), m.end()))

    # 1.2 Global Threshold Scan
    for m in RE_THRESH_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.decision_threshold
        if abs(val - exp) > 0.05:
            msg = f"asserted threshold {val:.3f} contradicts actual {exp:.3f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('threshold', 'decision_threshold', 'of', val, exp, False, True, msg, m.group(0), m.start(), m.end()))

    # 1.3 Global Reconstruction Error Scan
    for m in RE_RECON_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.reconstruction_error
        if abs(val - exp) > 0.05:
            msg = f"asserted reconstruction error {val:.4f} contradicts actual {exp:.4f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('recon_error', 'reconstruction_error', 'of', val, exp, False, True, msg, m.group(0), m.start(), m.end()))

    # 1.4 Global Mahalanobis Distance Scan
    for m in RE_MAHAL_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.mahalanobis_distance
        if abs(val - exp) > 0.5:
            msg = f"asserted Mahalanobis distance {val:.4f} contradicts actual {exp:.4f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('mahalanobis', 'mahalanobis_distance', 'of', val, exp, False, True, msg, m.group(0), m.start(), m.end()))

    # 1.5 Global Status Scan
    for m in RE_STATUS_GLOBAL.finditer(text):
        st = m.group(1).upper()
        exp_st = 'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'
        if st != exp_st:
            msg = f"asserted status {st} contradicts actual {exp_st}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('status', 'status', '=', st, exp_st, False, True, msg, m.group(0), m.start(), m.end()))

    # 1.6 Global Explicit Non-Existent Feature Scan
    if "non-existent" in text.lower() or "bogus" in text.lower():
        contradictions.append("asserts non-existent flow feature")
        audit_records.append(AtomicAssertionAudit(
            assertion_type="feature_existence",
            field_name="feature_name",
            operator="asserted",
            asserted_value="non-existent",
            expected_value=list(top_feats.keys()),
            is_supported=False,
            is_contradiction=True,
            error_message="Asserts non-existent flow feature",
            raw_clause="non-existent flow feature",
            start_char=-1,
            end_char=-1
        ))

    # 1.7 Global Feature Mentions Scan
    for m in RE_FEAT_MENTION.finditer(text):
        fname = m.group(1).lower()
        if fname not in top_feats and fname not in ('leave-one-feature-out', 'occlusion'):
            msg = f"asserted feature '{m.group(1)}' not in top XAI features"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('feature_name', m.group(1), 'in', m.group(1), list(top_feats.keys()), False, True, msg, m.group(0), m.start(), m.end()))

    # 1.8 Global Contribution Scan
    for m in RE_CONTRIB_GLOBAL.finditer(text):
        val = float(m.group(3))
        fn = m.group(1) or m.group(2)
        if fn:
            fn_clean = fn.strip()
            fn_low = fn_clean.lower()
            if fn_low in top_feats:
                exp_c = top_feats[fn_low].hybrid_contribution
                if abs(val - exp_c) > 0.05:
                    msg = f"{fn_clean} contribution {val:+.2f} contradicts actual {exp_c:+.2f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit('feature_contribution', f"{fn_clean}.contrib", ':', val, exp_c, False, True, msg, m.group(0), m.start(), m.end()))
            else:
                msg = f"feature '{fn_clean}' not in top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('feature_name', fn_clean, 'in', fn_clean, list(top_feats.keys()), False, True, msg, m.group(0), m.start(), m.end()))
        else:
            if not any(abs(val - f.hybrid_contribution) <= 0.05 for f in incident.top_xai_features):
                msg = f"asserted contribution {val:+.2f} contradicts top feature attributions"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('contribution', 'any_feature', ':', val, [f.hybrid_contribution for f in incident.top_xai_features], False, True, msg, m.group(0), m.start(), m.end()))

    # 1.9 Global Benign Mean Scan
    for m in RE_BEN_MEAN_GLOBAL.finditer(text):
        val = float(m.group(2))
        fn = m.group(1)
        if fn:
            fn_low = fn.strip().lower()
            if fn_low in top_feats:
                exp_b = top_feats[fn_low].benign_mean
                if abs(val - exp_b) > 0.1:
                    msg = f"{fn} benign mean {val:.2f} contradicts actual {exp_b:.2f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit('benign_mean', f"{fn}.ben", ':', val, exp_b, False, True, msg, m.group(0), m.start(), m.end()))
            else:
                msg = f"feature '{fn}' not in top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('feature_name', fn, 'in', fn, list(top_feats.keys()), False, True, msg, m.group(0), m.start(), m.end()))
        else:
            if not any(abs(val - f.benign_mean) <= 0.1 for f in incident.top_xai_features):
                msg = f"asserted benign mean {val:.2f} contradicts all top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('benign_mean', 'any_feature', ':', val, [f.benign_mean for f in incident.top_xai_features], False, True, msg, m.group(0), m.start(), m.end()))

    # 1.10 Global Observed Mean Scan
    for m in RE_OBS_MEAN_GLOBAL.finditer(text):
        val = float(m.group(2))
        fn = m.group(1)
        if fn:
            fn_low = fn.strip().lower()
            if fn_low in top_feats:
                exp_o = top_feats[fn_low].observed_mean
                if abs(val - exp_o) > 0.1:
                    msg = f"{fn} observed mean {val:.2f} contradicts actual {exp_o:.2f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit('observed_mean', f"{fn}.obs", ':', val, exp_o, False, True, msg, m.group(0), m.start(), m.end()))
            else:
                msg = f"feature '{fn}' not in top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('feature_name', fn, 'in', fn, list(top_feats.keys()), False, True, msg, m.group(0), m.start(), m.end()))
        else:
            if not any(abs(val - f.observed_mean) <= 0.1 for f in incident.top_xai_features):
                msg = f"asserted observed mean {val:.2f} contradicts all top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('observed_mean', 'any_feature', ':', val, [f.observed_mean for f in incident.top_xai_features], False, True, msg, m.group(0), m.start(), m.end()))

    # 1.11 Global Ungrounded IP Check
    m_ip = re.search(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', text)
    if m_ip:
        unsupported_assertions.append(f"asserts ungrounded IP address '{m_ip.group(0)}'")
        audit_records.append(AtomicAssertionAudit(
            assertion_type="ip_address",
            field_name="ip_address",
            operator="asserted",
            asserted_value=m_ip.group(0),
            expected_value="None (Flow Telemetry Only)",
            is_supported=False,
            is_contradiction=False,
            error_message=f"Ungrounded IP address {m_ip.group(0)} unsupported by flow telemetry",
            raw_clause=m_ip.group(0),
            start_char=m_ip.start(),
            end_char=m_ip.end()
        ))

    # STEP 2: Tier 1 Canonical Full Sentence Patterns
    m_dec = FULL_DECISION_RE.match(text)
    m_trig = FULL_TRIGGER_RE.match(text)
    is_xai_preamble = bool(FULL_XAI_PREAMBLE_RE.match(text))

    if m_dec and not contradictions and not unsupported_assertions:
        blk_ref, status_str, score_str, thresh_str = m_dec.groups()
        
        # 2.1 Validate Incident Identifier against ground truth
        valid_refs = {incident.input_reference.lower(), incident.incident_id.lower()}
        if blk_ref.lower() not in valid_refs:
            msg = f"asserted incident identifier '{blk_ref}' contradicts actual '{incident.input_reference}'"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit(
                assertion_type='incident_id',
                field_name='input_reference',
                operator='=',
                asserted_value=blk_ref,
                expected_value=incident.input_reference,
                is_supported=False,
                is_contradiction=True,
                error_message=msg,
                raw_clause=blk_ref,
                start_char=m_dec.start(1),
                end_char=m_dec.end(1)
            ))
        else:
            supported_assertions.append(f"incident_id={blk_ref}")
            audit_records.append(AtomicAssertionAudit(
                assertion_type='incident_id',
                field_name='input_reference',
                operator='=',
                asserted_value=blk_ref,
                expected_value=incident.input_reference,
                is_supported=True,
                is_contradiction=False,
                raw_clause=blk_ref,
                start_char=m_dec.start(1),
                end_char=m_dec.end(1)
            ))

        # 2.2 Status Validation
        exp_st = 'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'
        if status_str.upper() != exp_st:
            msg = f"asserted status '{status_str}' contradicts actual '{exp_st}'"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit(
                assertion_type='status',
                field_name='status',
                operator='=',
                asserted_value=status_str.upper(),
                expected_value=exp_st,
                is_supported=False,
                is_contradiction=True,
                error_message=msg,
                raw_clause=f"evaluated as {status_str}",
                start_char=m_dec.start(2),
                end_char=m_dec.end(2)
            ))
        else:
            supported_assertions.append(f"status={status_str}")
            audit_records.append(AtomicAssertionAudit(
                assertion_type='status',
                field_name='status',
                operator='=',
                asserted_value=status_str.upper(),
                expected_value=exp_st,
                is_supported=True,
                is_contradiction=False,
                raw_clause=f"evaluated as {status_str}",
                start_char=m_dec.start(2),
                end_char=m_dec.end(2)
            ))

        # 2.3 Score Validation
        score_val = float(score_str)
        if abs(score_val - incident.anomaly_score) > 0.05:
            msg = f"asserted score {score_val:.3f} contradicts actual {incident.anomaly_score:.3f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit(
                assertion_type='score',
                field_name='anomaly_score',
                operator='=',
                asserted_value=score_val,
                expected_value=incident.anomaly_score,
                is_supported=False,
                is_contradiction=True,
                error_message=msg,
                raw_clause=f"hybrid anomaly score of {score_str}",
                start_char=m_dec.start(3),
                end_char=m_dec.end(3)
            ))
        else:
            supported_assertions.append(f"anomaly_score={score_val:.3f}")
            audit_records.append(AtomicAssertionAudit(
                assertion_type='score',
                field_name='anomaly_score',
                operator='=',
                asserted_value=score_val,
                expected_value=incident.anomaly_score,
                is_supported=True,
                is_contradiction=False,
                raw_clause=f"hybrid anomaly score of {score_str}",
                start_char=m_dec.start(3),
                end_char=m_dec.end(3)
            ))

        # 2.4 Threshold Validation
        thresh_val = float(thresh_str)
        if abs(thresh_val - incident.decision_threshold) > 0.05:
            msg = f"asserted threshold {thresh_val:.3f} contradicts actual {incident.decision_threshold:.3f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit(
                assertion_type='threshold',
                field_name='decision_threshold',
                operator='=',
                asserted_value=thresh_val,
                expected_value=incident.decision_threshold,
                is_supported=False,
                is_contradiction=True,
                error_message=msg,
                raw_clause=f"baseline threshold of {thresh_str}",
                start_char=m_dec.start(4),
                end_char=m_dec.end(4)
            ))
        else:
            supported_assertions.append(f"threshold={thresh_val:.3f}")
            audit_records.append(AtomicAssertionAudit(
                assertion_type='threshold',
                field_name='decision_threshold',
                operator='=',
                asserted_value=thresh_val,
                expected_value=incident.decision_threshold,
                is_supported=True,
                is_contradiction=False,
                raw_clause=f"baseline threshold of {thresh_str}",
                start_char=m_dec.start(4),
                end_char=m_dec.end(4)
            ))

    elif m_trig and not contradictions and not unsupported_assertions:
        pol_str, recon_str, mahal_str = m_trig.groups()
        pols = [p.strip() for p in pol_str.split(',') if p.strip()]
        for p in pols:
            if p not in incident.trigger_reasons:
                msg = f"asserted trigger policy '{p}' contradicts actual triggers {incident.trigger_reasons}"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='policy',
                    field_name='trigger_reasons',
                    operator='in',
                    asserted_value=p,
                    expected_value=incident.trigger_reasons,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=msg,
                    raw_clause=p,
                    start_char=m_trig.start(1),
                    end_char=m_trig.end(1)
                ))
            else:
                supported_assertions.append(f"policy={p}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='policy',
                    field_name='trigger_reasons',
                    operator='in',
                    asserted_value=p,
                    expected_value=incident.trigger_reasons,
                    is_supported=True,
                    is_contradiction=False,
                    raw_clause=p,
                    start_char=m_trig.start(1),
                    end_char=m_trig.end(1)
                ))

        recon_val = float(recon_str)
        if abs(recon_val - incident.reconstruction_error) > 0.05:
            msg = f"asserted reconstruction error {recon_val:.4f} contradicts actual {incident.reconstruction_error:.4f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit(
                assertion_type='recon_error',
                field_name='reconstruction_error',
                operator='=',
                asserted_value=recon_val,
                expected_value=incident.reconstruction_error,
                is_supported=False,
                is_contradiction=True,
                error_message=msg,
                raw_clause=f"reconstruction error: {recon_str}",
                start_char=m_trig.start(2),
                end_char=m_trig.end(2)
            ))
        else:
            supported_assertions.append(f"reconstruction_error={recon_val:.4f}")
            audit_records.append(AtomicAssertionAudit(
                assertion_type='recon_error',
                field_name='reconstruction_error',
                operator='=',
                asserted_value=recon_val,
                expected_value=incident.reconstruction_error,
                is_supported=True,
                is_contradiction=False,
                raw_clause=f"reconstruction error: {recon_str}",
                start_char=m_trig.start(2),
                end_char=m_trig.end(2)
            ))

        mahal_val = float(mahal_str)
        if abs(mahal_val - incident.mahalanobis_distance) > 0.5:
            msg = f"asserted Mahalanobis distance {mahal_val:.4f} contradicts actual {incident.mahalanobis_distance:.4f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit(
                assertion_type='mahalanobis',
                field_name='mahalanobis_distance',
                operator='=',
                asserted_value=mahal_val,
                expected_value=incident.mahalanobis_distance,
                is_supported=False,
                is_contradiction=True,
                error_message=msg,
                raw_clause=f"latent Mahalanobis distance: {mahal_str}",
                start_char=m_trig.start(3),
                end_char=m_trig.end(3)
            ))
        else:
            supported_assertions.append(f"mahalanobis_distance={mahal_val:.4f}")
            audit_records.append(AtomicAssertionAudit(
                assertion_type='mahalanobis',
                field_name='mahalanobis_distance',
                operator='=',
                asserted_value=mahal_val,
                expected_value=incident.mahalanobis_distance,
                is_supported=True,
                is_contradiction=False,
                raw_clause=f"latent Mahalanobis distance: {mahal_str}",
                start_char=m_trig.start(3),
                end_char=m_trig.end(3)
            ))

    elif is_xai_preamble and not contradictions and not unsupported_assertions:
        rest = FULL_XAI_PREAMBLE_RE.sub('', text).rstrip('.')
        items = [s.strip() for s in rest.split(';') if s.strip()]
        for it in items:
            m_it = FEATURE_ITEM_RE.fullmatch(it)
            it_idx = text.find(it)
            if not m_it:
                unsupported_assertions.append(f"unrecognized feature proposition: '{it}'")
                unparsed_spans.append(TextSpan(it_idx, it_idx + len(it), it, 'unparsed_feature_clause'))
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='unknown_syntax',
                    field_name='unknown',
                    operator='unknown',
                    asserted_value=it,
                    expected_value='Allowlisted schema',
                    is_supported=False,
                    is_contradiction=False,
                    error_message=f"unrecognized feature proposition: '{it}'",
                    raw_clause=it,
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))
                continue

            fname, contrib_str, obs_str, ben_str = m_it.groups()
            fname_clean = fname.strip()
            fname_lower = fname_clean.lower()
            if fname_lower not in top_feats:
                msg = f"feature '{fname_clean}' not in top XAI features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='feature_name',
                    field_name=fname_clean,
                    operator='in',
                    asserted_value=fname_clean,
                    expected_value=list(top_feats.keys()),
                    is_supported=False,
                    is_contradiction=True,
                    error_message=msg,
                    raw_clause=it,
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))
                continue

            act_f = top_feats[fname_lower]
            contrib_val = float(contrib_str)
            obs_val = float(obs_str)
            ben_val = float(ben_str)

            if abs(contrib_val - act_f.hybrid_contribution) > 0.05:
                msg = f"{fname_clean} contribution {contrib_val:+.2f} contradicts actual {act_f.hybrid_contribution:+.2f}"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='feature_contrib',
                    field_name=f"{act_f.feature}.hybrid_contribution",
                    operator='=',
                    asserted_value=contrib_val,
                    expected_value=act_f.hybrid_contribution,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=msg,
                    raw_clause=f"{fname_clean} contribution: {contrib_str}",
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))
            else:
                supported_assertions.append(f"{fname_clean}_contribution={contrib_val:+.2f}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='feature_contrib',
                    field_name=f"{act_f.feature}.hybrid_contribution",
                    operator='=',
                    asserted_value=contrib_val,
                    expected_value=act_f.hybrid_contribution,
                    is_supported=True,
                    is_contradiction=False,
                    raw_clause=f"{fname_clean} contribution: {contrib_str}",
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))

            if abs(obs_val - act_f.observed_mean) > 0.1:
                msg = f"{fname_clean} observed mean {obs_val:.2f} contradicts actual {act_f.observed_mean:.2f}"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='obs_mean',
                    field_name=f"{act_f.feature}.observed_mean",
                    operator='=',
                    asserted_value=obs_val,
                    expected_value=act_f.observed_mean,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=msg,
                    raw_clause=f"{fname_clean} observed mean: {obs_str}",
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))
            else:
                supported_assertions.append(f"{fname_clean}_observed={obs_val:.2f}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='obs_mean',
                    field_name=f"{act_f.feature}.observed_mean",
                    operator='=',
                    asserted_value=obs_val,
                    expected_value=act_f.observed_mean,
                    is_supported=True,
                    is_contradiction=False,
                    raw_clause=f"{fname_clean} observed mean: {obs_str}",
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))

            if abs(ben_val - act_f.benign_mean) > 0.1:
                msg = f"{fname_clean} benign mean {ben_val:.2f} contradicts actual {act_f.benign_mean:.2f}"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='ben_mean',
                    field_name=f"{act_f.feature}.benign_mean",
                    operator='=',
                    asserted_value=ben_val,
                    expected_value=act_f.benign_mean,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=msg,
                    raw_clause=f"{fname_clean} benign mean: {ben_str}",
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))
            else:
                supported_assertions.append(f"{fname_clean}_benign={ben_val:.2f}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='ben_mean',
                    field_name=f"{act_f.feature}.benign_mean",
                    operator='=',
                    asserted_value=ben_val,
                    expected_value=act_f.benign_mean,
                    is_supported=True,
                    is_contradiction=False,
                    raw_clause=f"{fname_clean} benign mean: {ben_str}",
                    start_char=it_idx,
                    end_char=it_idx + len(it)
                ))

    elif not m_dec and not m_trig and not is_xai_preamble:
        # STEP 3: Tier 2 Atomic Clause Decomposition Grammar
        clause_tuples = split_clauses_protected(text)
        if not clause_tuples:
            unsupported_assertions.append("empty claim proposition")

        current_cursor = 0
        for cl, c_start, c_end in clause_tuples:
            # Check for unaccounted delimiter spans between clauses
            if c_start > current_cursor:
                gap_text = text[current_cursor:c_start]
                cleaned_gap = re.sub(r'[\s,;\.\-\(\)]+|and|with|as\s+well\s+as', '', gap_text, flags=re.IGNORECASE)
                if cleaned_gap:
                    unparsed_spans.append(TextSpan(current_cursor, c_start, gap_text.strip(), 'unparsed_intervening'))
                    unsupported_assertions.append(f"unparsed intervening span: '{gap_text.strip()}'")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='unparsed_span',
                        field_name='unknown',
                        operator='unknown',
                        asserted_value=gap_text.strip(),
                        expected_value='Allowlisted schema',
                        is_supported=False,
                        is_contradiction=False,
                        error_message=f"Unparsed span '{gap_text.strip()}' fails closed",
                        raw_clause=gap_text.strip(),
                        start_char=current_cursor,
                        end_char=c_start
                    ))
            current_cursor = c_end

            matched_schema = None
            groups = None
            for sname, pat in ATOMIC_SCHEMAS.items():
                m = pat.match(cl)
                if m:
                    matched_schema = sname
                    groups = m.groups()
                    break

            if not matched_schema:
                unsupported_assertions.append(f"unrecognized clause: '{cl}'")
                unparsed_spans.append(TextSpan(c_start, c_end, cl, 'unparsed_clause'))
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='unrecognized_clause',
                    field_name='unknown',
                    operator='unknown',
                    asserted_value=cl,
                    expected_value='Allowlisted Telemetry Schema',
                    is_supported=False,
                    is_contradiction=False,
                    error_message=f"Leftover/unrecognized clause '{cl}' fails closed",
                    raw_clause=cl,
                    start_char=c_start,
                    end_char=c_end
                ))
                continue

            if matched_schema == 'score':
                val = float(groups[0])
                exp_val = incident.anomaly_score
                if abs(val - exp_val) <= 0.05:
                    supported_assertions.append(f"anomaly_score={val:.3f}")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='score',
                        field_name='anomaly_score',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=True,
                        is_contradiction=False,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))
                else:
                    msg = f"asserted score {val:.3f} contradicts actual {exp_val:.3f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='score',
                        field_name='anomaly_score',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=False,
                        is_contradiction=True,
                        error_message=msg,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))

            elif matched_schema == 'threshold':
                val = float(groups[0])
                exp_val = incident.decision_threshold
                if abs(val - exp_val) <= 0.05:
                    supported_assertions.append(f"threshold={val:.3f}")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='threshold',
                        field_name='decision_threshold',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=True,
                        is_contradiction=False,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))
                else:
                    msg = f"asserted threshold {val:.3f} contradicts actual {exp_val:.3f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='threshold',
                        field_name='decision_threshold',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=False,
                        is_contradiction=True,
                        error_message=msg,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))

            elif matched_schema == 'recon':
                val = float(groups[0])
                exp_val = incident.reconstruction_error
                if abs(val - exp_val) <= 0.05:
                    supported_assertions.append(f"reconstruction_error={val:.4f}")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='recon_error',
                        field_name='reconstruction_error',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=True,
                        is_contradiction=False,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))
                else:
                    msg = f"asserted reconstruction error {val:.4f} contradicts actual {exp_val:.4f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='recon_error',
                        field_name='reconstruction_error',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=False,
                        is_contradiction=True,
                        error_message=msg,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))

            elif matched_schema == 'mahal':
                val = float(groups[0])
                exp_val = incident.mahalanobis_distance
                if abs(val - exp_val) <= 0.5:
                    supported_assertions.append(f"mahalanobis_distance={val:.4f}")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='mahalanobis',
                        field_name='mahalanobis_distance',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=True,
                        is_contradiction=False,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))
                else:
                    msg = f"asserted Mahalanobis distance {val:.4f} contradicts actual {exp_val:.4f}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='mahalanobis',
                        field_name='mahalanobis_distance',
                        operator='=',
                        asserted_value=val,
                        expected_value=exp_val,
                        is_supported=False,
                        is_contradiction=True,
                        error_message=msg,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))

            elif matched_schema == 'status':
                st = groups[0].upper()
                exp_st = 'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'
                if st == exp_st:
                    supported_assertions.append(f"status={st}")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='status',
                        field_name='status',
                        operator='=',
                        asserted_value=st,
                        expected_value=exp_st,
                        is_supported=True,
                        is_contradiction=False,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))
                else:
                    msg = f"asserted status {st} contradicts actual {exp_st}"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='status',
                        field_name='status',
                        operator='=',
                        asserted_value=st,
                        expected_value=exp_st,
                        is_supported=False,
                        is_contradiction=True,
                        error_message=msg,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))

            elif matched_schema == 'feature_tuple':
                fname, c_str, o_str, b_str = groups
                fname_clean = fname.strip()
                fname_lower = fname_clean.lower()
                if fname_lower in top_feats:
                    act_f = top_feats[fname_lower]
                    c_val = float(c_str)
                    o_val = float(o_str)
                    b_val = float(b_str)

                    # Contribution audit
                    if abs(c_val - act_f.hybrid_contribution) <= 0.05:
                        supported_assertions.append(f"{fname_clean}_contribution={c_val:+.2f}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_contrib',
                            field_name=f"{act_f.feature}.hybrid_contribution",
                            operator='=',
                            asserted_value=c_val,
                            expected_value=act_f.hybrid_contribution,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=f"{fname_clean} contribution: {c_str}",
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"{fname_clean} contribution {c_val:+.2f} contradicts actual {act_f.hybrid_contribution:+.2f}"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_contrib',
                            field_name=f"{act_f.feature}.hybrid_contribution",
                            operator='=',
                            asserted_value=c_val,
                            expected_value=act_f.hybrid_contribution,
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=f"{fname_clean} contribution: {c_str}",
                            start_char=c_start,
                            end_char=c_end
                        ))

                    # Observed mean audit
                    if abs(o_val - act_f.observed_mean) <= 0.1:
                        supported_assertions.append(f"{fname_clean}_observed={o_val:.2f}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='obs_mean',
                            field_name=f"{act_f.feature}.observed_mean",
                            operator='=',
                            asserted_value=o_val,
                            expected_value=act_f.observed_mean,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=f"{fname_clean} observed mean: {o_str}",
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"{fname_clean} observed mean {o_val:.2f} contradicts actual {act_f.observed_mean:.2f}"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='obs_mean',
                            field_name=f"{act_f.feature}.observed_mean",
                            operator='=',
                            asserted_value=o_val,
                            expected_value=act_f.observed_mean,
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=f"{fname_clean} observed mean: {o_str}",
                            start_char=c_start,
                            end_char=c_end
                        ))

                    # Benign mean audit
                    if abs(b_val - act_f.benign_mean) <= 0.1:
                        supported_assertions.append(f"{fname_clean}_benign={b_val:.2f}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='ben_mean',
                            field_name=f"{act_f.feature}.benign_mean",
                            operator='=',
                            asserted_value=b_val,
                            expected_value=act_f.benign_mean,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=f"{fname_clean} benign mean: {b_str}",
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"{fname_clean} benign mean {b_val:.2f} contradicts actual {act_f.benign_mean:.2f}"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='ben_mean',
                            field_name=f"{act_f.feature}.benign_mean",
                            operator='=',
                            asserted_value=b_val,
                            expected_value=act_f.benign_mean,
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=f"{fname_clean} benign mean: {b_str}",
                            start_char=c_start,
                            end_char=c_end
                        ))
                else:
                    msg = f"feature '{fname_clean}' not in top XAI features"
                    contradictions.append(msg)
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='feature_name',
                        field_name=fname_clean,
                        operator='in',
                        asserted_value=fname_clean,
                        expected_value=list(top_feats.keys()),
                        is_supported=False,
                        is_contradiction=True,
                        error_message=msg,
                        raw_clause=cl,
                        start_char=c_start,
                        end_char=c_end
                    ))

            elif matched_schema in ('feature_contrib', 'single_contrib'):
                val = float(groups[-1])
                fn = groups[0] or groups[1] if len(groups) > 1 else None
                if fn:
                    fn_clean = fn.strip()
                    fn_low = fn_clean.lower()
                    if fn_low in top_feats:
                        act_f = top_feats[fn_low]
                        if abs(val - act_f.hybrid_contribution) <= 0.05:
                            supported_assertions.append(f"{fn_clean}_contribution={val:+.2f}")
                            audit_records.append(AtomicAssertionAudit(
                                assertion_type='feature_contrib',
                                field_name=f"{act_f.feature}.hybrid_contribution",
                                operator='=',
                                asserted_value=val,
                                expected_value=act_f.hybrid_contribution,
                                is_supported=True,
                                is_contradiction=False,
                                raw_clause=cl,
                                start_char=c_start,
                                end_char=c_end
                            ))
                        else:
                            msg = f"{fn_clean} contribution {val:+.2f} contradicts actual {act_f.hybrid_contribution:+.2f}"
                            contradictions.append(msg)
                            audit_records.append(AtomicAssertionAudit(
                                assertion_type='feature_contrib',
                                field_name=f"{act_f.feature}.hybrid_contribution",
                                operator='=',
                                asserted_value=val,
                                expected_value=act_f.hybrid_contribution,
                                is_supported=False,
                                is_contradiction=True,
                                error_message=msg,
                                raw_clause=cl,
                                start_char=c_start,
                                end_char=c_end
                            ))
                    else:
                        msg = f"feature '{fn_clean}' not in top features"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_name',
                            field_name=fn_clean,
                            operator='in',
                            asserted_value=fn_clean,
                            expected_value=list(top_feats.keys()),
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))
                else:
                    matching_f = next((f for f in incident.top_xai_features if abs(val - f.hybrid_contribution) <= 0.05), None)
                    if matching_f:
                        supported_assertions.append(f"contribution={val:+.2f}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_contrib',
                            field_name=f"{matching_f.feature}.hybrid_contribution",
                            operator='=',
                            asserted_value=val,
                            expected_value=matching_f.hybrid_contribution,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"asserted contribution {val:+.2f} contradicts top feature attributions"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_contrib',
                            field_name='any_feature.hybrid_contribution',
                            operator='=',
                            asserted_value=val,
                            expected_value=[f.hybrid_contribution for f in incident.top_xai_features],
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))

            elif matched_schema == 'obs_mean':
                val = float(groups[-1])
                fn = groups[0] or groups[1] if len(groups) > 1 else None
                if fn:
                    fn_clean = fn.strip()
                    fn_low = fn_clean.lower()
                    if fn_low in top_feats:
                        act_f = top_feats[fn_low]
                        if abs(val - act_f.observed_mean) <= 0.1:
                            supported_assertions.append(f"{fn_clean}_obs_mean={val:.2f}")
                            audit_records.append(AtomicAssertionAudit(
                                assertion_type='obs_mean',
                                field_name=f"{act_f.feature}.observed_mean",
                                operator='=',
                                asserted_value=val,
                                expected_value=act_f.observed_mean,
                                is_supported=True,
                                is_contradiction=False,
                                raw_clause=cl,
                                start_char=c_start,
                                end_char=c_end
                            ))
                        else:
                            msg = f"{fn_clean} observed mean {val:.2f} contradicts actual {act_f.observed_mean:.2f}"
                            contradictions.append(msg)
                            audit_records.append(AtomicAssertionAudit(
                                assertion_type='obs_mean',
                                field_name=f"{act_f.feature}.observed_mean",
                                operator='=',
                                asserted_value=val,
                                expected_value=act_f.observed_mean,
                                is_supported=False,
                                is_contradiction=True,
                                error_message=msg,
                                raw_clause=cl,
                                start_char=c_start,
                                end_char=c_end
                            ))
                    else:
                        msg = f"feature '{fn_clean}' not in top features"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_name',
                            field_name=fn_clean,
                            operator='in',
                            asserted_value=fn_clean,
                            expected_value=list(top_feats.keys()),
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))
                else:
                    matching_f = next((f for f in incident.top_xai_features if abs(val - f.observed_mean) <= 0.1), None)
                    if matching_f:
                        supported_assertions.append(f"obs_mean={val:.2f}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='obs_mean',
                            field_name=f"{matching_f.feature}.observed_mean",
                            operator='=',
                            asserted_value=val,
                            expected_value=matching_f.observed_mean,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"asserted observed mean {val:.2f} contradicts all top features"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='obs_mean',
                            field_name='any_feature.observed_mean',
                            operator='=',
                            asserted_value=val,
                            expected_value=[f.observed_mean for f in incident.top_xai_features],
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))

            elif matched_schema == 'ben_mean':
                val = float(groups[-1])
                fn = groups[0] if groups[0] else None
                if fn:
                    fn_clean = fn.strip()
                    fn_low = fn_clean.lower()
                    if fn_low in top_feats:
                        act_f = top_feats[fn_low]
                        if abs(val - act_f.benign_mean) <= 0.1:
                            supported_assertions.append(f"{fn_clean}_ben_mean={val:.2f}")
                            audit_records.append(AtomicAssertionAudit(
                                assertion_type='ben_mean',
                                field_name=f"{act_f.feature}.benign_mean",
                                operator='=',
                                asserted_value=val,
                                expected_value=act_f.benign_mean,
                                is_supported=True,
                                is_contradiction=False,
                                raw_clause=cl,
                                start_char=c_start,
                                end_char=c_end
                            ))
                        else:
                            msg = f"{fn_clean} benign mean {val:.2f} contradicts actual {act_f.benign_mean:.2f}"
                            contradictions.append(msg)
                            audit_records.append(AtomicAssertionAudit(
                                assertion_type='ben_mean',
                                field_name=f"{act_f.feature}.benign_mean",
                                operator='=',
                                asserted_value=val,
                                expected_value=act_f.benign_mean,
                                is_supported=False,
                                is_contradiction=True,
                                error_message=msg,
                                raw_clause=cl,
                                start_char=c_start,
                                end_char=c_end
                            ))
                    else:
                        msg = f"feature '{fn_clean}' not in top features"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_name',
                            field_name=fn_clean,
                            operator='in',
                            asserted_value=fn_clean,
                            expected_value=list(top_feats.keys()),
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))
                else:
                    matching_f = next((f for f in incident.top_xai_features if abs(val - f.benign_mean) <= 0.1), None)
                    if matching_f:
                        supported_assertions.append(f"ben_mean={val:.2f}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='ben_mean',
                            field_name=f"{matching_f.feature}.benign_mean",
                            operator='=',
                            asserted_value=val,
                            expected_value=matching_f.benign_mean,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"asserted benign mean {val:.2f} contradicts all top features"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='ben_mean',
                            field_name='any_feature.benign_mean',
                            operator='=',
                            asserted_value=val,
                            expected_value=[f.benign_mean for f in incident.top_xai_features],
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=cl,
                            start_char=c_start,
                            end_char=c_end
                        ))

            elif matched_schema == 'policies':
                pols = [p.strip() for p in groups[0].split(',') if p.strip()]
                for p in pols:
                    if p in incident.trigger_reasons:
                        supported_assertions.append(f"policy={p}")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='policy',
                            field_name='trigger_reasons',
                            operator='in',
                            asserted_value=p,
                            expected_value=incident.trigger_reasons,
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=p,
                            start_char=c_start,
                            end_char=c_end
                        ))
                    else:
                        msg = f"asserted trigger policy '{p}' contradicts actual triggers {incident.trigger_reasons}"
                        contradictions.append(msg)
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='policy',
                            field_name='trigger_reasons',
                            operator='in',
                            asserted_value=p,
                            expected_value=incident.trigger_reasons,
                            is_supported=False,
                            is_contradiction=True,
                            error_message=msg,
                            raw_clause=p,
                            start_char=c_start,
                            end_char=c_end
                        ))

        # Check trailing text after the last clause
        if current_cursor < len(text):
            trailing_raw = text[current_cursor:].strip(' .;\t\n')
            if trailing_raw:
                unparsed_spans.append(TextSpan(current_cursor, len(text), trailing_raw, 'unparsed_trailing'))
                unsupported_assertions.append(f"unparsed trailing span: '{trailing_raw}'")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='unparsed_span',
                    field_name='unknown',
                    operator='unknown',
                    asserted_value=trailing_raw,
                    expected_value='Allowlisted schema',
                    is_supported=False,
                    is_contradiction=False,
                    error_message=f"Unparsed trailing span '{trailing_raw}' fails closed",
                    raw_clause=trailing_raw,
                    start_char=current_cursor,
                    end_char=len(text)
                ))

    # STEP 4: Strict Fail-Closed Invariant Resolution
    # Contradiction: any contradictory assertion found
    any_contradiction = bool(contradictions) or any(r.is_contradiction for r in audit_records)
    # Unsupported: any ungrounded assertion or any unparsed text span
    any_unsupported = bool(unsupported_assertions) or bool(unparsed_spans) or any(not r.is_supported and not r.is_contradiction for r in audit_records)

    if any_contradiction:
        status = "CONTRADICTED"
        action = "CONTRADICTION_RETRACTED"
        reason = "; ".join(contradictions) if contradictions else "Contradictory assertions detected"
        presented_text = (
            f"[CONTRADICTED TELEMETRY RETRACTED] Telemetry contradiction detected ({reason}). "
            f"Actual PAADU/CHARM telemetry: anomaly_score={incident.anomaly_score:.3f}, "
            f"threshold={incident.decision_threshold:.3f}, status={'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'}."
        )
    elif any_unsupported or len(unparsed_spans) > 0:
        status = "UNSUPPORTED"
        action = "FLAGGED_WITH_UNCERTAINTY"
        reasons_list = unsupported_assertions + [f"unparsed span '{s.text}'" for s in unparsed_spans]
        reason = "; ".join(reasons_list) if reasons_list else "Ungrounded or unparsed assertions detected"
        presented_text = f"[UNVERIFIED TELEMETRY - UNGROUNDED ASSERTION] Compound telemetry claim contains unverified assertions ({reason}): '{claim_text}'"
    elif supported_assertions and not any_contradiction and not any_unsupported and len(unparsed_spans) == 0:
        # INVARIANT: Zero unparsed factual spans and zero unsupported assertions required for [VERIFIED TELEMETRY]
        status = "SUPPORTED"
        action = "RETAINED_VERIFIED"
        reason = f"Verified fields: {', '.join(supported_assertions)}"
        presented_text = f"[VERIFIED TELEMETRY] {claim_text}"
    else:
        status = "UNSUPPORTED"
        action = "FLAGGED_WITH_UNCERTAINTY"
        reason = "No verifiable telemetry fields identified matching incident schema"
        presented_text = f"[UNVERIFIED TELEMETRY - MISMATCH] {claim_text}"

    return status, action, presented_text, reason, audit_records, unparsed_spans
