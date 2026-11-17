#!/usr/bin/env python3
"""
Allowlisted Grammar-Based Factual Assertion Parser and Strict Validator
for PAADU/CHARM Telemetry and XAI Explanations.

Replaces permissive regex matching with a strict allowlisted grammar of complete,
supported telemetry assertions. Parses signed floating point numbers, scientific
notation, multi-word flow features, and compound sentences.

Fail-Closed Invariants:
1. Every clause must match an allowlisted assertion schema with known fields,
   operators, values, signs, and optional units.
2. Any unmatched leftover text, causal inference, identity assertion, or unknown
   feature fails closed as UNSUPPORTED (FLAGGED_WITH_UNCERTAINTY).
3. Any numeric or categorical discrepancy against authoritative incident ground truth
   fails closed as CONTRADICTED (CONTRADICTION_RETRACTED).
4. A claim is ONLY labeled SUPPORTED [VERIFIED TELEMETRY] if every atomic assertion
   is independently supported and zero unrecognized or contradictory text remains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

NUM_PATTERN = r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'

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

DELIM_RE = re.compile(
    r';|\s+-\s+|\s+(?:and|but|because|while|whereas|as\s+well\s+as|with)\s+|,\s*(?:proving|due\s+to|suggesting|caused\s+by|attributed\s+to|indicating)\s*|,\s+',
    re.IGNORECASE
)


def split_clauses_protected(text: str) -> List[str]:
    """Splits compound claims into atomic clauses while preserving commas inside parentheses."""
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
    raw = DELIM_RE.split(protected)
    return [cl.replace('\x00', ',').strip().rstrip('.') for cl in raw if cl.strip()]


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
            'raw_clause': self.raw_clause
        }


def parse_and_validate_telemetry(
    claim_text: str,
    incident: Any
) -> Tuple[str, str, str, str, List[AtomicAssertionAudit]]:
    """
    Parses claim text using strict allowlisted grammar and validates every atomic
    factual assertion against authoritative incident ground truth.

    Returns:
        status: 'SUPPORTED', 'CONTRADICTED', or 'UNSUPPORTED'
        action: 'RETAINED_VERIFIED', 'CONTRADICTION_RETRACTED', or 'FLAGGED_WITH_UNCERTAINTY'
        presented_text: Safe narrative with verification badges or retraction notes
        reason: Diagnostic reason string detailing outcomes
        audit_records: List of AtomicAssertionAudit objects for structured assertion-level reporting
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
            raw_clause=text
        )
        return "CONTRADICTED", "CONTRADICTION_RETRACTED", text, "Pre-existing retraction marker", [rec]

    contradictions: List[str] = []
    unsupported_assertions: List[str] = []
    supported_assertions: List[str] = []
    audit_records: List[AtomicAssertionAudit] = []

    top_feats = {f.feature.lower(): f for f in incident.top_xai_features}

    # STEP 1: Global Contradiction Scan (Find ANY Contradictory Telemetry Assertion)
    # 1.1 Global Score Scan
    for m in RE_SCORE_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.anomaly_score
        if abs(val - exp) > 0.05:
            msg = f"asserted score {val:.3f} contradicts actual {exp:.3f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('score', 'anomaly_score', 'of', val, exp, False, True, msg, m.group(0)))

    # 1.2 Global Threshold Scan
    for m in RE_THRESH_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.decision_threshold
        if abs(val - exp) > 0.05:
            msg = f"asserted threshold {val:.3f} contradicts actual {exp:.3f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('threshold', 'decision_threshold', 'of', val, exp, False, True, msg, m.group(0)))

    # 1.3 Global Reconstruction Error Scan
    for m in RE_RECON_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.reconstruction_error
        if abs(val - exp) > 0.05:
            msg = f"asserted reconstruction error {val:.4f} contradicts actual {exp:.4f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('recon_error', 'reconstruction_error', 'of', val, exp, False, True, msg, m.group(0)))

    # 1.4 Global Mahalanobis Distance Scan
    for m in RE_MAHAL_GLOBAL.finditer(text):
        val = float(m.group(1))
        exp = incident.mahalanobis_distance
        if abs(val - exp) > 0.5:
            msg = f"asserted Mahalanobis distance {val:.4f} contradicts actual {exp:.4f}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('mahalanobis', 'mahalanobis_distance', 'of', val, exp, False, True, msg, m.group(0)))

    # 1.5 Global Status Scan
    for m in RE_STATUS_GLOBAL.finditer(text):
        st = m.group(1).upper()
        exp_st = 'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'
        if st != exp_st:
            msg = f"asserted status {st} contradicts actual {exp_st}"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('status', 'status', '=', st, exp_st, False, True, msg, m.group(0)))

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
            raw_clause="non-existent flow feature"
        ))

    # 1.7 Global Feature Mentions Scan (e.g. 'feature FooBar')
    for m in RE_FEAT_MENTION.finditer(text):
        fname = m.group(1).lower()
        if fname not in top_feats and fname not in ('leave-one-feature-out', 'occlusion'):
            msg = f"asserted feature '{m.group(1)}' not in top XAI features"
            contradictions.append(msg)
            audit_records.append(AtomicAssertionAudit('feature_name', m.group(1), 'in', m.group(1), list(top_feats.keys()), False, True, msg, m.group(0)))

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
                    audit_records.append(AtomicAssertionAudit('feature_contribution', f"{fn_clean}.contrib", ':', val, exp_c, False, True, msg, m.group(0)))
            else:
                msg = f"feature '{fn_clean}' not in top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('feature_name', fn_clean, 'in', fn_clean, list(top_feats.keys()), False, True, msg, m.group(0)))
        else:
            if not any(abs(val - f.hybrid_contribution) <= 0.05 for f in incident.top_xai_features):
                msg = f"asserted contribution {val:+.2f} contradicts top feature attributions"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('contribution', 'any_feature', ':', val, [f.hybrid_contribution for f in incident.top_xai_features], False, True, msg, m.group(0)))

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
                    audit_records.append(AtomicAssertionAudit('benign_mean', f"{fn}.ben", ':', val, exp_b, False, True, msg, m.group(0)))
            else:
                msg = f"feature '{fn}' not in top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('feature_name', fn, 'in', fn, list(top_feats.keys()), False, True, msg, m.group(0)))
        else:
            if not any(abs(val - f.benign_mean) <= 0.1 for f in incident.top_xai_features):
                msg = f"asserted benign mean {val:.2f} contradicts all top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('benign_mean', 'any_feature', ':', val, [f.benign_mean for f in incident.top_xai_features], False, True, msg, m.group(0)))

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
                    audit_records.append(AtomicAssertionAudit('observed_mean', f"{fn}.obs", ':', val, exp_o, False, True, msg, m.group(0)))
            else:
                msg = f"feature '{fn}' not in top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('feature_name', fn, 'in', fn, list(top_feats.keys()), False, True, msg, m.group(0)))
        else:
            if not any(abs(val - f.observed_mean) <= 0.1 for f in incident.top_xai_features):
                msg = f"asserted observed mean {val:.2f} contradicts all top features"
                contradictions.append(msg)
                audit_records.append(AtomicAssertionAudit('observed_mean', 'any_feature', ':', val, [f.observed_mean for f in incident.top_xai_features], False, True, msg, m.group(0)))

    # 1.11 Global ungrounded IP check
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
            raw_clause=m_ip.group(0)
        ))

    # STEP 2: Tier 1 Canonical Full Sentence Patterns
    m_dec = FULL_DECISION_RE.match(text)
    m_trig = FULL_TRIGGER_RE.match(text)
    is_xai_preamble = bool(FULL_XAI_PREAMBLE_RE.match(text))

    if m_dec and not contradictions:
        blk_ref, status_str, score_str, thresh_str = m_dec.groups()
        supported_assertions.append(f"status={status_str}")
        supported_assertions.append(f"anomaly_score={float(score_str):.3f}")
        supported_assertions.append(f"threshold={float(thresh_str):.3f}")

        exp_st = 'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'
        audit_records.append(AtomicAssertionAudit(
            assertion_type='status',
            field_name='status',
            operator='=',
            asserted_value=status_str.upper(),
            expected_value=exp_st,
            is_supported=True,
            is_contradiction=False,
            raw_clause=f"evaluated as {status_str}"
        ))
        audit_records.append(AtomicAssertionAudit(
            assertion_type='score',
            field_name='anomaly_score',
            operator='=',
            asserted_value=float(score_str),
            expected_value=incident.anomaly_score,
            is_supported=True,
            is_contradiction=False,
            raw_clause=f"hybrid anomaly score of {score_str}"
        ))
        audit_records.append(AtomicAssertionAudit(
            assertion_type='threshold',
            field_name='decision_threshold',
            operator='=',
            asserted_value=float(thresh_str),
            expected_value=incident.decision_threshold,
            is_supported=True,
            is_contradiction=False,
            raw_clause=f"baseline threshold of {thresh_str}"
        ))

    elif m_trig and not contradictions:
        pol_str, recon_str, mahal_str = m_trig.groups()
        pols = [p.strip() for p in pol_str.split(',') if p.strip()]
        for p in pols:
            if p not in incident.trigger_reasons:
                contradictions.append(f"asserted trigger policy '{p}' contradicts actual triggers")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='policy',
                    field_name='trigger_reasons',
                    operator='in',
                    asserted_value=p,
                    expected_value=incident.trigger_reasons,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=f"asserted trigger policy '{p}' contradicts actual triggers",
                    raw_clause=p
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
                    raw_clause=p
                ))
        supported_assertions.append(f"reconstruction_error={float(recon_str):.4f}")
        audit_records.append(AtomicAssertionAudit(
            assertion_type='recon_error',
            field_name='reconstruction_error',
            operator='=',
            asserted_value=float(recon_str),
            expected_value=incident.reconstruction_error,
            is_supported=True,
            is_contradiction=False,
            raw_clause=f"reconstruction error: {recon_str}"
        ))
        supported_assertions.append(f"mahalanobis_distance={float(mahal_str):.4f}")
        audit_records.append(AtomicAssertionAudit(
            assertion_type='mahalanobis',
            field_name='mahalanobis_distance',
            operator='=',
            asserted_value=float(mahal_str),
            expected_value=incident.mahalanobis_distance,
            is_supported=True,
            is_contradiction=False,
            raw_clause=f"latent Mahalanobis distance: {mahal_str}"
        ))

    elif is_xai_preamble and not contradictions:
        rest = FULL_XAI_PREAMBLE_RE.sub('', text).rstrip('.')
        items = [s.strip() for s in rest.split(';') if s.strip()]
        for it in items:
            m_it = FEATURE_ITEM_RE.fullmatch(it)
            if not m_it:
                unsupported_assertions.append(f"unrecognized feature proposition: '{it}'")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='unknown_syntax',
                    field_name='unknown',
                    operator='unknown',
                    asserted_value=it,
                    expected_value='Allowlisted schema',
                    is_supported=False,
                    is_contradiction=False,
                    error_message=f"unrecognized feature proposition: '{it}'",
                    raw_clause=it
                ))
                continue

            fname, contrib_str, obs_str, ben_str = m_it.groups()
            fname_clean = fname.strip()
            fname_lower = fname_clean.lower()
            if fname_lower not in top_feats:
                contradictions.append(f"feature '{fname_clean}' not in top XAI features")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='feature_name',
                    field_name=fname_clean,
                    operator='in',
                    asserted_value=fname_clean,
                    expected_value=list(top_feats.keys()),
                    is_supported=False,
                    is_contradiction=True,
                    error_message=f"feature '{fname_clean}' not in top XAI features",
                    raw_clause=it
                ))
                continue

            act_f = top_feats[fname_lower]
            contrib_val = float(contrib_str)
            obs_val = float(obs_str)
            ben_val = float(ben_str)

            if abs(contrib_val - act_f.hybrid_contribution) > 0.05:
                contradictions.append(f"{fname_clean} contribution {contrib_val:+.2f} contradicts actual {act_f.hybrid_contribution:+.2f}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='feature_contrib',
                    field_name=f"{act_f.feature}.hybrid_contribution",
                    operator='=',
                    asserted_value=contrib_val,
                    expected_value=act_f.hybrid_contribution,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=f"{fname_clean} contribution {contrib_val:+.2f} contradicts actual {act_f.hybrid_contribution:+.2f}",
                    raw_clause=f"{fname_clean} contribution: {contrib_str}"
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
                    raw_clause=f"{fname_clean} contribution: {contrib_str}"
                ))

            if abs(obs_val - act_f.observed_mean) > 0.1:
                contradictions.append(f"{fname_clean} observed mean {obs_val:.2f} contradicts actual {act_f.observed_mean:.2f}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='obs_mean',
                    field_name=f"{act_f.feature}.observed_mean",
                    operator='=',
                    asserted_value=obs_val,
                    expected_value=act_f.observed_mean,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=f"{fname_clean} observed mean {obs_val:.2f} contradicts actual {act_f.observed_mean:.2f}",
                    raw_clause=f"{fname_clean} observed mean: {obs_str}"
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
                    raw_clause=f"{fname_clean} observed mean: {obs_str}"
                ))

            if abs(ben_val - act_f.benign_mean) > 0.1:
                contradictions.append(f"{fname_clean} benign mean {ben_val:.2f} contradicts actual {act_f.benign_mean:.2f}")
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='ben_mean',
                    field_name=f"{act_f.feature}.benign_mean",
                    operator='=',
                    asserted_value=ben_val,
                    expected_value=act_f.benign_mean,
                    is_supported=False,
                    is_contradiction=True,
                    error_message=f"{fname_clean} benign mean {ben_val:.2f} contradicts actual {act_f.benign_mean:.2f}",
                    raw_clause=f"{fname_clean} benign mean: {ben_str}"
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
                    raw_clause=f"{fname_clean} benign mean: {ben_str}"
                ))

    elif not m_dec and not m_trig and not is_xai_preamble:
        # STEP 3: Tier 2 Atomic Clause Decomposition Grammar
        clauses = split_clauses_protected(text)
        if not clauses:
            unsupported_assertions.append("empty claim proposition")

        for cl in clauses:
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
                audit_records.append(AtomicAssertionAudit(
                    assertion_type='unrecognized_clause',
                    field_name='unknown',
                    operator='unknown',
                    asserted_value=cl,
                    expected_value='Allowlisted Telemetry Schema',
                    is_supported=False,
                    is_contradiction=False,
                    error_message=f"Leftover/unrecognized clause '{cl}' fails closed",
                    raw_clause=cl
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
                        raw_clause=cl
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
                        raw_clause=cl
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
                        raw_clause=cl
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
                        raw_clause=cl
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
                        raw_clause=cl
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
                    if (abs(c_val - act_f.hybrid_contribution) <= 0.05 and
                        abs(o_val - act_f.observed_mean) <= 0.1 and
                        abs(b_val - act_f.benign_mean) <= 0.1):
                        supported_assertions.append(f"{fname_clean}_tuple=SUPPORTED")
                        audit_records.append(AtomicAssertionAudit(
                            assertion_type='feature_tuple',
                            field_name=f"{act_f.feature}.tuple",
                            operator='=',
                            asserted_value=(c_val, o_val, b_val),
                            expected_value=(act_f.hybrid_contribution, act_f.observed_mean, act_f.benign_mean),
                            is_supported=True,
                            is_contradiction=False,
                            raw_clause=cl
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
                                raw_clause=cl
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
                            raw_clause=cl
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
                                raw_clause=cl
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
                            raw_clause=cl
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
                                raw_clause=cl
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
                            raw_clause=cl
                        ))

            elif matched_schema == 'policies':
                pols = [p.strip() for p in groups[0].split(',') if p.strip()]
                if all(p in incident.trigger_reasons for p in pols):
                    supported_assertions.append(f"policies=[{', '.join(pols)}]")
                    audit_records.append(AtomicAssertionAudit(
                        assertion_type='policy',
                        field_name='trigger_reasons',
                        operator='in',
                        asserted_value=pols,
                        expected_value=incident.trigger_reasons,
                        is_supported=True,
                        is_contradiction=False,
                        raw_clause=cl
                    ))

    # STEP 4: Strict Fail-Closed Verdict Resolution
    if contradictions:
        status = "CONTRADICTED"
        action = "CONTRADICTION_RETRACTED"
        reason = "; ".join(contradictions)
        presented_text = (
            f"[CONTRADICTED TELEMETRY RETRACTED] Telemetry contradiction detected ({reason}). "
            f"Actual PAADU/CHARM telemetry: anomaly_score={incident.anomaly_score:.3f}, "
            f"threshold={incident.decision_threshold:.3f}, status={'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'}."
        )
    elif unsupported_assertions:
        status = "UNSUPPORTED"
        action = "FLAGGED_WITH_UNCERTAINTY"
        reason = "; ".join(unsupported_assertions)
        presented_text = f"[UNVERIFIED TELEMETRY - UNGROUNDED ASSERTION] Compound telemetry claim contains unverified assertions ({reason}): '{claim_text}'"
    elif supported_assertions:
        status = "SUPPORTED"
        action = "RETAINED_VERIFIED"
        reason = f"Verified fields: {', '.join(supported_assertions)}"
        presented_text = f"[VERIFIED TELEMETRY] {claim_text}"
    else:
        status = "UNSUPPORTED"
        action = "FLAGGED_WITH_UNCERTAINTY"
        reason = "No verifiable telemetry fields identified matching incident schema"
        presented_text = f"[UNVERIFIED TELEMETRY - MISMATCH] {claim_text}"

    return status, action, presented_text, reason, audit_records
