"""Smart Tagging as extra evidence for the decoder, by fixed rules about what each code means.

Smart Tagging (Metrica's automatic tactical annotation, see DATA.md) says, for stretches of the
match, which team is in which phase. Some codes imply who has the ball; most do not. The rules:

- possession phases (BUILD UP, PROGRESSION, ATTACKING TRANSITION, CREATING CHANCES) with a team:
  that team has the ball whenever the ball is alive, so the *other* team is penalised;
- set pieces with a team (corner, goal kick, free kick): the restart belongs to that team. The
  restart can come seconds after the tag starts or ends, so the window runs to the end of the tag
  plus RESTART_TAIL, and again only the other team is penalised;
- GOALS: the ball is dead (celebration, kickoff), so both teams are penalised;
- everything else is ignored: the DEFENDING codes mirror a possession phase exactly (counting
  them would double the evidence), location codes (BALL IN THE BOX, ...) carry no possession, and
  throw-ins and kickoffs are tagged without a team.

Team evidence only ever penalises the other team and leaves D alone: "Barcelona has it" argues
against Atletic, not against a dead ball (a CREATING CHANCES stretch is half dead ball).

How strong each rule is comes from the match itself, without labels: on the frames a rule covers,
how often does the classifier (before decoding) favour the team the rule names? An agreement
`acc` becomes a log-odds `log(acc / (1 - acc))`, the weight a source of that accuracy deserves.
`acc` is capped at ACC_MAX because the tags are made from the same tracking as our features, so
the two tend to be wrong together and agreement overstates the tags' accuracy.

Set pieces and goals state one fact about a short moment (who restarts, the ball is dead), so
their weight applies to every frame of their window. A phase is different: it is one judgement
about a stretch that can hold several possessions (DATA.md: "phases, not possessions"). Applying
its weight to every frame turns it into a near-hard rule over hundreds of frames and wipes out
real short possessions of the other team inside it (seen on the first run: a Barcelona phase
erased a 3 s Athletic possession). So a phase counts as a single observation: its log-odds is
spread over its frames (weight / length per frame). That only decides between readings the
classifier finds close to equally likely; it cannot outvote a confident short possession.
Phases are off unless asked for (phases=True).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, asset_paths
from solution.common import FPS, A, D, H
from starter.load_smart_tagging import load as load_events

POSSESSION_CODES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES")
SET_PIECE_CODE = "SET PIECES"
RESTART_TYPES = ("corner kick", "goal kick", "free kick")
DEAD_CODES = ("GOALS",)
RESTART_TAIL = 3 * FPS  # frames after a set-piece tag ends in which the restart may still fall
ACC_MAX = 0.9  # cap on estimated agreement: at most log(9) ~ 2.2 per frame
MIN_FRAMES = 10 * FPS  # a rule seen on fewer live frames than this gets no weight


def home_team_name(game=GAME, data_dir=None):
    """Name of the home team: the first team in the tracking metadata (Barcelona, see DATA.md)."""
    root = ET.parse(asset_paths(game, data_dir)["atd_meta"]).getroot()
    team = next(root.iter("Team"))
    return team.findtext("Name").strip()


def tag_rules(events, home, phases=True):
    """[(rule, letter, start, stop)] for the tags that imply possession or a dead ball.

    letter is the team the rule names (H / A), or D for a dead-ball rule. Frames are half-open.
    phases=False leaves out the possession phases and keeps only set pieces and goals.
    """
    home = home.strip().lower()
    rules = []
    for event in events:
        code = event["code"].strip().upper()
        team = (event["tags"].get("Team") or "").strip().lower()
        kind = (event["tags"].get("Type") or "").strip().lower()
        start, stop = event["start_frame"], event["end_frame"]
        letter = None if team in ("", "n/a") else (H if team == home else A)
        if code in DEAD_CODES:
            rules.append((code, D, start, stop))
        elif code in POSSESSION_CODES and letter is not None and phases:
            rules.append((code, letter, start, stop))
        elif code == SET_PIECE_CODE and kind in RESTART_TYPES and letter is not None:
            rules.append((SET_PIECE_CODE, letter, start, stop + RESTART_TAIL))
    return rules


def _clip(start, stop, n_frames):
    return max(0, int(start)), min(n_frames, int(stop))


def agreement(rules, probs):
    """{rule: (acc, n_frames)}: how often the classifier agrees with each rule, no labels used.

    Team rules: share of the covered frames the classifier calls alive (p(D) < 0.5) on which it
    favours the named team over the other. Dead rules: share of covered frames with p(D) >= 0.5.
    """
    n_frames = len(probs)
    alive = probs[:, D] < 0.5
    hits, counts = {}, {}
    for rule, letter, start, stop in rules:
        lo, hi = _clip(start, stop, n_frames)
        if lo >= hi:
            continue
        p = probs[lo:hi]
        if letter == D:
            hit, count = (~alive[lo:hi]).sum(), hi - lo
        else:
            other = A if letter == H else H
            live = alive[lo:hi]
            hit, count = (live & (p[:, letter] > p[:, other])).sum(), live.sum()
        hits[rule] = hits.get(rule, 0) + int(hit)
        counts[rule] = counts.get(rule, 0) + int(count)
    return {
        rule: (hits[rule] / counts[rule] if counts[rule] else 0.5, counts[rule]) for rule in hits
    }


def weights(agree):
    """{rule: per-frame log-odds weight} from {rule: (acc, n)}; 0 for rules no better than chance."""
    out = {}
    for rule, (acc, count) in agree.items():
        acc = min(acc, ACC_MAX)
        out[rule] = float(np.log(acc / (1 - acc))) if acc > 0.5 and count >= MIN_FRAMES else 0.0
    return out


def build_prior(rules, rule_weights, n_frames):
    """[T, 3] log-score offsets to add to the classifier's log-probabilities before decoding.

    Possession phases count as one observation each, so their weight is spread over their frames;
    set pieces and goals apply their weight on every frame. Per frame and penalised label the
    strongest applicable penalty counts, not the sum, so overlapping tags do not stack.
    """
    penalty = np.zeros((n_frames, 3))
    for rule, letter, start, stop in rules:
        weight = rule_weights.get(rule, 0.0)
        lo, hi = _clip(start, stop, n_frames)
        if weight <= 0 or lo >= hi:
            continue
        if rule in POSSESSION_CODES:
            weight /= stop - start
        targets = (H, A) if letter == D else ((A,) if letter == H else (H,))
        for target in targets:
            np.maximum(penalty[lo:hi, target], weight, out=penalty[lo:hi, target])
    return -penalty


def match_prior(probs, game=GAME, data_dir=None, phases=False, verbose=True):
    """The prior for the competition match, or None if it has no Smart Tagging file."""
    if not os.path.isfile(asset_paths(game, data_dir).get("smart_tagging", "")):
        return None
    home = home_team_name(game, data_dir)
    events = load_events(game, data_dir)
    # The home-team mapping is checked on all team rules, also when phases are not used.
    check = agreement(tag_rules(events, home, phases=True), probs)
    team_rules = [r for r in check if r not in DEAD_CODES]
    hit = sum(check[r][0] * check[r][1] for r in team_rules)
    total = sum(check[r][1] for r in team_rules)
    if total and hit / total < 0.5:
        raise SystemExit(
            "Smart Tagging teams disagree with the classifier on most frames ({:.0%}): the home "
            "team mapping is probably flipped.".format(hit / total)
        )
    rules = tag_rules(events, home, phases=phases)
    agree = agreement(rules, probs)
    rule_weights = weights(agree)
    if verbose:
        print("Smart Tagging rules (agreement with the classifier -> weight)")
        for rule in sorted(agree):
            acc, count = agree[rule]
            unit = "per tag, spread" if rule in POSSESSION_CODES else "per frame"
            print(
                "  {:22s} acc {:.2f} on {:6d} frames -> w {:.2f} {}".format(
                    rule, acc, count, rule_weights[rule], unit
                )
            )
    return build_prior(rules, rule_weights, len(probs))
