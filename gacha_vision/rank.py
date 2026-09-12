"""Pure ranking logic: cards in, a Decision out. No image code here.

This module is deliberately free of OpenCV/OCR imports so the policy can be
unit-tested in isolation and reasoned about on its own.
"""

from __future__ import annotations

import math

from .config import Policy, normalise
from .models import Action, Card, Decision, FrameTier, Score


def _print_score(card: Card, policy: Policy) -> tuple[float, str]:
    if card.no_number:
        return policy.score_no_number, "no print number (E) -> weak"
    if card.print_no is None:
        return policy.score_unreadable, "print number unreadable -> neutral, review"
    if not card.print_trusted:
        # A shaky read is not a low print. Scoring it as if it were is how a
        # misread #1655 becomes a #16 worth spending an extra pick on.
        return policy.score_unreadable, f"print #{card.print_no} read with low confidence -> review"
    # Lower print number is rarer. log makes #1 vs #5 matter far more than
    # #400 vs #500, which matches how these games actually value prints.
    raw = policy.print_base - policy.print_decay * math.log10(max(card.print_no, 1))
    score = max(policy.print_floor, min(policy.print_base, raw))
    return score, f"print #{card.print_no}"


def _fame_score(card: Card, watchlist: dict[str, float], policy: Policy) -> tuple[float, str]:
    if not watchlist:
        return policy.fame_default, "no watchlist"
    ch = normalise(card.character)
    sr = normalise(card.series)
    if ch and ch in watchlist:
        return watchlist[ch], f"watchlist character: {card.character}"
    if sr and sr in watchlist:
        return watchlist[sr], f"watchlist series: {card.series}"
    return policy.fame_default, "not on watchlist"


def score_card(card: Card, policy: Policy, watchlist: dict[str, float]) -> Score:
    ps, pr = _print_score(card, policy)
    fs = policy.frame_score(card.frame)
    frame_note = f"frame {card.frame.value} ({fs:.0f})"
    if card.no_number and not policy.frame_lifts_unnumbered:
        capped = policy.frame_score(FrameTier.E)
        if fs > capped:
            frame_note = f"frame {card.frame.value} capped to {capped:.0f} (no print number)"
            fs = capped
    fa, far = _fame_score(card, watchlist, policy)

    total = policy.w_print * ps + policy.w_frame * fs + policy.w_fame * fa
    reasons = [pr, frame_note, far]
    return Score(
        slot=card.slot,
        total=round(total, 2),
        print_score=round(ps, 2),
        frame_score=round(fs, 2),
        fame_score=round(fa, 2),
        reasons=reasons,
    )


def _trusted_print(card: Card) -> int | None:
    """The print number, but only when it is solid enough to act on."""
    if card.no_number or card.print_no is None or not card.print_trusted:
        return None
    return card.print_no


def _is_must_claim(card: Card, score: Score, policy: Policy) -> str | None:
    tp = _trusted_print(card)
    if tp is not None and tp <= policy.must_claim_print:
        return f"print #{tp} <= must-claim {policy.must_claim_print}"
    if score.fame_score >= policy.must_claim_fame:
        return f"fame {score.fame_score:.0f} >= must-claim {policy.must_claim_fame}"
    if policy.always_claim_unknown_frame and card.frame is FrameTier.OTHER:
        # Never yet catalogued, so it could be the rare frame. Worth a pick
        # and a human glance either way.
        return "frame is neither E nor NORMAL -- uncatalogued, claim it"
    # A holo frame used to force a claim here. Real spawns killed that rule:
    # the ornate rainbow frames sit on the *worst* cards, so it fired on E
    # cards and picked the junk half of the spawn every time.
    return None


def decide(cards: list[Card], policy: Policy, watchlist: dict[str, float] | None = None) -> Decision:
    """Rank a spawn's cards and choose an action.

    Order of reasoning:
      1. Score every card.
      2. "Take both" if enough cards clear the low-print bar (the stated
         rule) or are individually excellent.
      3. Otherwise claim the single best card if it beats the floor.
      4. Otherwise skip.
    """
    watchlist = watchlist or {}
    if not cards:
        return Decision(Action.SKIP, [], [], ["no cards detected"])

    scores = [score_card(c, policy, watchlist) for c in cards]
    if len(cards) == 1 and policy.always_claim_lone_card and not cards[0].no_number:
        c = cards[0]
        return Decision(Action.CLAIM, [c.slot], scores,
                        [f"slot {c.slot}: {c.label()}",
                         "claimed: lone card in the spawn -- nothing to weigh it against"])
    by_slot = {c.slot: c for c in cards}
    by_slot_score = {s.slot: s for s in scores}
    ranked = sorted(scores, key=lambda s: s.total, reverse=True)

    # A card earns a pick on its own merit if it is a must-claim (a top
    # print or a watchlist favourite), or has a genuinely low
    # print, or simply scores very highly. E-cards never qualify on the
    # print rule -- "no number" is not a low number.
    #
    # This is evaluated for EVERY card, not just the highest scorer: a
    # must-claim card can be outscored by a card that is merely mediocre
    # (a famous character on an E print loses to a plain mid print), and
    # skipping the spawn in that case would be exactly backwards.
    eligible: list[tuple[Score, str]] = []
    for s in ranked:
        c = by_slot[s.slot]
        why = _is_must_claim(c, s, policy)
        tp = _trusted_print(c)
        if not why and tp is not None and tp <= policy.take_both_max_print:
            why = f"print #{tp} <= {policy.take_both_max_print}"
        if not why and s.total >= policy.take_both_min_score:
            why = f"score {s.total:.1f} >= {policy.take_both_min_score}"
        if why:
            eligible.append((s, why))

    if len(eligible) >= 2 and policy.max_claims >= 2:
        chosen = eligible[:policy.max_claims]          # already score-ordered
        reasons = [f"{len(eligible)} cards each merit a pick; spending the extra pick"]
        reasons += [f"slot {s.slot}: {by_slot[s.slot].label()} -> {s.total:.1f} ({why})"
                    for s, why in chosen]
        return Decision(Action.CLAIM_BOTH, [s.slot for s, _ in chosen], scores, reasons)

    if eligible:
        # A one-claim limit still selects from the cards that earned a pick.
        s, why = eligible[0]
        reasons = [f"slot {s.slot}: {by_slot[s.slot].label()} -> {s.total:.1f}", f"claimed: {why}"]
        if len(ranked) > 1:
            other = next(o for o in ranked if o.slot != s.slot)
            reasons.append(
                f"passed on slot {other.slot} {by_slot[other.slot].label()} ({other.total:.1f})"
            )
        return Decision(Action.CLAIM, [s.slot], scores, reasons)

    # --- nothing stands out; take the best numbered card anyway ------------
    #
    # Ordered the way the owner ranks them: any readable print beats an
    # unreadable one (a number we read is known, one we did not could be
    # anything), and both beat an E, which has no number by definition.
    if policy.claim_best_numbered:
        numbered = [c for c in cards if not c.no_number]
        if numbered:
            # _trusted_print, not print_no: a shaky read is not a low print.
            # A lone digit is usually the hook icon beside the badge, so
            # ranking on the raw value would let a misread "#1" beat a real
            # #9999 -- the exact failure the trust gate exists to stop.
            pick = min(numbered, key=lambda c: (_trusted_print(c) is None,
                                                _trusted_print(c) or 0,
                                                c.slot))
            s = by_slot_score[pick.slot]
            reasons = [f"slot {pick.slot}: {pick.label()} -> {s.total:.1f}",
                       "claimed: best numbered card in the spawn -- "
                       "a number always beats an E"]
            others = [c for c in cards if c.slot != pick.slot]
            if others:
                reasons.append("passed on " + ", ".join(
                    f"slot {c.slot} {c.label()}" for c in others))
            return Decision(Action.CLAIM, [pick.slot], scores, reasons)
        return Decision(Action.SKIP, [], scores,
                        ["every card is an E -- nothing to choose between them"])

    best = ranked[0]
    best_card = by_slot[best.slot]
    if best.total >= policy.min_claim_score:
        reasons = [
            f"best: slot {best.slot} {best_card.label()} -> {best.total:.1f}",
            f"score {best.total:.1f} >= claim floor {policy.min_claim_score}",
        ]
        if len(ranked) > 1:
            runner = ranked[1]
            reasons.append(
                f"passed on slot {runner.slot} {by_slot[runner.slot].label()} ({runner.total:.1f})"
            )
        return Decision(Action.CLAIM, [best.slot], scores, reasons)

    # --- skip ---
    reasons = [
        f"best card slot {best.slot} {best_card.label()} scored {best.total:.1f} "
        f"< claim floor {policy.min_claim_score}"
    ]
    return Decision(Action.SKIP, [], scores, reasons)
