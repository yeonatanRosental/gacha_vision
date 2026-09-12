"""A long-lived reader that another process talks to over stdin/stdout.

Spawning a fresh interpreter per spawn costs about 0.8s of startup before any
work happens, which is most of the budget for reacting to a live drop. This
stays up instead: one process, one import, ~35ms per answer.

    python -u -m gacha_vision.worker

Protocol -- newline-delimited JSON in both directions, one request per line,
exactly one reply per request so a caller can pipeline.

**Are you alive**

    ->  {"ping": true}
    <-  {"pong": true}

**What should I claim**

    ->  {"image": "<base64>", "expected": 2}
    <-  {"slots": [2], "cards": [
          {"slot": 1, "printNo": 37, "printTrusted": true, "frame": "normal"},
          {"slot": 2, "printNo": null, "printTrusted": false, "frame": "e"}
        ]}

`expected` is optional; omit it to let segmentation decide the card count.

`slots` is the answer: 1-based, left to right, matching the buttons under the
spawn. Empty means claim nothing.

`cards` is what the reader saw on the way to that answer. It costs nothing --
the cards are read inside `pick` either way and were previously discarded.
Three things about it are worth knowing on the far side:

* `printNo` is null for an `E` card *and* for a numbered card whose badge
  could not be read. `frame` is what separates those two cases.
* `printTrusted` is true only when a non-null `printNo` is trusted by the
  ranker. An untrusted number is a tentative OCR read, kept for diagnostics.
* `frame` is `"normal"`, `"e"`, `"other"` or `"unknown"`. `"other"` means a
  border matching neither known frame, which is claimed on sight whatever its
  print -- so it explains a claim that the number alone would not.

**When something goes wrong**

    <-  {"error": "ValueError: could not decode image from bytes"}

An error is never an empty `slots`: "nothing worth claiming" and "the image
was broken" must not look the same. A bad request is answered, never fatal --
the worker exits only when stdin closes.
"""

from __future__ import annotations

import base64
import json
import sys


def handle(request: dict) -> dict:
    """Answer one request. Pure: a dict in, a dict out."""
    try:
        if request.get("ping"):
            return {"pong": True}

        raw = request.get("image")
        if not raw:
            return {"error": "ValueError: request has no 'image'"}

        # Imported here so a ping costs nothing and the import error, if the
        # package is not installed, surfaces on the first real request rather
        # than at start-up where a caller may not be reading yet.
        from .analyze import analyze_cards, load_image
        from .config import Policy
        from .rank import decide

        image = load_image(base64.b64decode(raw, validate=True))
        cards = analyze_cards(image, request.get("expected"),
                              read_names=False, fast=True)
        slots = list(decide(cards, Policy(), {}).slots) if cards else []
        return {
            "slots": slots,
            "cards": [{"slot": c.slot,
                       "printNo": c.print_no,
                       "printTrusted": (c.print_no is not None
                                        and not c.no_number and c.print_trusted),
                       "frame": c.frame.value} for c in cards],
        }
    except Exception as exc:                      # never die on one bad request
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    # Warm the import before announcing readiness, so a caller that waits for
    # the ready line knows the next request will be fast rather than paying
    # for OpenCV on it.
    from . import pick  # noqa: F401

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
            continue
        print(json.dumps(handle(request)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
