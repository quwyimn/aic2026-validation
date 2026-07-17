"""
confusion.py — turning matched pairs into numbers.

The matrix carries two extra bands beyond the usual square:

    MISS  — ground truth existed, the model returned nothing there
    GHOST — the model returned something, no ground truth there

Without them, "didn't see the object" and "saw it but called it the wrong
name" collapse into the same cell, and those are different failures with
different fixes.
"""

from collections import Counter, defaultdict


class Confusion:
    """Rows are ground truth, columns are prediction."""

    def __init__(self, class_names, class_ids):
        self.class_names = list(class_names)
        self.class_ids = dict(class_ids)                  # name -> id
        self.name_by_id = {v: k for k, v in class_ids.items()}
        self.cells = defaultdict(int)                     # (gt_id, pred_id) -> n
        self.miss = Counter()                             # gt_id -> n
        self.ghost = Counter()                            # pred_id -> n

    # -- accumulate --------------------------------------------------------
    def add_pair(self, gt_id, pred_id):
        self.cells[(gt_id, pred_id)] += 1

    def add_miss(self, gt_id):
        self.miss[gt_id] += 1

    def add_ghost(self, pred_id):
        self.ghost[pred_id] += 1

    def merge(self, other):
        for k, v in other.cells.items():
            self.cells[k] += v
        self.miss.update(other.miss)
        self.ghost.update(other.ghost)

    # -- derive ------------------------------------------------------------
    def support(self, cid):
        """Total GT instances of this class: matched (however labelled) plus
        missed."""
        return (sum(v for (g, _), v in self.cells.items() if g == cid)
                + self.miss[cid])

    def correct(self, cid):
        return self.cells.get((cid, cid), 0)

    def predicted(self, cid):
        """Everything the model called this class, ghosts included — a ghost
        is a false positive for whatever class it claimed to be."""
        return (sum(v for (_, p), v in self.cells.items() if p == cid)
                + self.ghost[cid])

    def precision(self, cid):
        n = self.predicted(cid)
        return self.correct(cid) / n if n else float("nan")

    def recall(self, cid):
        n = self.support(cid)
        return self.correct(cid) / n if n else float("nan")

    def f1(self, cid):
        p, r = self.precision(cid), self.recall(cid)
        if p != p or r != r or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    def macro(self, fn):
        """Average over classes that actually have ground truth present.

        Macro, not micro: Person outnumbers the rarest class roughly 20 to 1,
        so a micro-average can read as excellent while every robot class is
        broken. Classes with no GT in this block are excluded rather than
        counted as zero — absent is not the same as failed.

        The exclusion has a cost, and absent_class_predictions() exists to pay
        it: predictions naming a class that has no GT here are false positives
        that this average cannot see. They are reported separately instead of
        being folded in, because folding four absent classes in at zero would
        drag a macro over seven classes down to 0.43 over a handful of stray
        boxes — misleading in the opposite direction.
        """
        vals = [fn(cid) for cid in self.class_ids.values() if self.support(cid) > 0]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def absent_class_predictions(self):
        """
        Predictions naming a class with no ground truth anywhere in this block.

        Every one is a false positive by construction: the class does not
        occur here, so nothing the model calls by that name can be right. This
        matters most on W020, where only 3 of the 7 classes are present — a
        misclassification has a 4-in-6 chance of landing on a class that
        cannot be checked by precision or recall, and would otherwise vanish
        from the report entirely.

        Returns {class_name: count}.
        """
        out = {}
        for name, cid in self.class_ids.items():
            if self.support(cid) == 0:
                n = self.predicted(cid)
                if n:
                    out[name] = n
        return out

    def total_gt(self):
        return sum(self.support(c) for c in self.class_ids.values())

    def total_miss(self):
        return sum(self.miss.values())

    def total_ghost(self):
        return sum(self.ghost.values())

    def off_diagonal(self):
        """Confusions only: matched pairs where the labels disagree."""
        return {k: v for k, v in self.cells.items() if k[0] != k[1] and v > 0}

    # -- systematic mapping detection --------------------------------------
    def mapping_signature(self):
        """
        Detect a miswired class table.

        A model that confuses objects errs here and there. A miswired lookup
        errs on every single instance of a class, always in the same
        direction. So: for each class, if essentially everything landed in one
        wrong column, that's not perception failing — that's a constant.

        Returns {gt_name: (pred_name, fraction)} for classes where one wrong
        column swallowed 90%+ of the matched instances.
        """
        out = {}
        for name, cid in self.class_ids.items():
            matched = {p: v for (g, p), v in self.cells.items() if g == cid and v > 0}
            total = sum(matched.values())
            if total == 0:
                continue
            top_pred, top_n = max(matched.items(), key=lambda kv: kv[1])
            if top_pred != cid and top_n / total >= 0.90:
                out[name] = (self.name_by_id.get(top_pred, str(top_pred)),
                             top_n / total)
        return out

    # -- render ------------------------------------------------------------
    def to_dict(self):
        return {
            "cells": {f"{self.name_by_id[g]}->{self.name_by_id[p]}": v
                      for (g, p), v in sorted(self.cells.items()) if v},
            "miss": {self.name_by_id[c]: n for c, n in sorted(self.miss.items())},
            "ghost": {self.name_by_id[c]: n for c, n in sorted(self.ghost.items())},
            "per_class": {
                name: {
                    "support": self.support(cid),
                    "correct": self.correct(cid),
                    "precision": self.precision(cid),
                    "recall": self.recall(cid),
                    "f1": self.f1(cid),
                    "miss": self.miss[cid],
                    "ghost": self.ghost[cid],
                }
                for name, cid in self.class_ids.items()
            },
            "macro": {
                "precision": self.macro(self.precision),
                "recall": self.macro(self.recall),
                "f1": self.macro(self.f1),
            },
            "totals": {
                "gt": self.total_gt(),
                "miss": self.total_miss(),
                "ghost": self.total_ghost(),
                "off_diagonal": sum(self.off_diagonal().values()),
                "absent_class_predictions": sum(self.absent_class_predictions().values()),
            },
            "absent_class_predictions": self.absent_class_predictions(),
            "mapping_signature": {
                k: {"predicted_as": v[0], "fraction": v[1]}
                for k, v in self.mapping_signature().items()
            },
        }

    def render(self, title="CONFUSION MATRIX", thin_threshold=10):
        names = [n for n in self.class_names]
        w = 13
        lines = [f"\n{title}", "rows = ground truth, columns = prediction"]

        head = f"{'':<14}" + "".join(f"{n[:11]:>{w}}" for n in names)
        head += f"{'MISS':>{w}}"
        lines.append("-" * len(head))
        lines.append(head)
        lines.append("-" * len(head))

        for gname in names:
            gid = self.class_ids[gname]
            row = f"{gname[:13]:<14}"
            for pname in names:
                pid = self.class_ids[pname]
                v = self.cells.get((gid, pid), 0)
                if v == 0:
                    row += f"{'.':>{w}}"
                elif gid == pid:
                    row += f"{v:>{w}}"
                else:
                    row += f"{('[' + str(v) + ']'):>{w}}"
            row += f"{(self.miss[gid] or '.'):>{w}}"
            lines.append(row)

        ghost_row = f"{'GHOST':<14}"
        for pname in names:
            v = self.ghost[self.class_ids[pname]]
            ghost_row += f"{(v or '.'):>{w}}"
        lines.append("-" * len(head))
        lines.append(ghost_row)
        lines.append("-" * len(head))
        lines.append("[n] = misclassification — matched to the object, "
                     "called it something else")

        # per-class table
        lines.append(f"\n{'Class':<16}{'Support':>10}{'Precision':>11}"
                     f"{'Recall':>9}{'F1':>8}{'Miss':>9}{'Ghost':>8}")
        lines.append("-" * 71)
        for name in names:
            cid = self.class_ids[name]
            sup = self.support(cid)
            if sup == 0:
                n = self.predicted(cid)
                if n:
                    lines.append(f"{name:<16}{'—':>10}{'0.000':>11}{'—':>9}{'—':>8}"
                                 f"{'—':>9}{self.ghost[cid]:>8}"
                                 f"  << {n} predictions, no GT — all false")
                else:
                    lines.append(f"{name:<16}{'—':>10}{'no GT in this block':>39}")
                continue
            flag = "  (thin)" if sup < thin_threshold else ""
            lines.append(
                f"{name:<16}{sup:>10}{self.precision(cid):>11.4f}"
                f"{self.recall(cid):>9.4f}{self.f1(cid):>8.4f}"
                f"{self.miss[cid]:>9}{self.ghost[cid]:>8}{flag}")
        lines.append("-" * 71)
        lines.append(f"{'MACRO AVG':<16}{self.total_gt():>10}"
                     f"{self.macro(self.precision):>11.4f}"
                     f"{self.macro(self.recall):>9.4f}"
                     f"{self.macro(self.f1):>8.4f}"
                     f"{self.total_miss():>9}{self.total_ghost():>8}")
        absent = self.absent_class_predictions()
        if absent:
            lines.append("macro covers only classes with GT present — the block "
                         "below covers the rest")
            total = sum(absent.values())
            lines.append(f"\n!! {total} predictions name a class with no ground "
                         f"truth in this block.")
            lines.append("   Every one is a false positive, and none of them "
                         "reach the macro average above.")
            for name, n in sorted(absent.items(), key=lambda kv: -kv[1]):
                lines.append(f"     {name:<16}{n:>10}")
        return "\n".join(lines)