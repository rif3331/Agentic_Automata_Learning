from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

from pyvis.network import Network
from output_paths import get_artifact_dir


EDGE_FONT_BIG = {
    "size": 30,
    "align": "middle",
    "color": "#111111",
}

EDGE_FONT_SMALL = {
    "size": 16,
    "align": "middle",
    "color": "#111111",
}

SELF_LOOP_FONT_BIG = {
    "size": 30,
    "align": "middle",
    "color": "#111111",
    "strokeWidth": 8,
    "strokeColor": "#ffffff",
}

SELF_LOOP_FONT_SMALL = {
    "size": 16,
    "align": "middle",
    "color": "#111111",
    "strokeWidth": 8,
    "strokeColor": "#ffffff",
}


def draw_DFA_html(dfa) -> str:
    out_dir = get_artifact_dir("DFA")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"DFA_{ts}.html"

    states_sorted = sorted([str(s) for s in dfa.states])
    id_of = {s: i for i, s in enumerate(states_sorted)}

    init = str(dfa.initial_state)
    finals = {str(s) for s in dfa.final_states}

    state_label: Dict[str, str] = {
        s: f"state_{i}" for i, s in enumerate(states_sorted)
    }

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.toggle_physics(False)

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          },
          "physics": {
            "enabled": false
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 1.0
              }
            },
            "font": {
              "size": 30,
              "align": "middle",
              "color": "#111111"
            },
            "smooth": {
              "enabled": false
            }
          }
        }
        """
    )

    n = max(1, len(states_sorted))
    R = 300
    positions: Dict[str, Tuple[float, float]] = {}

    for idx, s in enumerate(states_sorted):
        a = 2.0 * math.pi * idx / n
        positions[s] = (R * math.cos(a), R * math.sin(a))

    def node_bg(state_str: str) -> str:
        if state_str == init and state_str in finals:
            return "#f1c40f"
        if state_str == init:
            return "#2ecc71"
        if state_str in finals:
            return "#e74c3c"
        return "#4a4ae6"

    border = "#2020a0"

    for s in states_sorted:
        sid = id_of[s]
        x, y = positions[s]
        bg = node_bg(s)

        net.add_node(
            sid,
            label=state_label[s],
            title="",
            shape="circle",
            size=55,
            color={
                "background": bg,
                "border": border,
                "highlight": {"background": bg, "border": border},
                "hover": {"background": bg, "border": border},
            },
            font={"size": 22, "color": "#ffffff", "multi": "html"},
            x=x,
            y=y,
            fixed=True,
        )

    edge_map: Dict[Tuple[str, str], List[str]] = {}

    for src, trans in dfa.transitions.items():
        ssrc = str(src)
        for sym, dst in trans.items():
            sdst = str(dst)
            edge_map.setdefault((ssrc, sdst), []).append(str(sym))

    bidir: Set[Tuple[str, str]] = set()

    for a, b in edge_map.keys():
        if a != b and (b, a) in edge_map:
            bidir.add(tuple(sorted((a, b))))

    def ordered_label(symbols: List[str]) -> str:
        return ",".join(sorted(set(symbols)))

    def self_loop_angle(state_str: str) -> float:
        x, y = positions[state_str]
        return math.atan2(y, x)

    for (ssrc, sdst), syms in edge_map.items():
        src_id = id_of[ssrc]
        dst_id = id_of[sdst]
        lab = ordered_label(syms)

        if ssrc == sdst:
            angle = self_loop_angle(ssrc)

            net.add_edge(
                src_id,
                dst_id,
                label=lab,
                title=lab,
                color="#444444",
                width=3,
                arrows="to",
                dashes=True,
                smooth={
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": 0.65,
                },
                selfReference={
                    "size": 65,
                    "angle": angle,
                    "renderBehindTheNode": True,
                },
                font=SELF_LOOP_FONT_BIG,
            )
            continue

        pair = tuple(sorted((ssrc, sdst)))

        if pair in bidir:
            if ssrc < sdst:
                smooth_cfg = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": 0.22,
                }
            else:
                smooth_cfg = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": -0.22,
                }
        else:
            smooth_cfg = {
                "enabled": False
            }

        net.add_edge(
            src_id,
            dst_id,
            label=lab,
            title=lab,
            color="#444444",
            width=2,
            arrows="to",
            smooth=smooth_cfg,
            font=EDGE_FONT_BIG,
        )

    net.write_html(str(out_path))
    return str(out_path)


def draw_DFA_html_option2(dfa) -> str:
    out_dir = get_artifact_dir("DFA")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"DFA_{ts}.html"

    def is_sink_obj(x: Any) -> bool:
        s = str(x).lower()
        return "__sink__" in s or "sink" in s or "reject" in s

    def disp(x: Any) -> str:
        if isinstance(x, frozenset):
            elems = sorted(str(e) for e in x)
            return elems[0] if len(elems) == 1 else "|".join(elems)
        return str(x)

    def coords(x: Any) -> Tuple[float, float] | None:
        if isinstance(x, frozenset):
            pts = [coords(e) for e in x]
            pts2 = [p for p in pts if p is not None]
            if not pts2:
                return None
            return (
                sum(p[0] for p in pts2) / len(pts2),
                sum(p[1] for p in pts2) / len(pts2),
            )

        if hasattr(x, "i") and hasattr(x, "j"):
            return (float(getattr(x, "j")), float(getattr(x, "i")))

        return None

    order = ["up", "down", "left", "right"]
    order_rank = {a: i for i, a in enumerate(order)}

    def ordered_join(syms: List[str]) -> str:
        u = sorted(set(syms), key=lambda s: order_rank.get(s, 999))
        return ",".join(u)

    states_list = list(dfa.states)

    label_of = {s: disp(s) for s in states_list}
    obj_of_label: Dict[str, Any] = {}

    for s in states_list:
        obj_of_label[label_of[s]] = s

    labels = sorted(obj_of_label.keys())
    id_of = {lab: idx for idx, lab in enumerate(labels)}

    init_lab = disp(dfa.initial_state)
    finals_lab = {disp(s) for s in dfa.final_states}

    grid_pts = []

    for s in states_list:
        if is_sink_obj(s):
            continue

        c = coords(s)
        if c is not None:
            grid_pts.append(c)

    max_x = max((x for x, _ in grid_pts), default=0.0)
    max_y = max((y for _, y in grid_pts), default=0.0)

    spacing_x = 300.0
    spacing_y = 240.0

    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.toggle_physics(False)

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          },
          "physics": {
            "enabled": false
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 1.0
              }
            },
            "font": {
              "size": 16,
              "align": "middle",
              "color": "#111111"
            },
            "smooth": {
              "enabled": false
            }
          }
        }
        """
    )

    def node_bg(label: str) -> str:
        if label == init_lab and label in finals_lab:
            return "#f1c40f"
        if label == init_lab:
            return "#2ecc71"
        if label in finals_lab:
            return "#e74c3c"
        if is_sink_obj(label):
            return "#2c3e50"
        return "#4a4ae6"

    border = "#2020a0"
    BOX_W = 120
    BOX_H = 70
    FONT_SIZE = 12

    coord_of_label: Dict[str, Tuple[float, float] | None] = {}

    for lab, obj in obj_of_label.items():
        if is_sink_obj(obj):
            x = (max_x + 2.2) * spacing_x
            y = (max_y / 2.0) * spacing_y
            coord_of_label[lab] = None
        else:
            c = coords(obj)
            coord_of_label[lab] = c

            if c is None:
                x = 0.0
                y = 0.0
            else:
                gx, gy = c
                x = gx * spacing_x
                y = gy * spacing_y

        bg = node_bg(lab)

        net.add_node(
            id_of[lab],
            label=lab,
            title="",
            shape="box",
            widthConstraint={"minimum": BOX_W, "maximum": BOX_W},
            heightConstraint={"minimum": BOX_H},
            color={
                "background": bg,
                "border": border,
                "highlight": {"background": bg, "border": border},
                "hover": {"background": bg, "border": border},
            },
            font={"size": FONT_SIZE, "color": "#ffffff"},
            x=x,
            y=y,
            fixed=True,
        )

    edge_map: Dict[Tuple[str, str], List[str]] = {}

    for src, trans in dfa.transitions.items():
        ssrc = disp(src)
        for sym, dst in trans.items():
            sdst = disp(dst)
            edge_map.setdefault((ssrc, sdst), []).append(str(sym))

    bidir: Set[Tuple[str, str]] = set()

    for a, b in edge_map.keys():
        if a != b and (b, a) in edge_map:
            bidir.add(tuple(sorted((a, b))))

    def self_loop_angle(label: str) -> float:
        c = coord_of_label.get(label)

        if c is None:
            return 0.0

        x, y = c
        cx = max_x / 2.0 if max_x else 0.0
        cy = max_y / 2.0 if max_y else 0.0

        return math.atan2(y - cy, x - cx)

    for (a, b), syms_ab in edge_map.items():
        label = ordered_join(syms_ab)

        if a == b:
            net.add_edge(
                id_of[a],
                id_of[b],
                label=label,
                title=label,
                color="#444444",
                width=3,
                arrows="to",
                dashes=True,
                smooth={
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": 0.65,
                },
                selfReference={
                    "size": 60,
                    "angle": self_loop_angle(a),
                    "renderBehindTheNode": True,
                },
                font=SELF_LOOP_FONT_SMALL,
            )
            continue

        pair = tuple(sorted((a, b)))

        if pair in bidir:
            if a < b:
                smooth_cfg = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": 0.22,
                }
            else:
                smooth_cfg = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": -0.22,
                }
        else:
            smooth_cfg = {
                "enabled": False
            }

        net.add_edge(
            id_of[a],
            id_of[b],
            label=label,
            title=label,
            color="#444444",
            width=2,
            arrows="to",
            smooth=smooth_cfg,
            font=EDGE_FONT_SMALL,
        )

    net.write_html(str(out_path))
    return str(out_path)


def draw_DFA_html_with_option(dfa, option: int = 1) -> str:
    if option == 2:
        return draw_DFA_html_option2(dfa)

    return draw_DFA_html(dfa)