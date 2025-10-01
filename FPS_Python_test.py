#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte‑Carlo Compromise Search (à partir d'un .dot) :
  • parse le graphe d'attaque depuis un .dot (pydot)
  • lit l'Excel (P0 par TID, ε par (mitigation, TID), coûts)
  • pour chaque budget :
       MINLP exact (Pyomo) -> sélection de contrôles
       + N_MC tirages Monte‑Carlo (P0, ε) -> stats de risque
  • figures : frontier.png, risk_hist.png, chemins_critiques.png
  • affichage des mesures sélectionnées AVEC, entre parenthèses, les nœuds concernés

Prérequis :
  pip install pyomo pandas numpy scipy tqdm matplotlib pydot
  # et un solveur MINLP pour optimiser :
  # BONMIN (COIN-OR) -> SOLVER_NAME="bonmin" + BONMIN_EXEC
  # ou "scip" / "couenne" / "baron" si installés
"""

# --------------------- PARAMÈTRES UTILISATEUR -----------------------------
import matplotlib
matplotlib.use("Agg")  # backend non-interactif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from scipy.stats import beta, triang, gaussian_kde
import pydot
from tqdm import trange
from pathlib import Path
import re, math, json
from collections import defaultdict

# === FICHIERS (édite ces chemins) ===
DOT_PATH   = r"Good_graph.dot"   # <-- ton .dot
EXCEL_PATH = r"Attack_Measures1.xlsx"      # <-- ton Excel complété
OUT_DIR    = r"OUT"                                        # dossier de sortie (créé si n'existe pas)

# === SOLVEUR MINLP ===
SOLVER_NAME = "bonmin"                 # "bonmin" / "scip" / "couenne" / "baron" ; "" = pas d'optimisation (x=0)
BONMIN_EXEC = r"/usr/local/bin/bonmin" # chemin exécutable bonmin (si utilisé)

# === PARAMÈTRES RISQUE/BUDGET/MC ===
theta_global   = 3500.0                 # seuil de risque total (€)
target_prob    = 0.85                   # proba‑cible P(R ≤ θ)
budgets_to_try = list(range(40_000, 200_000, 10_000))   # 15k → 45k
N_MC           = 3000                   # tirages Monte‑Carlo par budget
SEED           = 42                     # seed RNG
CONCENTRATION  = 50                     # Beta concentration autour de P0
TRI_WIDTH      = 0.20                   # largeur (asym.) triangulaire autour du mode ε
DEFAULT_P0     = 0.30                   # fallback si P0 absent/invalide

# === IMPACTS par attribut (monnaie) — clés alignées sur le .dot : C,I,A,Au,R,AC,Authz
impact = {"C": 12000, "I": 10000, "A": 7500, "Au": 1500, "R": 2500, "AC": 5000, "Authz": 5000}

# --------------------- OUTILS & PARSING -----------------------------------
ATTR_KEYS = ["C","I","A","Au","R","AC","Authz"]
RNG = np.random.default_rng(SEED)

def strip_quotes(s: str) -> str:
    return s.strip('"').strip() if isinstance(s, str) else s

def extract_tid_from_label(label: str):
    if not isinstance(label, str): return None
    m = re.search(r'\[(T\d{4}(?:\.\d{3})?)\]', label)
    return m.group(1).upper() if m else None

def parse_dot(dot_path: str):
    """
    Reconstruit :
      - nodes_tech : ids de nœuds techniques (hors attributs)
      - parents    : dict[node] -> [parents]
      - node_tid   : dict[node] -> TID MITRE (Txxxx[.xxx]) ou None
      - leaves_by_attr : dict[attr] -> [nœuds techniques menant à l'attribut]
      - roots      : racines (aucun parent)
    """
    graphs = pydot.graph_from_dot_file(dot_path)
    if not graphs:
        raise RuntimeError(f"Impossible de lire le DOT : {dot_path}")
    g = graphs[0]

    node_tid = {}
    nodes_seen = set()

    for n in g.get_nodes():
        nid = strip_quotes(n.get_name())
        if nid in ('graph', 'node', 'edge'):
            continue
        nodes_seen.add(nid)
        label = n.get_label()
        tid = extract_tid_from_label(label if label else "")
        node_tid[nid] = tid

    parents = defaultdict(list)
    leaves_by_attr = {k: [] for k in ATTR_KEYS}

    for e in g.get_edges():
        u = strip_quotes(e.get_source()); v = strip_quotes(e.get_destination())
        if u in ('graph', 'node', 'edge') or v in ('graph', 'node', 'edge'):
            continue
        nodes_seen.add(u); nodes_seen.add(v)
        if v in ATTR_KEYS:
            leaves_by_attr[v].append(u)       # u -> attribut v
        else:
            parents[v].append(u)              # u -> v (technique -> technique)

    nodes_tech = [n for n in nodes_seen if n not in ATTR_KEYS]
    for n in nodes_tech:
        parents.setdefault(n, [])
    roots = [n for n in nodes_tech if len(parents[n]) == 0]

    return nodes_tech, parents, node_tid, leaves_by_attr, roots

def read_measures(excel_path: str):
    """
    Renvoie :
      - P0_by_tid : dict[TID] -> P0∈[0,1] (moyenne robuste par TID)
      - controls  : dict[mid] -> {"targets": set(TID), "eff": dict[TID]->ε, "cost": float}
    """
    xls = pd.ExcelFile(excel_path)
    frames = [pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names]
    df = pd.concat(frames, ignore_index=True)

    cols_lower = {c.lower(): c for c in df.columns}
    def getcol(*cands):
        for c in cands:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    col_tid  = getcol("technique","technique_id","tid","tactic_id","id","title","name","description")
    col_p0   = getcol("technique_probability","p0","p_0","p0_t")
    col_mid  = getcol("mitigation_id","mid","control_id")
    col_eff  = getcol("mitigation_effectiveness","epsilon","effectiveness")
    col_cost = getcol("implementation_cost_usd","cost","cost_usd")

    def extract_tid(x):
        if not isinstance(x, str): return None
        m = re.search(r'(T\d{4}(?:\.\d{3})?)', x)
        return m.group(1).upper() if m else None

    # P0 par TID (robuste)
    P0_by_tid = {}
    if col_tid and col_p0:
        tmp = df[[col_tid, col_p0]].copy()
        tmp["tid"] = tmp[col_tid].apply(extract_tid)
        tmp = tmp.dropna(subset=["tid"])
        tmp["pnum"] = pd.to_numeric(tmp[col_p0], errors="coerce")
        grp = tmp.groupby("tid")["pnum"].mean()    # ignore NaN
        grp = grp.fillna(DEFAULT_P0).clip(lower=0.0, upper=1.0)
        P0_by_tid = grp.to_dict()
    if not P0_by_tid:
        print("[WARN] P0_by_tid vide — DEFAULT_P0 utilisé partout.")

    # Controls
    controls = {}
    if col_mid:
        for _, r in df.iterrows():
            mid = r[col_mid]
            if not isinstance(mid, str) or not mid.strip():
                continue
            mid = mid.strip().replace("\xa0","").replace(" ", "")
            controls.setdefault(mid, {"targets": set(), "eff": {}, "cost": 0.0})

            if col_cost and pd.notna(r[col_cost]):
                controls[mid]["cost"] = max(controls[mid]["cost"], float(r[col_cost]))

            if col_tid and col_eff and isinstance(r[col_tid], str) and pd.notna(r[col_eff]):
                tid = extract_tid(r[col_tid])
                if tid:
                    eff = float(r[col_eff])
                    eff = max(0.0, min(1.0, eff))
                    controls[mid]["targets"].add(tid)
                    controls[mid]["eff"][tid] = max(controls[mid]["eff"].get(tid, 0.0), eff)

    for mid in controls:
        if controls[mid]["cost"] <= 0.0:
            controls[mid]["cost"] = 15000.0

    return P0_by_tid, controls

# --------------------- MINLP exact (Option C) ------------------------------

def solve_minlp(nodes, parents, node_tid, leaves_by_attr, P0_by_tid, controls,
                theta_global_eur: float, budget_max: float):
    """
    Modèle exact :
      p_t = P0_t * Π_m (1 - ε_{m,tid(t)} x_m)
      q_t = p_t                      si t racine
            p_t * (1 - Π_p (1 - q_p)) sinon
      r_k = I_k * (1 - Π_{l∈L_k} (1 - q_l))
    r_k ≤ θ_k (réparti ∝ impact) et budget ≤ B.
    Retourne la liste des mid sélectionnés (x_m=1), ou [] si infeasible.
    """
    # P0 par nœud via TID (clamp + fallback)
    P0_node = {}
    for t in nodes:
        tid = node_tid.get(t)
        v = P0_by_tid.get(tid, DEFAULT_P0) if tid else DEFAULT_P0
        if v is None or (isinstance(v, float) and (np.isnan(v) or v < 0 or v > 1)):
            v = DEFAULT_P0
        P0_node[t] = float(np.clip(v, 0.0, 1.0))

    # Ensembles
    T = sorted(nodes)
    M = sorted(controls.keys())
    K = [k for k in ATTR_KEYS if len(leaves_by_attr.get(k, [])) > 0]

    # θ_k proportionnel à l'impact
    tot_imp = sum(impact[k] for k in K) if K else 1.0
    theta = {k: theta_global_eur * (impact[k] / tot_imp) for k in K}

    # Modèle
    m = pyo.ConcreteModel("ControlSelectionOptionC")
    m.T = pyo.Set(initialize=T)
    m.M = pyo.Set(initialize=M)
    m.K = pyo.Set(initialize=K)

    m.P0 = pyo.Param(m.T, initialize=P0_node, within=pyo.UnitInterval)
    m.C  = pyo.Param(m.M, initialize={mid: controls[mid]["cost"] for mid in M}, within=pyo.NonNegativeReals)

    def eps_init(_, t, mid):
        tid = node_tid.get(t)
        if not tid: return 0.0
        return float(controls[mid]["eff"].get(tid, 0.0))
    m.eps = pyo.Param(m.T, m.M, initialize=eps_init, within=pyo.UnitInterval)

    par_map = {t: tuple(sorted(parents.get(t, []))) for t in T}
    m.Par = pyo.Set(m.T, initialize=lambda _, t: par_map.get(t, tuple()))

    Lk_map = {k: tuple(sorted(set(leaves_by_attr.get(k, [])) & set(T))) for k in K}
    m.L = pyo.Set(m.K, initialize=lambda _, k: Lk_map[k])

    m.I  = pyo.Param(m.K, initialize={k: impact[k] for k in K}, within=pyo.NonNegativeReals)
    m.TH = pyo.Param(m.K, initialize=theta, within=pyo.NonNegativeReals)

    m.x = pyo.Var(m.M, within=pyo.Binary)
    m.p = pyo.Var(m.T, bounds=(0,1))
    m.q = pyo.Var(m.T, bounds=(0,1))
    m.r = pyo.Var(m.K, within=pyo.NonNegativeReals)

    # p_t = P0_t * Π_m (1 - eps_{m,t} x_m)  (robuste : pas de .value)
    def p_rule(_, t):
        if len(list(m.M)) == 0:
            return m.p[t] == m.P0[t]
        return m.p[t] == m.P0[t] * pyo.prod((1 - m.eps[t, mid] * m.x[mid]) for mid in m.M)
    m.pdef = pyo.Constraint(m.T, rule=p_rule)

    roots = set([t for t in T if len(par_map[t]) == 0])

    # q_t
    def q_rule(_, t):
        if t in roots:
            return m.q[t] == m.p[t]
        return m.q[t] == m.p[t] * (1 - pyo.prod((1 - m.q[p]) for p in m.Par[t]))
    m.qprop = pyo.Constraint(m.T, rule=q_rule)

    # r_k
    def r_rule(_, k):
        Lk = list(m.L[k])
        if not Lk:
            return m.r[k] == 0.0
        return m.r[k] == m.I[k] * (1 - pyo.prod((1 - m.q[l]) for l in Lk))
    m.risk = pyo.Constraint(m.K, rule=r_rule)

    # Seuils + budget
    if len(list(m.K)) > 0:
        m.thres  = pyo.Constraint(m.K, rule=lambda _, k: m.r[k] <= m.TH[k])
    if len(list(m.M)) > 0:
        m.budget = pyo.Constraint(expr=sum(m.C[mid] * m.x[mid] for mid in m.M) <= float(budget_max))

    # Objectif coût
    obj_expr = sum(m.C[mid] * m.x[mid] for mid in m.M) if len(list(m.M)) > 0 else 0.0
    m.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

    # Solve
    if not SOLVER_NAME or len(list(m.M)) == 0:
        if not SOLVER_NAME:
            print("[WARN] SOLVER_NAME vide → pas d’optimisation (x=0).")
        if len(list(m.M)) == 0:
            print("[WARN] Aucune mitigation détectée dans l’Excel → x=0.")
        return []

    try:
        opt = (pyo.SolverFactory("bonmin", executable=BONMIN_EXEC)
               if SOLVER_NAME.lower() == "bonmin" else
               pyo.SolverFactory(SOLVER_NAME))
        if (opt is None) or (not opt.available()):
            print(f"[WARN] Solveur '{SOLVER_NAME}' non disponible. x=0 utilisé.")
            return []
        if SOLVER_NAME.lower() == "bonmin":
            opt.options["bonmin.algorithm"] = "B-BB"
            opt.options["bonmin.time_limit"] = 300
        results = opt.solve(m, tee=False)
        tc = results.solver.termination_condition
        st = results.solver.status
        if tc not in (TerminationCondition.optimal,
                      TerminationCondition.locallyOptimal,
                      TerminationCondition.feasible):
            print(f"[WARN] MINLP non résolu (status={st}, tc={tc}). Plan vide (x=0).")
            return []
    except Exception as e:
        print(f"[WARN] Échec MINLP ({SOLVER_NAME}) : {e}. Plan vide (x=0).")
        return []

    selected = [mid for mid in m.M if (m.x[mid].value or 0) > 0.5]
    return list(map(str, selected))

# --------------------- MONTE‑CARLO -----------------------------------------

def sample_params(P0_node, controls):
    """Tire P0_s par nœud (Beta) et ε_s par (mid, tid) (Triangulaire)."""
    P0_s_by_node = {}
    for t, p0 in P0_node.items():
        p = float(np.clip(p0, 1e-3, 1-1e-3))
        a = p * CONCENTRATION
        b = (1 - p) * CONCENTRATION
        P0_s_by_node[t] = beta.rvs(a, b, random_state=RNG)

    eff_s_by_tid = {}
    for mid, obj in controls.items():
        for tid, mode in obj["eff"].items():
            low = 0.0
            high = min(1.0, mode + TRI_WIDTH)
            if high <= low + 1e-9:
                eff_s_by_tid[(mid, tid)] = float(mode)
                continue
            c = (mode - low) / (high - low)
            eff_s_by_tid[(mid, tid)] = triang.rvs(c, loc=low, scale=high - low, random_state=RNG)
    return P0_s_by_node, eff_s_by_tid

def risk_calc(xbin, nodes, parents, node_tid, leaves_by_attr, P0_s_by_node, eff_s_by_tid, controls):
    """Calcule le risque par attribut et le total pour un tirage MC."""
    # p_t
    p = {}
    for t in nodes:
        tid = node_tid.get(t)
        prod = 1.0
        if tid:
            for mid, on in xbin.items():
                if on and tid in controls[mid]["targets"]:
                    prod *= (1 - eff_s_by_tid.get((mid, tid), 0.0))
        p[t] = P0_s_by_node[t] * prod

    # q_t (mémo)
    q_cache = {}
    def q_of(u):
        if u in q_cache:
            return q_cache[u]
        pars = parents.get(u, [])
        if not pars:
            val = p[u]
        else:
            val = p[u] * (1 - np.prod([1 - q_of(pp) for pp in pars]))
        q_cache[u] = float(val)
        return q_cache[u]

    # r_k
    risks = {}
    for a in impact:
        leaves = [l for l in leaves_by_attr.get(a, []) if l in nodes]
        if not leaves:
            risks[a] = 0.0
        else:
            risks[a] = impact[a] * (1 - np.prod([1 - q_of(l) for l in leaves]))
    return risks, float(sum(risks.values()))

# --------------------- AIDES D'AFFICHAGE (nouveau) -------------------------

def nodes_for_control(mid: str, nodes, node_tid, controls):
    """Renvoie la liste triée des nœuds (ids) auxquels la mitigation 'mid' s'applique (d'après les TID ciblées)."""
    targets = controls[mid]["targets"]
    return sorted([t for t in nodes if node_tid.get(t) in targets])

def format_selected_controls_with_nodes(selected_list, nodes, node_tid, controls):
    """Formate 'MID (n1,n2,...)' pour l'affichage console."""
    parts = []
    for mid in selected_list:
        nn = nodes_for_control(mid, nodes, node_tid, controls)
        nodes_str = ",".join(nn) if nn else "—"
        parts.append(f"{mid} ({nodes_str})")
    return ", ".join(parts)

# --------------------- PIPELINE PRINCIPAL ----------------------------------

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Graphe .dot
    nodes, parents, node_tid, leaves_by_attr, roots = parse_dot(DOT_PATH)
    print(f"[INFO] Graphe : {len(nodes)} nœuds techniques, {sum(len(v) for v in parents.values())} arêtes t→t")
    print(f"[INFO] Racines : {len(roots)}  |  Attributs présents : {[k for k in ATTR_KEYS if leaves_by_attr.get(k)]}")

    # 2) Excel : P0_by_tid, controls
    P0_by_tid, controls = read_measures(EXCEL_PATH)

    # P0 par nœud (pour MC & chemins)
    P0_node = {t: float(np.clip(P0_by_tid.get(node_tid.get(t), DEFAULT_P0), 0, 1)) for t in nodes}

    # 3) Balayage budgets → MINLP + MC
    frontier = []
    best_plan = None

    print("\n------ FRONTIER & selected controls ------")
    print("Budget(k)\tCost(k)\tMeanR\tP(R≤θ)\tControls (with nodes)")

    for B in budgets_to_try:
        # MINLP
        x_sel = solve_minlp(nodes, parents, node_tid, leaves_by_attr, P0_by_tid, controls,
                            theta_global_eur=theta_global, budget_max=float(B))
        xdict = {m: 1 for m in x_sel}
        ctrl_str = format_selected_controls_with_nodes(x_sel, nodes, node_tid, controls)

        # MC
        totals = []
        for _ in trange(N_MC, desc=f"MC {B//1000}k", leave=False):
            P0_s_by_node, eff_s_by_tid = sample_params(P0_node, controls)
            _, total_r = risk_calc(xdict, nodes, parents, node_tid, leaves_by_attr,
                                   P0_s_by_node, eff_s_by_tid, controls)
            totals.append(total_r)

        R_arr = np.asarray(totals)
        prob_ok = float((R_arr <= theta_global).mean())
        mean_r  = float(R_arr.mean())
        cost    = float(sum(controls[m]["cost"] for m in x_sel))
        frontier.append((B, cost, mean_r, prob_ok, x_sel, R_arr))

        print(f"{B/1e3:.0f}\t\t{cost/1e3:.1f}\t{mean_r:.0f}\t{prob_ok:.1%}\t{ctrl_str if ctrl_str else '—'}")

        if prob_ok >= target_prob and best_plan is None:
            best_plan = frontier[-1]

    if best_plan is None:
        best_plan = frontier[-1]
        print("\n❌ Aucun budget testé n’atteint la proba cible — on garde le plus élevé.")
    B_star, c_star, mR_star, p_star, x_star, R_series = best_plan
    ctrl_str_star = format_selected_controls_with_nodes(x_star, nodes, node_tid, controls)
    print(f"\n✅ Plan compromis : budget {B_star/1e3:.0f}k (coût {c_star/1e3:.1f}k) – P(R≤θ) = {p_star:.1%}")
    print("Contrôles sélectionnés :", ctrl_str_star if ctrl_str_star else "—")

    # 4) Figures
    # -- risk_hist.png
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(R_series, bins=30, density=True, alpha=0.7, edgecolor="white", linewidth=0.4, label="Histogram")
    kde = gaussian_kde(R_series)
    xs = np.linspace(R_series.min(), R_series.max(), 200)
    ax.plot(xs, kde(xs), lw=2, label="KDE")
    ax.axvline(theta_global, color="red", ls="--", label=r"$\theta_{\mathrm{global}}$")
    ax.set_xlabel("Total residual risk  R  ($)")
    ax.set_ylabel("Density")
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(Path(OUT_DIR)/"risk_hist.png"), dpi=300)
    plt.close()

    # -- frontier.png
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([B for B, *_ in frontier],
            [r for _,_,r,_,_,_ in frontier],
            marker="o", lw=1.2)
    ax.scatter([B_star], [mR_star], color="red", zorder=5,
               label=f"Compromise ({B_star/1e3:.0f}k $)")
    ax.set_xlabel("Budget limit ($)")
    ax.set_ylabel("Mean residual risk  E[R] ($)")
    ax.set_title(" ")
    ax.grid(ls=":")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(Path(OUT_DIR)/"frontier.png"), dpi=300)
    plt.close()

    # -- chemins_critiques.png
    def q_nom(t, cache={}):
        if t in cache:
            return cache[t]
        tid = node_tid.get(t)
        prod = 1.0
        if tid:
            for m in x_star:
                if tid in controls[m]["targets"]:
                    prod *= (1 - controls[m]["eff"][tid])
        p_t = P0_node[t] * prod
        pars = parents.get(t, [])
        if not pars:
            cache[t] = p_t
        else:
            cache[t] = p_t * (1 - np.prod([1 - q_nom(p) for p in pars]))
        return cache[t]

    all_leaves = sorted(set(sum([leaves_by_attr.get(k, []) for k in ATTR_KEYS], [])))
    crit = sorted([(l, q_nom(l)) for l in all_leaves], key=lambda x: x[1], reverse=True)[:3]

    graph = pydot.Dot(graph_type="digraph")
    added = set()

    def choose_parent(u):
        pars = parents.get(u, [])
        if not pars: return None
        return max(pars, key=lambda p: q_nom(p))

    def resid_risk_attr(attr):
        leaves = [l for l in leaves_by_attr.get(attr, []) if l in nodes]
        return impact[attr] * (1 - np.prod([1 - q_nom(l) for l in leaves])) if leaves else 0.0

    for leaf, _ in crit:
        path = [leaf]
        while True:
            p = choose_parent(path[-1])
            if not p: break
            path.append(p)
        path.reverse()

        attr = None
        for k in ATTR_KEYS:
            if leaf in leaves_by_attr.get(k, []):
                attr = k; break
        term_id = f"TERM_{attr or 'ATT'}"
        term_label = f"{attr or 'Attr'}\\nResidual Risk: {resid_risk_attr(attr) if attr else 0:.0f}"

        for n in path + [term_id]:
            if n in added: continue
            fill = "#FFD966" if n in roots else "#EA9999" if n == term_id else "lightblue"
            if n == term_id:
                label = term_label
            else:
                tid = node_tid.get(n) or "-"
                # mentionne aussi, ligne 3, les mitigations appliquées à ce nœud
                mids = [m for m in x_star if (node_tid.get(n) and node_tid[n] in controls[m]["targets"])]
                label = f"{n}\\n{tid}\\n" + (", ".join(mids) if mids else "—")
            graph.add_node(pydot.Node(n, shape="box", style="filled",
                                      fillcolor=fill, color="transparent",
                                      label=label))
            added.add(n)
        for u, v in zip(path + [term_id], (path + [term_id])[1:]):
            graph.add_edge(pydot.Edge(u, v))

    graph.write_png(str(Path(OUT_DIR)/"chemins_critiques.png"))

    # 5) JSON récap (avec les nœuds par mesure)
    out_json = Path(OUT_DIR)/"frontier_summary.json"
    payload = []
    for (B, cost, mean_r, prob_ok, x_sel, R_arr) in frontier:
        mapping = {mid: nodes_for_control(mid, nodes, node_tid, controls) for mid in x_sel}
        payload.append({
            "budget": B,
            "cost_selected": cost,
            "E_R": mean_r,
            "P_R_le_theta": prob_ok,
            "selected_controls": x_sel,
            "selected_controls_nodes": mapping
        })
    best_mapping = {mid: nodes_for_control(mid, nodes, node_tid, controls) for mid in x_star}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"theta_global": theta_global, "frontier": payload,
                   "best_plan": {"budget": B_star, "cost": c_star, "E_R": mR_star, "P_R_le_theta": p_star,
                                 "selected_controls": x_star,
                                 "selected_controls_nodes": best_mapping}},
                  f, indent=2)

    print("→ Figures écrites dans :", OUT_DIR)
    print("   - risk_hist.png")
    print("   - frontier.png")
    print("   - chemins_critiques.png")
    print("→ Résumé JSON :", out_json)

if __name__ == "__main__":
    main()
