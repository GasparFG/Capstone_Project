"""
Data-centre scheduling optimisation model.
Planning horizon : 1 day
Slot duration    : 1 hour  (24 slots, 00:00 – 23:00)
Solver           : Gurobi

The R_{ijk} variable is eliminated — it is fully determined by X and is
substituted inline wherever it appears, keeping the model within the 2000-
variable / 2000-constraint limit of the restricted Gurobi licence.
With a full licence set N_SLOTS=96 and delta_t=0.25 in data.json for 15-min
fidelity.

Constraint numbers #5-#36 are kept as comments for cross-referencing.
"""

from gurobipy import GRB
import json
import gurobipy as gp

params = {
    "WLSACCESSID": "fc17fa3a-ef7f-41d2-b95c-20c3b221a483",
    "WLSSECRET": "6bee54d1-5c9f-4f12-9d64-0c7b16e0dd52",
    "LICENSEID": 2804943
}
env = gp.Env(params=params)

# create the model within
# model = gp.Model(env=env)


# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------
with open("data.json") as f:
    data = json.load(f)

I = data["sets"]["I"]
I_B = data["sets"]["I_B"]
I_V = data["sets"]["I_V"]
I_C = data["sets"]["I_C"]
J = data["sets"]["J"]
K = data["sets"]["K"]
F = data["sets"]["F"]
E = data["sets"]["E"]
A = data["sets"]["A"]
G = data["sets"]["G"]
S = {int(k): v for k, v in data["eligibility"].items()}

jp = data["job_params"]
d = {int(k): v for k, v in jp["d"].items()}
r = {int(k): v for k, v in jp["r"].items()}
a = {int(k): v for k, v in jp["a"].items()}
b = {int(k): v for k, v in jp["b"].items()}
q = {int(k): v for k, v in jp["q"].items()}
rho = {int(k): v for k, v in jp["rho"].items()}

for i in I:
    if i not in q:
        q[i] = 1

sp = data["server_params"]
C = {int(k): v for k, v in sp["C"].items()}
theta = {int(k): v for k, v in sp["theta"].items()}
P0 = {int(k): v for k, v in sp["P0"].items()}
dP = {int(k): v for k, v in sp["dP"].items()}
alpha = {int(k): v for k, v in sp["alpha"].items()}
lambda0 = {int(k): v for k, v in sp["lambda0"].items()}
lambda_pm = {int(k): v for k, v in sp["lambda_pm"].items()}
Lambda = {int(k): v for k, v in sp["Lambda"].items()}

th = data["thermal"]
T_sup = th["T_sup"]
T_busy = th["T_busy"]
T_idle = th["T_idle"]
M_big = th["M_big"]
D = th["D"]

eta = {k: data["cooling"]["eta"][k] for k in K}
P_ov = data["power"]["P_ov"]
Pi_max = data["power"]["Pi_max"]
d_pm = data["maintenance"]["d_pm"]
c_pm = data["maintenance"]["c_pm"]
c_cm = data["maintenance"]["c_cm"]
c_e = data["costs"]["c_e"]
c_sw = data["costs"]["c_sw"]
S_max = data["costs"]["S_max"]
Dk = {k: data["demand"]["D"][k] for k in K}
N_min = data["redundancy"]["N_min"]
kappa = data["redundancy"]["kappa"]
Q_max = data["redundancy"]["Q_max"]
delta_t = data["slot_duration"]
nK = len(K)


def slot_to_time(k):
    mins = int(k * delta_t * 60)
    return f"{mins // 60:02d}:{mins % 60:02d}"


def valid_starts(job):
    """Valid start slots: honour release time, horizon end, and (for interactive) hard deadline."""
    upper = nK - d[job]
    if job in I_V:
        upper = min(upper, b[job] - d[job])
    return [k for k in K if a[job] <= k <= upper]


def running_at(job, j, k):
    """List of start slots k' such that (job,j,k') is a valid X variable
       and job would be executing at slot k if it started at k'."""
    return [kp for kp in valid_starts(job)
            if (job, j, kp) in X and kp <= k < kp + d[job]]


# ---------------------------------------------------------------------------
# 2.  Build model
# ---------------------------------------------------------------------------
mdl = gp.Model("datacenter_1day",env=env)
mdl.setParam("TimeLimit", 120)
mdl.setParam("MIPGap",    0.02)

# ---------------------------------------------------------------------------
# 3.  Decision variables  (R eliminated — see module docstring)
# ---------------------------------------------------------------------------
X = {(i, j, k): mdl.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
     for i in I for j in S[i] for k in valid_starts(i)}

y = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")
     for j in J for k in K}

d_on = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"don_{j}_{k}")
        for j in J for k in K[:-1]}
d_off = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"doff_{j}_{k}")
         for j in J for k in K[:-1]}

m_j = {j: mdl.addVar(vtype=GRB.BINARY, name=f"m_{j}") for j in J}

# All possible starts for preventive maintenance
pm_starts = [k for k in K if k <= nK - d_pm]

v = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"v_{j}_{k}")
     for j in J for k in pm_starts}


z = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")
     for j in J for k in K}

l_var = {i: mdl.addVar(lb=0.0, name=f"l_{i}") for i in I_B}

L = {(j, k): mdl.addVar(lb=0.0, ub=1.0, name=f"L_{j}_{k}")
     for j in J for k in K}
H = {(j, k): mdl.addVar(lb=0.0, name=f"H_{j}_{k}")
     for j in J for k in K}
PIT = {k: mdl.addVar(lb=0.0, name=f"PIT_{k}") for k in K}
Pcool = {k: mdl.addVar(lb=0.0, name=f"Pcool_{k}") for k in K}
Ptot = {k: mdl.addVar(lb=0.0, name=f"Ptot_{k}") for k in K}

s = {i: mdl.addVar(lb=0.0, ub=nK - 1, name=f"s_{i}") for i in I}

psi = {(j, k): mdl.addVar(lb=0.0, name=f"psi_{j}_{k}")
       for j in J for k in K}

mdl.update()

# ---------------------------------------------------------------------------
# 4.  Objective  (#4)
# ---------------------------------------------------------------------------
energy_cost = c_e * delta_t / 1000.0 * gp.quicksum(Ptot[k] for k in K)
pm_cost = gp.quicksum(c_pm * m_j[j] for j in J)
cm_cost = c_cm * gp.quicksum(
    lambda0[j] * y[j, k] - (lambda0[j] - lambda_pm[j]) * m_j[j] * y[j, k]
    for j in J for k in K)
sw_cost = c_sw * gp.quicksum(d_on[j, k] + d_off[j, k]
                             for j in J for k in K[:-1])
late_cost = gp.quicksum(rho[i] * l_var[i] for i in I_B)

mdl.setObjective(energy_cost + pm_cost + cm_cost + sw_cost + late_cost,
                 GRB.MINIMIZE)

# ---------------------------------------------------------------------------
# 5.  Constraints
# ---------------------------------------------------------------------------

# --- #5  Job assignment (exact replica count) ---
for i in I:
    mdl.addConstr(
        gp.quicksum(X[i, j, k] for j in S[i] for k in valid_starts(i)) == q[i],
        name=f"c5_{i}")

# --- #6/#7  Release time / interactive hard deadline enforced in valid_starts() ---

# --- #8  Precedence ---
for (i_pred, i_succ) in E:
    mdl.addConstr(s[i_succ] >= s[i_pred] + d[i_pred],
                  name=f"c8_{i_pred}_{i_succ}")

# --- Start-time definition ---
for i in I:
    for k in valid_starts(i):
        mdl.addConstr(
            s[i] >= gp.quicksum(k * X[i, j, k]
                            for j in S[i]),
                     name=f"cs_{i}")

# --- #9  Batch-only capacity (interactive reservation) ---
for j in J:
    for k in K:
        batch_load = gp.quicksum(
            r[i] * X[i, j, kp]
            for i in I_B if j in S[i]
            for kp in running_at(i, j, k))
        mdl.addConstr(batch_load <= (1 - theta[j]) * C[j] * y[j, k],
                      name=f"c9_{j}_{k}")

# --- #10  Batch lateness ---
# Non-critical batch jobs: use start-time variable directly (one start, so s[i] is exact)
for i in I_B:
    if i not in I_C:
        mdl.addConstr(l_var[i] >= s[i] + d[i] - b[i], name=f"c10nc_{i}")
# Critical batch jobs: big-M per assignment (handles multiple replicas correctly)
M_lat = nK
for i in I_B:
    if i in I_C:
        for j in S[i]:
            for k in valid_starts(i):
                mdl.addConstr(
                    l_var[i] >= k + d[i] - b[i] - M_lat * (1 - X[i, j, k]),
                    name=f"c10cr_{i}_{j}_{k}")

# --- #12/#13  Load definition + server capacity  (c11 is implied by these two) ---
for j in J:
    for k in K:
        load_expr = gp.quicksum(
            r[i] * X[i, j, kp]
            for i in I if j in S[i]
            for kp in running_at(i, j, k))
        mdl.addConstr(L[j, k] == load_expr, name=f"c13_{j}_{k}")
        mdl.addConstr(L[j, k] <= C[j] * y[j, k], name=f"c12_{j}_{k}")

# --- #14  Server cannot be active and under PM simultaneously ---
for j in J:
    for k in K:
        mdl.addConstr(y[j, k] + z[j, k] <= 1, name=f"c14_{j}_{k}")

# --- #15  Total IT power ---
for k in K:
    mdl.addConstr(
        PIT[k] == gp.quicksum(P0[j] * y[j, k] + dP[j] * L[j, k] for j in J),
        name=f"c15_{k}")

# --- #16  Heat per server ---
for j in J:
    for k in K:
        mdl.addConstr(
            H[j, k] == alpha[j] * (P0[j] * y[j, k] + dP[j] * L[j, k]),
            name=f"c16_{j}_{k}")

# --- #17  Cooling power ---
for k in K:
    mdl.addConstr(
        Pcool[k] == (1.0 / eta[k]) * gp.quicksum(H[j, k] for j in J),
        name=f"c17_{k}")

# --- #18  Total facility power ---
for k in K:
    mdl.addConstr(Ptot[k] == PIT[k] + Pcool[k] + P_ov, name=f"c18_{k}")

# --- #20  PUE cap ---
for k in K:
    mdl.addConstr(Ptot[k] <= Pi_max * PIT[k], name=f"c20_{k}")

# --- #21  Thermal: server inlet temperature ---
for j in J:
    for k in K:
        recirc = gp.quicksum(
            D[j][jp] * alpha[jp] * (P0[jp] * y[jp, k] + dP[jp] * L[jp, k])
            for jp in J)
        mdl.addConstr(
            T_sup + recirc <= T_idle -
            (T_idle - T_busy) * y[j, k] + M_big * z[j, k],
            name=f"c21_{j}_{k}")

# --- #22  PM count per server ---
for j in J:
    mdl.addConstr(
        gp.quicksum(v[j, k] for k in pm_starts) == m_j[j],
        name=f"c22_{j}")

# --- #23  PM active window ---
for j in J:
    for k in K:
        win = [kp for kp in pm_starts if max(0, k - d_pm + 1) <= kp <= k]
        mdl.addConstr(
            z[j, k] == gp.quicksum(v[j, kp] for kp in win),
            name=f"c23_{j}_{k}")

# --- #24  Max servers under PM per slot ---
for k in K:
    mdl.addConstr(
        gp.quicksum(z[j, k] for j in J) <= len(J) - N_min,
        name=f"c24_{k}")

# --- #25  Cumulative load ---
for j in J:
    for k in K:
        if k == 0:
            mdl.addConstr(psi[j, k] == L[j, k], name=f"c25_{j}_{k}")
        else:
            mdl.addConstr(psi[j, k] == psi[j, k - 1] +
                          L[j, k], name=f"c25_{j}_{k}")

# --- #26  PM may only start once cumulative load >= Lambda ---
for j in J:
    for k in pm_starts:
        mdl.addConstr(psi[j, k] >= Lambda[j] * v[j, k], name=f"c26_{j}_{k}")

# --- #27/#28  Server state-change tracking ---
for j in J:
    for k in K[:-1]:
        mdl.addConstr(y[j, k + 1] - y[j, k] <=
                      d_on[j, k],  name=f"c27_{j}_{k}")
        mdl.addConstr(y[j, k] - y[j, k + 1] <=
                      d_off[j, k], name=f"c28_{j}_{k}")

# --- #29  Total switching budget ---
mdl.addConstr(
    gp.quicksum(d_on[j, k] + d_off[j, k] for j in J for k in K[:-1]) <= S_max,
    name="c29")

# --- #30  Minimum aggregate workload demand ---
for k in K:
    mdl.addConstr(gp.quicksum(L[j, k] for j in J) >= Dk[k], name=f"c30_{k}")

# --- #31  Anti-affinity isolation for critical job pairs ---
for (i1, i2) in G:
    if i1 in I_C and i2 in I_C:
        for j in J:
            for k in K:
                r1 = gp.quicksum(X[i1, j, kp] for kp in running_at(i1, j, k))
                r2 = gp.quicksum(X[i2, j, kp] for kp in running_at(i2, j, k))
                mdl.addConstr(r1 + r2 <= 1, name=f"c31_{i1}_{i2}_{j}_{k}")

# --- #32  Affinity: same server ---
for (i1, i2) in A:
    for j in J:
        s1 = gp.quicksum(X[i1, j, k]
                         for k in valid_starts(i1) if (i1, j, k) in X)
        s2 = gp.quicksum(X[i2, j, k]
                         for k in valid_starts(i2) if (i2, j, k) in X)
        mdl.addConstr(s1 == s2, name=f"c32_{i1}_{i2}_{j}")

# --- #33  Affinity critical pairs: same replica count (data check) ---
for (i1, i2) in A:
    if i1 in I_C:
        assert q[i1] == q[i2], f"Affinity pair ({i1},{i2}): replica counts must match"

# --- #34  Hot standby buffer ---
for k in K:
    mdl.addConstr(
        gp.quicksum(y[j, k] for j in J) >= N_min + kappa,
        name=f"c34_{k}")

# --- #35  Rack diversity for critical jobs ---
for i in I_C:
    for f_idx, Ff in enumerate(F):
        mdl.addConstr(
            gp.quicksum(X[i, j, k]
                        for j in Ff if j in S[i]
                        for k in valid_starts(i) if (i, j, k) in X) <= 1,
            name=f"c35_{i}_{f_idx}")

# --- #36  Replication overhead budget ---
mdl.addConstr(
    gp.quicksum((q[i] - 1) * r[i] for i in I_C) <= Q_max,
    name="c36")

# --- Non-critical anti-affinity: can't share a server ---
for (i1, i2) in G:
    if not (i1 in I_C and i2 in I_C):
        for j in J:
            lhs = gp.quicksum(X[i1, j, k]
                              for k in valid_starts(i1) if (i1, j, k) in X)
            rhs = gp.quicksum(X[i2, j, k]
                              for k in valid_starts(i2) if (i2, j, k) in X)
            mdl.addConstr(lhs + rhs <= 1, name=f"c_nca_{i1}_{i2}_{j}")

# ---------------------------------------------------------------------------
# 6.  Solve
# ---------------------------------------------------------------------------
print(f"\nModel: {mdl.NumVars} variables ({mdl.NumBinVars} binary), "
      f"{mdl.NumConstrs} constraints\n")
mdl.optimize()

# ---------------------------------------------------------------------------
# 7.  Report results
# ---------------------------------------------------------------------------
SEP = "=" * 65


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


if mdl.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and mdl.SolCount > 0:

    gap_pct = 100.0 * mdl.MIPGap
    section("OBJECTIVE BREAKDOWN")
    print(f"  Total cost          : {mdl.ObjVal:10.4f}  (gap {gap_pct:.2f}%)")
    print(f"  (I)  Energy cost    : {energy_cost.getValue():10.4f}")
    print(f"  (II) PM fixed cost  : {pm_cost.getValue():10.4f}")
    print(f"  (III) Expected CM   : {cm_cost.getValue():10.4f}")
    print(f"  (IV) Switching cost : {sw_cost.getValue():10.4f}")
    print(f"  (V)  Lateness       : {late_cost.getValue():10.4f}")

    section("JOB SCHEDULE")
    hdr = f"  {'Job':>3}  {'Type':>13}  {'Srv':>3}  {'Start':>5}  {'End':>5}  {'Dur':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in sorted(I):
        for j in S[i]:
            for k in valid_starts(i):
                if (i, j, k) in X and X[i, j, k].X > 0.5:
                    tag = "batch" if i in I_B else "interactive"
                    crit = " [CRIT]" if i in I_C else "       "
                    print(f"  {i:>3}  {tag:>8}{crit}  {j:>3}  "
                          f"{slot_to_time(k):>5}  {slot_to_time(k + d[i]):>5}  "
                          f"{d[i] * delta_t:>4.1f}h")

    section("SERVER SCHEDULE  (1=active, P=under PM, 0=off)")
    print("  Srv | " + " ".join(f"{slot_to_time(k)}" for k in K[::4]))
    print("      | " + " ".join(f"{'—':>5}" for _ in K[::4]))
    for j in J:
        row = []
        for k in K:
            if z[j, k].X > 0.5:
                row.append("P")
            elif y[j, k].X > 0.5:
                row.append("1")
            else:
                row.append("0")
        # print every 4th slot (hourly ticks if 15-min slots, or every 4 h for hourly)
        print(f"    {j} | " + "  ".join(row))

    section("PM SCHEDULE")
    for j in J:
        if m_j[j].X > 0.5:
            starts = [k for k in pm_starts if v[j, k].X > 0.5]
            windows = [
                f"{slot_to_time(k)}–{slot_to_time(k + d_pm)}" for k in starts]
            print(f"  Server {j}: PM window {windows}  "
                  f"(failure rate {lambda0[j]:.3f} -> {lambda_pm[j]:.3f})")
        else:
            print(f"  Server {j}: no PM scheduled")

    section("HOURLY ENERGY & THERMAL")
    total_kwh = 0.0
    print(f"  {'Hr':>2}  {'PIT(W)':>8}  {'Pcool(W)':>9}  {'Ptot(W)':>8}  "
          f"{'PUE':>5}  {'COP':>4}  {'Srv on':>6}")
    print("  " + "-" * 58)
    for k in K:
        pit = PIT[k].X
        pcl = Pcool[k].X
        ptot = Ptot[k].X
        pue = ptot / pit if pit > 1e-6 else float("nan")
        n_on = sum(1 for j in J if y[j, k].X > 0.5)
        total_kwh += ptot * delta_t / 1000
        print(f"  {k:>2}  {pit:>8.1f}  {pcl:>9.1f}  {ptot:>8.1f}  "
              f"{pue:>5.3f}  {eta[k]:>4.1f}  {n_on:>6}")
    print(f"\n  Total facility energy: {total_kwh:.3f} kWh")

    section("BATCH LATENESS")
    for i in I_B:
        late = max(0.0, l_var[i].X)
        if late > 0.01:
            print(f"  Job {i}: {late:.2f} slots = {late * delta_t:.2f} h late")
        else:
            print(f"  Job {i}: on time")

else:
    print(f"\nSolver status: {mdl.status}  — no feasible solution found.")
