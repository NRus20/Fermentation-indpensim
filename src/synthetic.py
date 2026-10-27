"""SYNTHETIC stand-in for IndPenSim, used ONLY to smoke-test the pipeline (CI / no-network environments).
Mimics the column naming, batch layout (1-30 recipe, 31-60 operator, 61-90 APC+Raman, 91-100 faults) and
sampling, but the dynamics are toy. NEVER report results obtained from this file."""
import numpy as np, pandas as pd


def make(n_batches=100, dt=0.5, seed=0):
    """returns (timeseries_df, yield_stats_df)"""
    rng = np.random.RandomState(seed); out = []
    for b in range(1, n_batches + 1):
        strat = 0 if b <= 30 else 1 if b <= 60 else 2 if b <= 90 else 0
        T = rng.randint(190, 240); t = np.arange(0, T, dt); n = len(t); f = t / T
        qual = rng.normal(0, 1); lag = 1 - np.exp(-t / 40)
        X = 20 * lag * (1 - .2 * f) + rng.normal(0, .3, n); S = np.maximum(.05, 1.5 - .008 * t + rng.normal(0, .05, n))
        ph = 6.5 + .08 * np.sin(t / 12) + .05 * qual + rng.normal(0, .02, n)
        do2 = 15 - 6 * lag + .5 * qual + rng.normal(0, .3, n); temp = 298 + rng.normal(0, .15, n)
        base = 30 + 4 * strat + 3 * qual
        P = np.maximum(0, base * (1 / (1 + np.exp(-(t - 80) / 25))) * (1 - .05 * f)) + rng.normal(0, .1, n)
        our = 40 * lag + rng.normal(0, 1, n); cer = 0.9 * our + rng.normal(0, 1, n)
        ff = np.zeros(n); fr = np.zeros(n)
        if b > 90:
            on = rng.randint(80, 150); m = t >= on; ff[m] = 1; fr[m] = 1
            ph[m] += .25 + .004 * (t[m] - on); do2[m] -= 2 + .02 * (t[m] - on); P[m] *= np.linspace(1, .6, m.sum())
        z = lambda v: np.full(n, v, float)
        d = pd.DataFrame({"Time (h)": t + dt, "Aeration rate(Fg:L/h)": 60 + rng.normal(0, 1, n),
                          "Agitator RPM(RPM:RPM)": 100 + rng.normal(0, 1, n),
                          "Sugar feed rate(Fs:L/h)": 8 + rng.normal(0, .3, n) + strat,
                          "Acid flow rate(Fa:L/h)": np.abs(rng.normal(0, .05, n)), "Base flow rate(Fb:L/h)": np.abs(rng.normal(.1, .05, n)),
                          "Heating/cooling water flow rate(Fc:L/h)": 20 + rng.normal(0, 1, n) + 5 * lag,
                          "Heating water flow rate(Fh:L/h)": z(0.0001), "Water for injection/dilution(Fw:L/h)": z(0),
                          "Air head pressure(pressure:bar)": z(.6), "Dumped broth flow(Fremoved:L/h)": z(0),
                          "Substrate concentration(S:g/L)": S, "Dissolved oxygen concentration(DO2:mg/L)": do2,
                          "Penicillin concentration(P:g/L)": P, "Vessel Volume(V:L)": 58000 + 20 * t,
                          "Vessel Weight(Wt:Kg)": 62000 + 22 * t, "pH(pH:pH)": ph, "Temperature(T:K)": temp,
                          "Generated heat(Q:kJ)": 1e-6 + 100 * lag + rng.normal(0, 3, n),
                          "carbon dioxide percent in off-gas(CO2outgas:%)": .04 + .8 * lag * .01 + rng.normal(0, .002, n),
                          "PAA flow(Fpaa:PAA flow (L/h))": 5 + rng.normal(0, .2, n),
                          "PAA concentration offline(PAA_offline:PAA (g L^{-1}))": np.where(np.arange(n) % 60 == 0, 1.0, np.nan),
                          "Oil flow(Foil:L/hr)": np.abs(rng.normal(0, 1, n)),
                          "NH3 concentration off-line(NH3_offline:NH3 (g L^{-1}))": np.where(np.arange(n) % 60 == 0, 1.0, np.nan),
                          "Oxygen Uptake Rate(OUR:(g min^{-1}))": our, "Oxygen in percent in off-gas(O2:O2  (%))": 20 - .5 * lag + rng.normal(0, .05, n),
                          "Offline Penicillin concentration(P_offline:P(g L^{-1}))": np.where(np.arange(n) % 60 == 0, P, np.nan),
                          "Offline Biomass concentratio(X_offline:X(g L^{-1}))": np.where(np.arange(n) % 60 == 0, X, np.nan),
                          "Carbon evolution rate(CER:g/h)": cer, "Ammonia shots(NH3_shots:kgs)": z(0),
                          "Viscosity(Viscosity_offline:centPoise)": np.where(np.arange(n) % 60 == 0, 50.0, np.nan),
                          "Fault reference(Fault_ref:Fault ref)": fr,
                          "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)": z(int(strat == 1)),
                          "1- No Raman spec": z(int(strat != 2)), "1-Raman spec recorded": z(int(strat == 2)),
                          "2-PAT control(PAT_ref:PAT ref)": z(int(strat == 2)),
                          "Batch reference(Batch_ref:Batch ref)": z(b), "Batch ID": b, "Fault flag": ff,
                          "2400": rng.rand(n), "2399": rng.rand(n)})
        d["_T"] = P[-1]  # carried for synthetic yield table
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    yl = df.groupby("Batch ID")["_T"].first(); df = df.drop(columns="_T")
    ys = pd.DataFrame({"Batch ref": yl.index, "Penicllin_harvested_during_batch(kg)": yl.values * 3e4,
                       "Penicllin_harvested_end_of_batch (kg)": yl.values * 5e4, "Penicllin_yield_total (kg)": yl.values * 8e4,
                       "Fault ref(0-NoFault 1-Fault)": (yl.index > 90).astype(int)})
    return df, ys
