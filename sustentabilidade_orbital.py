import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)


matplotlib.use("Agg")

URL = "https://celestrak.org/pub/satcat.csv"
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida")
os.makedirs(SAIDA, exist_ok=True)

# Variáveis numéricas analisadas (rótulo -> coluna)
VARIAVEIS = {
    "Altitude (km)": "altitude",
    "Período (min)": "period",
    "Inclinação (°)": "inclination",
    "Velocidade (km/s)": "vel_kms",
    "Idade (anos)": "age_years",
}



# 1. COLETA

def coletar(url=URL):
    print(">> Baixando catálogo oficial CelesTrak...")
    df = pd.read_csv(url)
    print(f"   Objetos no catálogo: {len(df)}")
    print(f"   Colunas: {list(df.columns)}")
    return df



# 2. PREPARAÇÃO E VARIÁVEIS DERIVADAS

def _faixa(r):
    a, p = r["apogee"], r["perigee"]
    if pd.isna(a) or pd.isna(p):
        return "Indef."
    alt = (a + p) / 2
    if (a - p) > 2000 and a > p * 2:
        return "HEO"
    if alt < 2000:
        return "LEO"
    if 35000 <= alt <= 36500:
        return "GEO"
    if 2000 <= alt < 35000:
        return "MEO"
    return "HEO"


def preparar(df):
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["period", "inclination", "apogee", "perigee", "rcs"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["decay_date"] = pd.to_datetime(df["decay_date"], errors="coerce")

    hoje = pd.Timestamp.today()
    df["age_years"] = (hoje - df["launch_date"]).dt.days / 365.25
    df["on_orbit"] = df["decay_date"].isna()

    tipo = {"PAY": "Satélite", "R/B": "Corpo de foguete",
            "DEB": "Detrito", "UNK": "Desconhecido"}
    df["tipo"] = df["object_type"].map(tipo).fillna("Outro")

    ativos = {"+", "P", "B", "S", "X"}
    df["ativo"] = df["ops_status_code"].isin(ativos)

    df["altitude"] = (df["apogee"] + df["perigee"]) / 2
    re_terra = 6378.137
    df["vel_kms"] = np.where(
        df["period"] > 0,
        2 * np.pi * (re_terra + df["altitude"]) / (df["period"] * 60),
        np.nan,
    )
    df["faixa"] = df.apply(_faixa, axis=1)
    return df



# 3. ESTATÍSTICA DESCRITIVA

def descritiva(s, nome):
    s = s.dropna()
    moda = s.round(1).mode()
    return pd.Series({
        "n": int(s.size),
        "média": s.mean(), "mediana": s.median(),
        "moda": moda.iloc[0] if len(moda) else np.nan,
        "mínimo": s.min(), "máximo": s.max(), "amplitude": s.max() - s.min(),
        "variância": s.var(), "desv.padrão": s.std(),
        "CV (%)": s.std() / s.mean() * 100 if s.mean() else np.nan,
        "Q1": s.quantile(.25), "Q2 (mediana)": s.quantile(.5), "Q3": s.quantile(.75),
        "P10": s.quantile(.1), "P90": s.quantile(.9),
        "assimetria": s.skew(),
    }, name=nome)


def tabela_descritiva(orb):
    return pd.concat(
        [descritiva(orb[v], n) for n, v in VARIAVEIS.items()], axis=1
    ).round(2)



# 4. DETECÇÃO DE ANOMALIAS (OUTLIERS)

def outliers_iqr(s):
    s = s.dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return lo, hi, s[(s < lo) | (s > hi)]


def outliers_z(s, k=3):
    s = s.dropna()
    z = np.abs(stats.zscore(s))
    return s[z > k]


def relatorio_outliers(orb):
    print("\n=== ANOMALIAS (IQR 1,5x  e  Z-Score |z|>3) ===")
    for nome, v in VARIAVEIS.items():
        lo, hi, o_iqr = outliers_iqr(orb[v])
        o_z = outliers_z(orb[v])
        pct = len(o_iqr) / orb[v].notna().sum() * 100
        print(f"{nome:18s} | IQR: {len(o_iqr):5d} ({pct:4.1f}%)  "
              f"limites [{lo:.1f}, {hi:.1f}]  | Z-Score: {len(o_z)}")

    lo, hi, o = outliers_iqr(orb["altitude"])
    anom = orb.loc[o.index, ["object_name", "tipo", "faixa",
                             "apogee", "perigee", "altitude", "period"]]
    print(f"\nObjetos com altitude anômala (IQR): {len(anom)}")
    print(anom.sort_values("altitude", ascending=False).head(15).to_string(index=False))



# 5. VISUALIZAÇÕES

def graficos(orb):
    # Histograma
    plt.figure()
    sns.histplot(orb[orb["altitude"] < 45000]["altitude"], bins=60, color="#2c7fb8")
    plt.axvline(orb["altitude"].median(), color="red", ls="--", label="mediana")
    plt.title("Distribuição da altitude dos objetos em órbita")
    plt.xlabel("Altitude (km)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(SAIDA, "01_hist_altitude.png"), dpi=130); plt.close()

    # Boxplot
    ordem = ["LEO", "MEO", "GEO", "HEO"]
    plt.figure()
    sns.boxplot(data=orb[orb["faixa"].isin(ordem)], x="faixa", y="altitude",
                order=ordem, palette="viridis")
    plt.yscale("log"); plt.title("Altitude por faixa orbital (escala log)")
    plt.tight_layout(); plt.savefig(os.path.join(SAIDA, "02_boxplot_faixa.png"), dpi=130); plt.close()

    # Barras
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    orb["tipo"].value_counts().plot.bar(ax=ax[0], color="#d95f0e")
    ax[0].set_title("Objetos por tipo"); ax[0].tick_params(axis="x", rotation=20)
    orb["faixa"].value_counts().reindex(["LEO", "MEO", "GEO", "HEO", "Indef."]).dropna().plot.bar(
        ax=ax[1], color="#2c7fb8")
    ax[1].set_title("Objetos por faixa orbital")
    plt.tight_layout(); plt.savefig(os.path.join(SAIDA, "03_barras_tipo_faixa.png"), dpi=130); plt.close()

    # Dispersão apogeu x perigeu
    plt.figure()
    m = orb["faixa"].isin(ordem)
    sns.scatterplot(data=orb[m], x="perigee", y="apogee", hue="faixa", s=10, alpha=.4)
    plt.xscale("log"); plt.yscale("log"); plt.title("Apogeu × Perigeu")
    plt.tight_layout(); plt.savefig(os.path.join(SAIDA, "04_scatter_apogeu_perigeu.png"), dpi=130); plt.close()

    # Heatmap de correlação
    num = orb[["altitude", "perigee", "apogee", "inclination", "period",
               "vel_kms", "age_years"]].rename(columns={
        "altitude": "Alt", "perigee": "Perigeu", "apogee": "Apogeu",
        "inclination": "Incl", "period": "Período", "vel_kms": "Veloc", "age_years": "Idade"})
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Matriz de correlação")
    plt.tight_layout(); plt.savefig(os.path.join(SAIDA, "05_heatmap_correlacao.png"), dpi=130); plt.close()

    print(f"\n>> Gráficos salvos em: {SAIDA}")
    return num



# 6. ÍNDICE DE INSTABILIDADE ORBITAL

def indice_instabilidade(orb):
    total = len(orb)
    std_global = orb["altitude"].std()
    res = {}
    for f in ["LEO", "MEO", "GEO", "HEO"]:
        sub = orb[orb["faixa"] == f]
        if len(sub) < 5:
            continue
        conc = len(sub) / total
        inativos = (~sub["ativo"]).mean()
        _, _, o = outliers_iqr(sub["altitude"])
        out = len(o) / max(sub["altitude"].notna().sum(), 1)
        disp = min(sub["altitude"].std() / std_global, 1) if std_global else 0
        score = (0.40 * conc + 0.25 * inativos + 0.25 * out + 0.10 * disp) * 100
        res[f] = dict(score=round(score, 1), n=len(sub),
                      concentracao=round(conc * 100, 1),
                      inativos=round(inativos * 100, 1),
                      outliers=round(out * 100, 1),
                      dispersao=round(disp, 2))
    return pd.DataFrame(res).T.sort_values("score", ascending=False)



# 7. PERGUNTAS DE NEGÓCIO

def perguntas_de_negocio(orb, num, idx):
    print("\n================ PERGUNTAS DE NEGÓCIO ================")

    print("\nP1 - Concentração por faixa orbital:")
    print((orb["faixa"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

    print("\nP2 - Objetos fora do padrão:")
    for v in ["altitude", "vel_kms"]:
        lo, hi, o = outliers_iqr(orb[v])
        print(f"   {v}: {len(o)} outliers (IQR)  limites [{lo:.1f}, {hi:.1f}]")

    print("\nP3 - Equilíbrio da distribuição:")
    s = orb["altitude"]
    print(f"   Altitude  média={s.mean():.0f}  mediana={s.median():.0f}  assimetria={s.skew():.2f}")
    print("   média >> mediana e assimetria positiva => distribuição concentrada/assimétrica.")

    print("\nP4 - Composição do ambiente orbital:")
    print(orb["tipo"].value_counts().to_string())
    pay = orb[orb["tipo"] == "Satélite"]
    print(f"   {(~pay['ativo']).mean() * 100:.1f}% dos satélites não estão operacionais")

    print("\nP5 - Correlações (Pearson):")
    print(num.corr()[["Período", "Veloc"]].round(2).to_string())

    print("\nP6 - Faixa de maior instabilidade:")
    print(f"   {idx.index[0]}  | índice = {idx.iloc[0]['score']}")
    print("   Concentração + inativos + anomalias => pressão sobre as órbitas mais usadas.")



# 8. DASHBOARD INTERATIVO (Plotly -> HTML)

def dashboard(orb):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sub = orb[orb["faixa"].isin(["LEO", "MEO", "GEO", "HEO"])]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Objetos por faixa", "Objetos por tipo",
                        "Altitude (hist)", "Apogeu x Perigeu"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "scatter"}]],
    )
    vc = sub["faixa"].value_counts()
    fig.add_bar(x=vc.index, y=vc.values, marker_color="#2c7fb8", row=1, col=1)
    tc = orb["tipo"].value_counts()
    fig.add_bar(x=tc.index, y=tc.values, marker_color="#d95f0e", row=1, col=2)
    fig.add_histogram(x=sub[sub["altitude"] < 45000]["altitude"], nbinsx=60,
                      marker_color="#41b6c4", row=2, col=1)
    for f in ["LEO", "MEO", "GEO", "HEO"]:
        s = sub[sub["faixa"] == f]
        fig.add_scatter(x=s["perigee"], y=s["apogee"], mode="markers",
                        name=f, marker=dict(size=4, opacity=.4), row=2, col=2)
    fig.update_xaxes(type="log", row=2, col=2)
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_layout(height=720, template="plotly_dark", showlegend=True,
                      title_text="Painel - Sustentabilidade da Economia Orbital (CelesTrak)")
    destino = os.path.join(SAIDA, "dashboard_orbital.html")
    fig.write_html(destino)
    print(f"\n>> Dashboard salvo em: {destino}")



# MAIN

def main():
    df = coletar()
    df = preparar(df)

    orb = df[df["on_orbit"] & df["apogee"].notna() &
             df["perigee"].notna() & (df["period"] > 0)].copy()
    print(f"\n>> Objetos em órbita com elementos válidos: {len(orb)}")
    print(orb["tipo"].value_counts().to_string())
    print("\nAtivos x inativos:")
    print(orb["ativo"].value_counts().to_string())

    print("\n=== ESTATÍSTICA DESCRITIVA ===")
    print(tabela_descritiva(orb).to_string())

    relatorio_outliers(orb)

    num = graficos(orb)

    print("\n=== ÍNDICE DE INSTABILIDADE ORBITAL ===")
    idx = indice_instabilidade(orb)
    print(idx.to_string())

    perguntas_de_negocio(orb, num, idx)

    dashboard(orb)

    print("\n>> Concluído. Veja a pasta ./saida/ para gráficos e dashboard.")


if __name__ == "__main__":
    main()