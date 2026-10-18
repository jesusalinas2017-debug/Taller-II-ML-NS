# =============================================================================
# app.py — Dashboard del Certamen 2 (Grupo 2)
# "¿Qué ingrediente le falta a tu receta?" — Reglas de asociación sobre Food.com
#
# El dashboard NO mina reglas: lee los CSV que exporta el notebook
# (sección 9) desde la carpeta data/. Así la app es rápida y estable.
#
# Ejecutar en local:   streamlit run app.py
# =============================================================================

import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from plotly.colors import sample_colorscale
from pyvis.network import Network

st.set_page_config(page_title="¿Qué le falta a tu receta?", layout="wide")

# Carpeta de los CSV: data/ (estructura del repositorio). Si los CSV quedaron sueltos junto a app.py,
# también los encuentra, así la app no se cae por cómo se subieron los archivos a GitHub.
BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(BASE, "data") if os.path.isdir(os.path.join(BASE, "data")) else BASE
AZUL = "#0b1f45"
GRIS = "#52607a"
ROJO = "#c0392b"

# Canastas de ejemplo (las mismas del notebook, sección 6.1)
EJEMPLOS = {
    "— elegir —": None,
    "Pan casero": ["flour", "yeast"],
    "Salteado asiático": ["soy sauce", "ginger"],
    "Pasta al pesto": ["basil", "pine nut"],
    "Carne con cebolla": ["ground beef", "onion"],
    "Cóctel (arranque en frío)": ["rum", "ice"],
}


# -----------------------------------------------------------------------------
# 1. Carga de datos
# -----------------------------------------------------------------------------
def a_conjunto(texto):
    # "flour, yeast" → frozenset({"flour", "yeast"})  (así se guardaron los itemsets en el CSV)
    return frozenset(texto.split(", "))


def texto_itemset(itemset):
    # frozenset → "{flour, yeast}" (igual que en el notebook)
    return "{" + ", ".join(sorted(itemset)) + "}"


@st.cache_data
def cargar_datos():
    # Se lee una sola vez y queda en caché mientras la app esté activa
    resumen = pd.read_csv(f"{CARPETA}/resumen.csv").set_index("indicador")["valor"]
    populares = pd.read_csv(f"{CARPETA}/populares.csv")
    hitrate = pd.read_csv(f"{CARPETA}/hitrate.csv")
    cobertura = pd.read_csv(f"{CARPETA}/cobertura.csv")

    # Reglas de la Parte 1 (minsop = 0,01; minconf = 0,5) con marcas de Fisher + Holm y redundancia
    reglas = pd.read_csv(f"{CARPETA}/reglas.csv")
    reglas["antecedents"] = reglas["antecedente"].apply(a_conjunto)
    reglas["consequents"] = reglas["consecuente"].apply(a_conjunto)
    reglas["regla"] = reglas["antecedents"].apply(texto_itemset) + " ⇒ " + reglas["consequents"].apply(texto_itemset)

    # Reglas del recomendador (un solo consecuente; minsop = 0,005; minconf = 0,3; lift > 1)
    reglas_rec = pd.read_csv(f"{CARPETA}/reglas_recomendador.csv")
    reglas_rec["antecedents"] = reglas_rec["antecedente"].apply(a_conjunto)
    reglas_rec["consequents"] = reglas_rec["producto"].apply(lambda p: frozenset([p]))
    reglas_rec = reglas_rec.rename(columns={"confianza": "confidence", "soporte": "support"})

    return resumen, populares, reglas, reglas_rec, hitrate, cobertura


resumen, populares, reglas, reglas_rec, hitrate, cobertura = cargar_datos()
mas_usados = populares["ingrediente"].tolist()                      # línea base (ordenada por soporte)
soporte_item = dict(zip(populares["ingrediente"], populares["soporte"]))


def miles(x):
    # 28878 → "28.878" (formato chileno)
    return f"{int(x):,}".replace(",", ".")


def color_lift(valor, minimo, maximo):
    # Tono de azul según el lift: más oscuro = asociación más fuerte (igual idea que en el notebook)
    proporcion = (valor - minimo) / (maximo - minimo) if maximo > minimo else 1
    return sample_colorscale("Blues", [0.35 + 0.65 * proporcion])[0]


# -----------------------------------------------------------------------------
# 2. Recomendador (mismo código que el notebook, sección 6)
# -----------------------------------------------------------------------------
def recomendar(canasta, reglas, populares, top=3, orden="lift"):
    canasta = set(canasta)

    # 1) Reglas activas: el antecedente está dentro de la canasta y el producto aún no está
    activa = np.array([antecedente <= canasta for antecedente in reglas["antecedents"]], dtype=bool)
    nueva = ~reglas["producto"].isin(canasta).values
    candidatas = reglas[activa & nueva]

    # 2) Ordenamos por el criterio elegido y dejamos una sola regla por producto (la mejor)
    desempate = "confidence" if orden == "lift" else "lift"
    candidatas = candidatas.sort_values([orden, desempate], ascending=False)
    candidatas = candidatas.drop_duplicates("producto").head(top)

    sugerencias = []
    for _, r in candidatas.iterrows():
        sugerencias.append({
            "producto": r["producto"],
            "motivo": texto_itemset(r["antecedents"]) + " ⇒ " + texto_itemset(r["consequents"]),
            "confianza": r["confidence"],
            "lift": r["lift"],
        })

    # 3) Arranque en frío o faltan sugerencias: completamos con los más usados que no estén en la canasta
    ya_sugeridos = {s["producto"] for s in sugerencias}
    for producto in populares:
        if len(sugerencias) >= top:
            break
        if producto not in canasta and producto not in ya_sugeridos:
            sugerencias.append({"producto": producto, "motivo": "más usado (respaldo)",
                                "confianza": np.nan, "lift": np.nan})

    return pd.DataFrame(sugerencias, columns=["producto", "motivo", "confianza", "lift"])


def recomendar_hibrido(canasta, reglas, populares, top=3, lift_minimo=2.0):
    # Híbrido conmutado: reglas (por lift) solo si alguna regla activa supera el lift mínimo
    canasta_set = set(canasta)
    activa = np.array([antecedente <= canasta_set for antecedente in reglas["antecedents"]], dtype=bool)
    nueva = ~reglas["producto"].isin(canasta_set).values
    lift_activas = reglas.loc[activa & nueva, "lift"]

    if len(lift_activas) > 0 and lift_activas.max() > lift_minimo:
        return recomendar(canasta, reglas, populares, top=top, orden="lift")
    sugerencias = [p for p in populares if p not in canasta_set][:top]
    return pd.DataFrame({"producto": sugerencias,
                         "motivo": f"más usado (híbrido: sin regla con lift > {lift_minimo})",
                         "confianza": np.nan, "lift": np.nan})


# -----------------------------------------------------------------------------
# 3. Grafos con pyvis (mismo diseño que el notebook, sección 5.4)
# -----------------------------------------------------------------------------
def grafo_reglas(tabla, alto=600):
    # Cuadros = ingredientes; círculos = reglas (tamaño = soporte, color = lift)
    red = Network(height=f"{alto}px", width="100%", directed=True, cdn_resources="remote")
    lift_min, lift_max = tabla["lift"].min(), tabla["lift"].max()
    soporte_max = tabla["soporte"].max()

    for i, r in tabla.reset_index(drop=True).iterrows():
        nombre = f"regla {i + 1}"
        detalle = (f"{r['regla']}\nsoporte: {r['soporte']:.3f}\n"
                   f"confianza: {r['confianza']:.3f}\nlift: {r['lift']:.2f}")
        red.add_node(nombre, label=" ", title=detalle, shape="dot",
                     size=10 + 25 * r["soporte"] / soporte_max,
                     color=color_lift(r["lift"], lift_min, lift_max))
        for item in r["antecedents"] | r["consequents"]:
            red.add_node(item, label=item, shape="box", color="#e6e9ef",
                         font={"color": AZUL, "size": 18}, title=f"soporte: {soporte_item.get(item, 0):.3f}")
        for item in r["antecedents"]:
            red.add_edge(item, nombre, color=GRIS)
        for item in r["consequents"]:
            red.add_edge(nombre, item, color=GRIS)

    red.repulsion(node_distance=140, spring_length=120)
    return red.generate_html()


def mapa_flujo(tabla, alto=600):
    # Solo reglas simples {A} ⇒ {B}: grosor = confianza, color = lift, tamaño del nodo = soporte
    red = Network(height=f"{alto}px", width="100%", directed=True, cdn_resources="remote")
    lift_min, lift_max = tabla["lift"].min(), tabla["lift"].max()

    for item in set(tabla["antecedente"]) | set(tabla["consecuente"]):
        red.add_node(item, label=item, shape="dot", color=AZUL, size=10 + 80 * soporte_item.get(item, 0),
                     font={"size": 22, "color": AZUL, "strokeWidth": 5, "strokeColor": "white"},
                     title=f"{item}\nsoporte: {soporte_item.get(item, 0):.3f}")
    for _, r in tabla.iterrows():
        red.add_edge(r["antecedente"], r["consecuente"], width=1 + 8 * r["confianza"],
                     color=color_lift(r["lift"], lift_min, lift_max),
                     title=f"{r['regla']}\nconfianza: {r['confianza']:.3f}\nlift: {r['lift']:.2f}")

    red.barnes_hut(gravity=-9000, spring_length=160)
    return red.generate_html()


# -----------------------------------------------------------------------------
# 4. Barra lateral: filtros de reglas (aplican al Explorador y a las Visualizaciones)
# -----------------------------------------------------------------------------
st.sidebar.title("¿Qué le falta a tu receta?")
st.sidebar.caption("Análisis de canasta sobre recetas de Food.com · Grupo 2")
st.sidebar.header("Filtros de reglas")
st.sidebar.caption("Aplican a las pestañas **Explorador** y **Visualizaciones**. Las reglas se minaron "
                   "con soporte ≥ 0,01 y confianza ≥ 0,5, por eso los controles parten ahí.")

min_soporte = st.sidebar.slider("Soporte mínimo", 0.010, float(round(reglas["soporte"].max(), 3)),
                                0.010, step=0.001, format="%.3f")
min_confianza = st.sidebar.slider("Confianza mínima", 0.50, 1.00, 0.50, step=0.01)
min_lift = st.sidebar.slider("Lift mínimo", 0.9, float(np.ceil(reglas["lift"].max())), 0.9, step=0.1)
solo_signif = st.sidebar.checkbox("Solo significativas (Fisher + Holm)", value=True)
sin_redund = st.sidebar.checkbox("Excluir redundantes", value=True)

items_en_reglas = sorted(set().union(*reglas["antecedents"]) | set().union(*reglas["consequents"]))
item_filtro = st.sidebar.selectbox("Filtrar por ingrediente", ["(todos)"] + items_en_reglas)
donde = st.sidebar.radio("¿Dónde debe aparecer?", ["Antecedente", "Consecuente", "Cualquiera"], horizontal=True)

# Aplicamos los filtros
filtro = ((reglas["soporte"] >= min_soporte) & (reglas["confianza"] >= min_confianza) &
          (reglas["lift"] >= min_lift))
if solo_signif:
    filtro &= reglas["significativa"]
if sin_redund:
    filtro &= ~reglas["redundante"]
if item_filtro != "(todos)":
    en_ant = reglas["antecedents"].apply(lambda x: item_filtro in x)
    en_con = reglas["consequents"].apply(lambda x: item_filtro in x)
    filtro &= {"Antecedente": en_ant, "Consecuente": en_con, "Cualquiera": en_ant | en_con}[donde]

filtradas = reglas[filtro].sort_values("lift", ascending=False).reset_index(drop=True)
st.sidebar.metric("Reglas que cumplen los filtros", f"{len(filtradas)} de {len(reglas)}")


# -----------------------------------------------------------------------------
# 5. Pestañas
# -----------------------------------------------------------------------------
st.title("¿Qué ingrediente le falta a tu receta?")
st.markdown("Reglas de asociación y recomendador *“quienes usan estos ingredientes también usan…”* "
            "sobre recetas de **Food.com**. **Transacción** = una receta · **Ítem** = un ingrediente normalizado.")

tab_resumen, tab_explorador, tab_visual, tab_rec, tab_eval = st.tabs(
    ["Resumen", "Explorador de reglas", "Visualizaciones", "Recomendador", "Evaluación"])

# ---------------------------- 5.1 Resumen de datos ---------------------------
with tab_resumen:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Recetas originales", miles(resumen["recetas_originales"]))
    c2.metric("Transacciones (muestra)", miles(resumen["transacciones"]))
    c3.metric("Ingredientes", miles(resumen["items"]))
    c4.metric("Densidad", f"{resumen['densidad']:.1%}".replace(".", ","))
    c5.metric("Ingredientes por receta", f"{resumen['items_por_receta']:.1f}".replace(".", ","))
    c6.metric("Reglas finales", miles(resumen["reglas_finales"]))

    st.caption("Muestra aleatoria de 30.000 recetas (semilla 42) e ingredientes con soporte ≥ 0,5 %. "
               "Densidad = proporción de unos en la matriz receta × ingrediente.")

    izq, der = st.columns([3, 2])
    with izq:
        top_n = st.slider("¿Cuántos ingredientes mostrar?", 10, 40, 20, step=5)
        top = populares.head(top_n).sort_values("soporte")
        fig = px.bar(top, x="soporte", y="ingrediente", orientation="h", hover_data={"recetas": True},
                     color_discrete_sequence=[AZUL], height=28 * top_n + 80,
                     title=f"Los {top_n} ingredientes más frecuentes")
        fig.update_layout(yaxis_title=None, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with der:
        st.subheader("De 880 reglas a 785")
        embudo = pd.DataFrame({
            "etapa": ["Minadas (sop ≥ 0,01; conf ≥ 0,5)", "Significativas (Fisher + Holm)",
                      "Significativas y no redundantes"],
            "reglas": [resumen["reglas_minadas"], resumen["reglas_significativas"], resumen["reglas_finales"]],
        })
        fig = px.funnel(embudo, x="reglas", y="etapa", color_discrete_sequence=[AZUL], height=300)
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("- **Cola larga:** `salt` está en el 53 % de las recetas; la mayoría de los ingredientes, en menos del 2 %.\n"
                    "- Las 32 reglas que no pasan Fisher + Holm son todas `X ⇒ {salt}` con lift ≈ 1.\n"
                    "- Las asociaciones más fuertes son de repostería y condimentos: `{onion powder} ⇒ {garlic powder}` (lift 13,6).")

# ---------------------------- 5.2 Explorador de reglas -----------------------
with tab_explorador:
    st.subheader(f"{len(filtradas)} reglas con los filtros actuales")
    st.caption("Mueve los filtros de la barra lateral. Haz clic en el encabezado de una columna para ordenar.")

    orden_tabla = st.selectbox("Ordenar por", ["lift", "confianza", "soporte", "leverage", "conviccion"])
    columnas = ["regla", "soporte", "confianza", "lift", "leverage", "conviccion",
                "p_ajustado", "significativa", "redundante"]
    tabla = filtradas[columnas].sort_values(orden_tabla, ascending=False).reset_index(drop=True)

    st.dataframe(
        tabla, hide_index=True, use_container_width=True, height=520,
        column_config={
            "regla": st.column_config.TextColumn("regla", width="large"),
            "soporte": st.column_config.NumberColumn(format="%.4f"),
            "confianza": st.column_config.NumberColumn(format="%.3f"),
            "lift": st.column_config.NumberColumn(format="%.2f"),
            "leverage": st.column_config.NumberColumn(format="%.4f"),
            "conviccion": st.column_config.NumberColumn("convicción", format="%.2f"),
            "p_ajustado": st.column_config.NumberColumn("p ajustado (Holm)", format="%.1e"),
        },
    )
    st.download_button("Descargar estas reglas (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                       file_name="reglas_filtradas.csv", mime="text/csv")

# ---------------------------- 5.3 Visualizaciones ----------------------------
with tab_visual:
    if filtradas.empty:
        st.warning("Ninguna regla cumple los filtros. Relaja los filtros de la barra lateral.")
    else:
        fig = px.scatter(filtradas, x="soporte", y="confianza", color="lift", hover_name="regla",
                         hover_data={"soporte": ":.4f", "confianza": ":.3f", "lift": ":.2f"},
                         color_continuous_scale="Blues", height=480,
                         title=f"Soporte vs. confianza de {len(filtradas)} reglas (color = lift)")
        fig.update_traces(marker=dict(size=8, line=dict(width=0.3, color="white")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Las reglas de mayor lift (azul oscuro) tienen soporte bajo: las asociaciones fuertes son de nicho.")

        st.divider()
        tipo = st.radio("Tipo de grafo", ["Grafo de reglas", "Mapa de flujo ingrediente → ingrediente"],
                        horizontal=True)
        k = st.slider("Reglas a mostrar (las de mayor lift)", 5, 60, 30, step=5)

        if tipo == "Grafo de reglas":
            seleccion = filtradas.head(k)
            st.caption("Cuadros: ingredientes · círculos: reglas (tamaño = soporte, color = lift). "
                       "Pasa el mouse sobre un círculo para ver la regla.")
            components.html(grafo_reglas(seleccion), height=620)
        else:
            simples = filtradas[(filtradas["antecedents"].apply(len) == 1) &
                                (filtradas["consequents"].apply(len) == 1)].head(k)
            if simples.empty:
                st.info("Con estos filtros no hay reglas simples {A} ⇒ {B}.")
            else:
                st.caption(f"{len(simples)} reglas simples. Grosor = confianza · color = lift · "
                           "tamaño del nodo = soporte.")
                components.html(mapa_flujo(simples), height=620)

# ---------------------------- 5.4 Recomendador -------------------------------
with tab_rec:
    st.subheader("Arma tu canasta y recibe sugerencias con su motivo")

    # Canasta inicial y botón para cargar ejemplos
    if "canasta" not in st.session_state:
        st.session_state["canasta"] = ["flour", "yeast"]

    def cargar_ejemplo():
        elegido = EJEMPLOS[st.session_state["ejemplo"]]
        if elegido is not None:
            st.session_state["canasta"] = elegido

    st.selectbox("Cargar una canasta de ejemplo", list(EJEMPLOS), key="ejemplo", on_change=cargar_ejemplo)
    canasta = st.multiselect("Ingredientes que ya tienes (selección múltiple)", sorted(mas_usados), key="canasta")

    c1, c2, c3 = st.columns(3)
    top_n_rec = c1.slider("N° de sugerencias (N)", 1, 10, 5)
    criterio = c2.radio("Criterio", ["Reglas por lift", "Reglas por confianza", "Híbrido conmutado"])
    lift_minimo = c3.slider("Lift mínimo del híbrido", 1.0, 5.0, 2.0, step=0.5,
                            disabled=(criterio != "Híbrido conmutado"))

    if len(canasta) == 0:
        st.info("Agrega al menos un ingrediente. Sin canasta, el sistema solo puede sugerir los más usados.")

    if criterio == "Reglas por lift":
        sugerencias = recomendar(canasta, reglas_rec, mas_usados, top=top_n_rec, orden="lift")
    elif criterio == "Reglas por confianza":
        sugerencias = recomendar(canasta, reglas_rec, mas_usados, top=top_n_rec, orden="confidence")
    else:
        sugerencias = recomendar_hibrido(canasta, reglas_rec, mas_usados, top=top_n_rec, lift_minimo=lift_minimo)

    # ¿Cuántas reglas se activan con esta canasta?
    activa = np.array([a <= set(canasta) for a in reglas_rec["antecedents"]], dtype=bool)
    activas = reglas_rec[activa & ~reglas_rec["producto"].isin(canasta).values]
    sugerencias["origen"] = np.where(sugerencias["motivo"].str.startswith("más usado"),
                                     "respaldo: más usados", "regla")

    st.dataframe(sugerencias.rename(columns={"motivo": "motivo (regla)"}), hide_index=True,
                 use_container_width=True,
                 column_config={"confianza": st.column_config.NumberColumn(format="%.3f"),
                                "lift": st.column_config.NumberColumn(format="%.2f")})

    n_respaldo = (sugerencias["origen"] != "regla").sum()
    if len(activas) == 0 and len(canasta) > 0:
        st.warning(f"**Arranque en frío:** ninguna regla se activa con esta canasta. "
                   f"Las {n_respaldo} sugerencias vienen del respaldo de los más usados.")
    elif criterio == "Híbrido conmutado" and n_respaldo == len(sugerencias) and len(canasta) > 0:
        st.warning(f"**Híbrido:** se activan {len(activas)} reglas, pero ninguna supera lift {lift_minimo}. "
                   "Se usan los más usados en vez de una asociación débil.")
    elif n_respaldo > 0:
        st.info(f"Se activan {len(activas)} reglas, pero solo alcanzan para {len(sugerencias) - n_respaldo} "
                f"productos distintos. Las otras {n_respaldo} posiciones se completan con los más usados.")
    else:
        st.success(f"Se activan {len(activas)} reglas. Todas las sugerencias vienen de una regla.")

    st.caption(f"Reglas de un solo consecuente minadas con soporte ≥ 0,005, confianza ≥ 0,3 y lift > 1 "
               f"({miles(len(reglas_rec))} reglas). Los ingredientes de la canasta nunca se recomiendan.")

# ---------------------------- 5.5 Evaluación ---------------------------------
with tab_eval:
    st.subheader("HitRate@N: ¿el ingrediente oculto aparece entre las N sugerencias?")
    st.caption("Partición 80/20 con semilla 42. Reglas minadas solo con entrenamiento (23.102 recetas). "
               "A cada una de las 5.776 recetas de prueba se le oculta un ingrediente al azar (leave-one-out).")

    izq, der = st.columns([3, 2])
    with izq:
        largo = hitrate.melt(id_vars="N", var_name="sistema", value_name="HitRate")
        fig = px.line(largo, x="N", y="HitRate", color="sistema", markers=True, height=420,
                      color_discrete_map={"Reglas (lift)": AZUL, "Reglas (confianza)": GRIS,
                                          "Híbrido": "#2e86c1", "Más usados": ROJO})
        fig.update_layout(xaxis=dict(tickmode="array", tickvals=hitrate["N"].tolist()),
                          yaxis_tickformat=".0%", legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with der:
        tabla_hr = hitrate.set_index("N").map(lambda v: f"{v:.1%}".replace(".", ","))
        st.dataframe(tabla_hr, use_container_width=True)

    st.subheader("Reglas activas, cobertura y novedad")
    tabla_cob = cobertura.copy()
    for col in tabla_cob.columns[1:3]:
        tabla_cob[col] = tabla_cob[col].map(lambda v: f"{v:.1%}".replace(".", ","))
    tabla_cob[tabla_cob.columns[3]] = tabla_cob[tabla_cob.columns[3]].map(lambda v: f"{v:.3f}".replace(".", ","))
    st.dataframe(tabla_cob, hide_index=True, use_container_width=True)

    st.markdown(
        "- **Las reglas superan a la línea base:** por confianza, HitRate@1 = 17,8 % vs. 11,3 % de los más usados.\n"
        "- **Precisión vs. novedad:** la confianza acierta más porque sugiere ingredientes populares "
        "(soporte medio del top 3 = 0,28); el lift sugiere ingredientes menos obvios (0,16) y acierta menos en N = 1.\n"
        "- **Cobertura:** los sistemas por reglas solo llegan a recomendar el 18 % del catálogo "
        "(40 de 222 ingredientes) en su top 10; los más usados, el 9 %.")

st.divider()
st.caption("Datos: *Food.com Recipes and User Interactions* (Kaggle, shuyangli94; Majumder et al., 2019). "
           "Grupo 2 — Lucas Torres Baeza · María Jesús Salinas · Aprendizaje no supervisado, UDD.")
