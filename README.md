# ¿Qué ingrediente le falta a tu receta?

**Certamen 2 — Reglas de asociación y sistemas de recomendación**
Aprendizaje no supervisado · UDD · Docente: Danilo Gómez Correa
**Grupo 2:** Lucas Torres Baeza · María Jesús Salinas

**Dashboard publicado:** https://taller-ii-ml-ns.streamlit.app

---

## Problema

Una app de recetas o un supermercado en línea quiere ayudar a quien arma una receta o su lista de compras:
*"quienes usan estos ingredientes también usan…"*. El recomendador sugiere el ingrediente que probablemente
falta (por ejemplo, *"¿te falta la levadura?"*) y muestra la **regla** que justifica cada sugerencia.

## Datos

| | |
|---|---|
| **Base** | [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) (Kaggle, `shuyangli94`) |
| **Fuente** | Majumder, B. P., Li, S., Ni, J. y McAuley, J. (2019). *Generating Personalized Recipes from Historical User Preferences*. EMNLP. |
| **Licencia** | Publicada en Kaggle por los autores; uso académico citando la fuente. Los datos crudos **no** se suben al repositorio. |
| **Transacción** | Una **receta** |
| **Ítem** | Un **ingrediente** normalizado |
| **Tamaño final** | 28.878 recetas × 222 ingredientes · 7,2 ingredientes por receta · densidad 3,2 % |

## Resultados principales

- **880 reglas** (soporte ≥ 0,01; confianza ≥ 0,5) → **848** significativas (Fisher + Holm, α = 0,01) → **785** no redundantes.
- Regla más fuerte: `{onion powder} ⇒ {garlic powder}` (confianza 0,65; lift 13,6).
- Las 32 reglas descartadas por Fisher son todas `X ⇒ {salt}` con lift ≈ 1: la sal está en el 53 % de las recetas.
- **HitRate@N** (leave-one-out, 20 % de prueba): las reglas por confianza aciertan 17,8 % con N = 1 frente a 11,3 % de los más usados, y 46,1 % vs. 41,6 % con N = 10.

## Estructura del repositorio

```
certamen2-grupo-2/
├── README.md                ← este archivo
├── requirements.txt         ← dependencias
├── app.py                   ← dashboard (Streamlit)
├── .streamlit/config.toml   ← colores del dashboard
├── ficha_base.md            ← ficha de la base (entregable 1)
├── data/                    ← CSV que exporta el notebook y lee el dashboard
│   ├── resumen.csv               indicadores de la base
│   ├── populares.csv             soporte de cada ingrediente (línea base)
│   ├── reglas.csv                reglas de la Parte 1 (con p ajustado y redundancia)
│   ├── reglas_recomendador.csv   reglas de un consecuente para el recomendador
│   ├── hitrate.csv               HitRate@1, 3, 5 y 10 por sistema
│   └── cobertura.csv             reglas activas, cobertura y popularidad
└── Certamen2_Grupo_2.ipynb  ← notebook completo (Colab)
```

## Cómo ejecutar

### Notebook (Google Colab)

1. Abrir `Certamen2_Grupo_2.ipynb` en Colab.
2. Los datos se descargan solos con `kagglehub`. Si falla, descargar `archive.zip` desde Kaggle y subirlo al panel *Archivos*.
3. *Entorno de ejecución → Ejecutar todas*. La última celda descarga `certamen2-grupo-2.zip` con **todo el
   repositorio** listo para GitHub: este README, `requirements.txt`, `app.py`, los CSV de `data/` y una copia del
   notebook ejecutado.

### Dashboard en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Publicar en Streamlit Community Cloud

1. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
2. *Create app* → elegir este repositorio, rama `main` y archivo `app.py`.
3. En *Advanced settings* elegir **Python 3.12** (con 3.14 la instalación de pandas se queda pegada).
4. Copiar el link público en este README y en la entrega de Canvas.

## Módulos del dashboard

| Pestaña | Contenido |
|---|---|
| Resumen | Transacciones, ingredientes, densidad, ingredientes más frecuentes y embudo de reglas |
| Explorador de reglas | Filtros de soporte, confianza y lift; Fisher y redundancia; filtro por ingrediente en antecedente o consecuente; tabla ordenable y descarga CSV |
| Visualizaciones | Dispersión soporte–confianza coloreada por lift, grafo de reglas y mapa de flujo (pyvis) |
| Recomendador | Canasta con selección múltiple; N sugerencias con regla, confianza y lift; lift, confianza o híbrido; avisa cuando usa el respaldo de populares |
| Evaluación | Tabla y gráfico de HitRate@N frente a la línea base, cobertura y precisión vs. novedad |

## Uso de inteligencia artificial

Se usó IA (Claude) como apoyo para revisar el código, construir el dashboard y redactar este README.
El código de la guía de clase se adaptó a la base, y todas las decisiones están explicadas en el notebook.
