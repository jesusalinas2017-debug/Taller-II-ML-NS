# Ficha de la base — Grupo 2

| | |
|---|---|
| **Nombre** | Food.com Recipes and User Interactions |
| **Enlace** | https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions |
| **Origen** | Recetas publicadas en Food.com entre 2000 y 2018 (Majumder et al., 2019, EMNLP) |
| **Archivos usados** | `RAW_recipes.csv` (231.637 recetas) e `ingr_map.pkl` (normalización de ingredientes) |
| **Transacción** | Una **receta** |
| **Ítem** | Un **ingrediente** con su nombre normalizado (`garlic clove` → `garlic`, `all-purpose flour` → `flour`) |
| **N° de transacciones** | 28.878 recetas (muestra aleatoria de 30.000 con semilla 42, después de la limpieza y de exigir ≥ 2 ingredientes) |
| **N° de ítems** | 222 ingredientes (soporte ≥ 0,5 % en la muestra) |
| **Ítems por transacción** | 7,2 en promedio (mediana 7) |
| **Uso del recomendador** | Sugerir el ingrediente que falta al armar una receta o una lista de compras (venta cruzada en un supermercado en línea), mostrando la regla que lo justifica |
