import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

# --- Configuración Inicial ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# --- Parte I: Definición de Supuestos (Juicio de Experto) ---
print("--- Parte I: Definición de Supuestos para el Pago por Potencia ---")

# --- ACCIÓN REQUERIDA: Define tus supuestos aquí ---
# 1. VALOR CENTRAL: El valor más probable que esperas para el Pago por Potencia.
VALOR_MAS_PROBABLE = 5122.98636
# 2. INCERTIDUMBRE: ¿Qué tanto podría variar este precio? (en porcentaje)
INCERTIDUMBRE_PORCENTAJE = 20.0
# --- Fin de la Acción Requerida ---

# --- Parte II: Construcción de la Distribución Triangular ---
print("\n--- Parte II: Construyendo la Distribución Triangular ---")

incertidumbre_decimal = INCERTIDUMBRE_PORCENTAJE / 100.0
VALOR_MINIMO = VALOR_MAS_PROBABLE * (1 - incertidumbre_decimal)
VALOR_MAXIMO = VALOR_MAS_PROBABLE * (1 + incertidumbre_decimal)

loc = VALOR_MINIMO
scale = VALOR_MAXIMO - VALOR_MINIMO
c = (VALOR_MAS_PROBABLE - loc) / scale

distribucion_triangular = stats.triang(c=c, loc=loc, scale=scale)

print("Parámetros de la Distribución Triangular calculados:")
print(f"  - Mínimo (loc):    {loc:.2f}")
print(f"  - Máximo:          {VALOR_MAXIMO:.2f}")
print(f"  - Más Probable (c): {VALOR_MAS_PROBABLE:.2f}")

# --- Parte III: Generación de Escenarios de 15 Años (por Estación) ---
print("\n\n--- Parte III: Generación de Escenarios de 15 Años para Pago por Potencia (por Estación) ---")

# --- CONFIGURACIÓN DE LA SIMULACIÓN ---
N_ANIOS_SIMULADOS = 15
N_ESCENARIOS = 1000  # Número total de escenarios de 15 años a generar
# --- FIN DE LA CONFIGURACIÓN ---

season_sequence = ['Verano', 'Otoño', 'Invierno', 'Primavera']
# Se calcula el total de períodos (4 estaciones por año)
total_periodos_simulacion = len(season_sequence) * N_ANIOS_SIMULADOS

print(f"Generando {N_ESCENARIOS} escenarios, cada uno de {N_ANIOS_SIMULADOS} años ({total_periodos_simulacion} períodos estacionales por escenario).")
np.random.seed(43) # Usamos una semilla diferente a la del CMO para asegurar independencia

# Generar todos los valores aleatorios de una sola vez para máxima eficiencia
# Dimensiones: (total_periodos x N_ESCENARIOS)
all_scenarios = distribucion_triangular.rvs(size=(total_periodos_simulacion, N_ESCENARIOS))

# Asegurarse de que no haya precios negativos (aunque es improbable con la dist. triangular)
all_scenarios[all_scenarios < 0] = 0

# Crear un índice jerárquico (MultiIndex) para el DataFrame final
index_tuples = []
for anio in range(1, N_ANIOS_SIMULADOS + 1):
    for season in season_sequence:
        # Se elimina el bucle de la hora
        index_tuples.append((anio, season))

multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Anio', 'Estacion'])

# Crear el DataFrame final
escenarios_df = pd.DataFrame(
    all_scenarios,
    index=multi_index,
    columns=[f'Escenario_{i+1}' for i in range(N_ESCENARIOS)]
)

print("\nSimulación de escenarios de 15 años completada.")
print("Dimensiones de la matriz de escenarios (períodos x escenarios):", escenarios_df.shape)

# --- Parte IV: Visualización y Guardado de Resultados ---
print("\n--- Parte IV: Visualización y Guardado de Resultados ---")

plt.figure(figsize=(15, 7))
N_GRAFICAR = 50
plot_index = np.arange(total_periodos_simulacion)
plt.plot(plot_index, escenarios_df.iloc[:, :N_GRAFICAR], color='lightblue', alpha=0.3)

media_escenarios = escenarios_df.mean(axis=1)
plt.plot(plot_index, media_escenarios, color='red', linewidth=2, label='Valor Esperado (Media)')

plt.title(f'Simulación de Montecarlo de Pago por Potencia para {N_ANIOS_SIMULADOS} Años ({N_ESCENARIOS} escenarios)')
plt.ylabel('Pago por Potencia (ej. u$s/MW-mes)')
plt.xlabel(f'Período Estacional a lo largo de la simulación ({total_periodos_simulacion} períodos totales)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

output_csv_path = 'escenarios_15_anios_pago_potencia_por_estacion.csv'
escenarios_df.to_csv(output_csv_path)
print(f"Los escenarios simulados se han guardado en: '{output_csv_path}'")
