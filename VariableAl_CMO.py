import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os

# --- Configuración Inicial ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# --- PARÁMETROS DE CONFIGURACIÓN ---
# --------------------------------------------------------------------------
# --- Rutas de Archivos ---
FILE_PATH = r"D:\2025\FACULTAD\Trabajo final integrador\Python\CMO\costos_marginales_horarios_actualizados_a_12_2024.csv"

# --- Parámetros de Simulación ---
N_ANIOS_SIMULADOS = 15
N_ESCENARIOS = 1000

# --- Parámetros de Filtrado y Diagnóstico ---
GENERATE_DIAGNOSTIC_PLOTS = True 
OUTLIER_FILTERING_QUANTILE = 0.75
# Umbral de desviación estándar. Días con una desviación estándar menor a este valor
# serán considerados "planos" y se descartarán. Puedes ajustar este valor.
MIN_DAILY_STD_DEV = 6
# --------------------------------------------------------------------------


# --- Parte I: Carga y Preprocesamiento Robusto ---
print("--- Parte I: Carga y Preprocesamiento Robusto ---")

# 1.1. Carga y Acondicionamiento de Datos
print("\n1.1. Carga y Acondicionamiento de Datos")
try:
    df = pd.read_csv(FILE_PATH, sep=',', encoding='latin-1')
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo de datos en la ruta: {FILE_PATH}")
    exit()

# Seleccionar columnas por posición para mayor robustez
df = df.iloc[:, [2, 5, -1]].copy()
df.columns = ['fecha', 'hora', 'costo_marginal']
df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)

df['hora_ajustada'] = df['hora'] - 1
df['timestamp'] = df['fecha'] + pd.to_timedelta(df['hora_ajustada'], unit='h')
df.set_index('timestamp', inplace=True)
df = df[['costo_marginal']]
df['costo_marginal'] = pd.to_numeric(df['costo_marginal'], errors='coerce')
df.dropna(inplace=True)

df['Fecha'] = df.index.date
df['Hora'] = df.index.hour
print("Carga y acondicionamiento completado.")

# 1.2. Filtrar Días Incompletos
print("\n1.2. Filtrando días con datos incompletos...")
daily_counts = df.groupby('Fecha')['Hora'].count()
complete_days_index = daily_counts[daily_counts == 24].index
df_complete = df[df['Fecha'].isin(complete_days_index)].copy()
print(f"Se encontraron {len(complete_days_index)} días completos con 24 registros horarios.")

# 1.3. Segmentación por Estación
print("\n1.3. Segmentando datos por estación...")
def get_season(month):
    if month in [12, 1, 2]: return 'Verano'
    elif month in [3, 4, 5]: return 'Otoño'
    elif month in [6, 7, 8]: return 'Invierno'
    else: return 'Primavera'

df_complete['Estacion'] = pd.to_datetime(df_complete['Fecha']).dt.month.map(get_season)
print("Segmentación completada.")


# --- Parte II: Creación de Bancos de Curvas Diarias por Estación ---
print("\n--- Parte II: Creación de Bancos de Curvas Diarias por Estación ---")

seasonal_curve_banks = {}
seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
input_directory = os.path.dirname(FILE_PATH)

for season in tqdm(seasons, desc="Procesando Estaciones"):
    print(f"\nProcesando estación: {season}")
    
    season_data = df_complete[df_complete['Estacion'] == season]
    daily_curves_raw = season_data.pivot_table(index='Fecha', columns='Hora', values='costo_marginal')
    
    if len(daily_curves_raw) < 20:
        print(f"  -> No hay suficientes datos para {season}. Se omite.")
        continue

    # 2.1. Diagnóstico visual de datos históricos
    if GENERATE_DIAGNOSTIC_PLOTS:
        plt.figure(figsize=(12, 7))
        plt.plot(daily_curves_raw.T, color='gray', alpha=0.2)
        plt.plot(daily_curves_raw.mean(axis=0), color='red', linewidth=2, label='Curva Promedio Histórica')
        plt.title(f'Diagnóstico: Todas las curvas diarias históricas para {season}')
        plt.xlabel('Hora del día')
        plt.ylabel('CMO (u$s/MWh)')
        plt.legend()
        diagnostic_plot_path = os.path.join(input_directory, f'diagnostico_curvas_{season}.png')
        plt.savefig(diagnostic_plot_path)
        plt.close()
        print(f"  -> Gráfico de diagnóstico guardado en: {diagnostic_plot_path}")

    # 2.2. Filtrado de outliers (basado en la forma general)
    mean_curve = daily_curves_raw.mean(axis=0)
    mse_distances = ((daily_curves_raw - mean_curve)**2).mean(axis=1)
    distance_threshold = mse_distances.quantile(OUTLIER_FILTERING_QUANTILE)
    is_inlier = mse_distances < distance_threshold
    daily_curves_filtered_shape = daily_curves_raw[is_inlier]
    print(f"  -> Filtrado de outliers (forma): {len(daily_curves_raw)} curvas originales -> {len(daily_curves_filtered_shape)} curvas retenidas.")
        
    # FILTRO DEFINITIVO: 2.3. Filtrado de días planos por desviación estándar
    daily_std = daily_curves_filtered_shape.std(axis=1)
    is_dynamic = daily_std > MIN_DAILY_STD_DEV
    daily_curves_final = daily_curves_filtered_shape[is_dynamic]
    print(f"  -> Filtrado de curvas planas (std < {MIN_DAILY_STD_DEV}): {len(daily_curves_filtered_shape)} curvas -> {len(daily_curves_final)} curvas retenidas.")

    # 2.4. Guardar el banco de curvas final para la estación
    seasonal_curve_banks[season] = daily_curves_final.to_numpy()


# --- Parte III: Generación de Escenarios con Bootstrap Directo ---
print("\n--- Parte III: Generación de Escenarios con Bootstrap Directo ---")

season_sequence = ['Verano', 'Otoño', 'Invierno', 'Primavera']
print(f"Generando {N_ESCENARIOS} escenarios, cada uno con un día típico por estación para {N_ANIOS_SIMULADOS} años.")

index_tuples = [(anio, season, hour) for anio in range(1, N_ANIOS_SIMULADOS + 1) for season in season_sequence for hour in range(24)]
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Anio', 'Estacion', 'Hora'])
escenarios_df = pd.DataFrame(index=multi_index, columns=[f'Escenario_{i+1}' for i in range(N_ESCENARIOS)], dtype=float)

for i in tqdm(range(1, N_ESCENARIOS + 1), desc="Generando Escenarios"):
    scenario_column_data = []
    for anio in range(N_ANIOS_SIMULADOS):
        for season in season_sequence:
            if season not in seasonal_curve_banks or len(seasonal_curve_banks[season]) == 0:
                dia_tipico = np.zeros(24)
            else:
                curve_bank = seasonal_curve_banks[season]
                random_index = np.random.randint(0, len(curve_bank))
                dia_tipico = curve_bank[random_index]
                
            scenario_column_data.extend(dia_tipico)
    escenarios_df[f'Escenario_{i}'] = scenario_column_data

escenarios_df[escenarios_df < 0] = 0
print("\nSimulación de escenarios completada.")

# --- Parte IV: Visualización y Guardado de Resultados ---
print("\n--- Parte IV: Visualización y Guardado de Resultados ---")

plt.figure(figsize=(15, 8))
primer_escenario = escenarios_df['Escenario_1'].values[:96]
plt.plot(np.arange(96), primer_escenario, marker='o', linestyle='-', label='Escenario 1 (Primer Año)')
for j, season in enumerate(season_sequence):
    plt.axvline(x=j*24, color='grey', linestyle='--', linewidth=1)
    plt.text(j*24 + 12, plt.ylim()[1]*0.9, season, ha='center')
plt.title(f'Ejemplo de un Año Simulado (4 Días Típicos) con Bootstrap Directo')
plt.ylabel('CMO (u$s/MWh)')
plt.xlabel('Hora a lo largo del año simulado')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plot_path = os.path.join(input_directory, 'grafico_anio_simulado_bootstrap.png')
plt.savefig(plot_path)
plt.show()
print(f"Gráfico de ejemplo guardado en: '{plot_path}'")

output_csv_path = os.path.join(input_directory, 'escenarios_cmo_bootstrap.csv')
escenarios_df.to_csv(output_csv_path)
print(f"Los escenarios simulados se han guardado en: '{output_csv_path}'")