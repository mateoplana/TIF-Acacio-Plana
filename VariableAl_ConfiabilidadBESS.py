import numpy as np
import pandas as pd
from scipy.stats import beta
import os
from tqdm import tqdm

def get_beta_params(mean, variance):
    """Calcula los parámetros alpha y beta a partir de la media y la varianza."""
    if variance >= mean * (1 - mean):
        raise ValueError("La varianza es demasiado alta para esta media.")
    common_term = (mean * (1 - mean) / variance) - 1
    alpha = mean * common_term
    beta_param = (1 - mean) * common_term
    return alpha, beta_param

# --- 1. Parámetros Principales de la Simulación ---
NUM_ESCENARIOS = 1000  # Número de escenarios de disponibilidad a generar
NUM_ANIOS = 15        # Horizonte de la simulación en años

# --- 2. Parámetros del BESS y Confiabilidad ---
BESS_POWER_MW = 75
BESS_ENERGY_MWH = 300
INVERTER_POWER_KW = 500
CONTAINER_ENERGY_MWH = 2

# Parámetros de Confiabilidad (MTBF en horas)
MTBF_inversor_horas = 50000
MTBF_hvac_container_horas = 40000

# Parámetros de la distribución Beta (estado de salud general)
mean_availability = 0.99
variance = 0.0001

# --- 3. Configuración del Entorno ---
# Cálculo de componentes
num_inverters = BESS_POWER_MW / (INVERTER_POWER_KW / 1000)
num_containers = BESS_ENERGY_MWH / CONTAINER_ENERGY_MWH
n_hours_per_year = 8760

print("--- Configuración del Modelo de Disponibilidad del BESS ---")
print(f"Generando {NUM_ESCENARIOS} escenarios para {NUM_ANIOS} años, usando días típicos.")
print(f"Número de inversores: {int(num_inverters)}")
print(f"MTBF por inversor: {MTBF_inversor_horas:,} horas")
print(f"MTBF por HVAC de contenedor: {MTBF_hvac_container_horas:,} horas")

# --- 4. Pre-construcción del DataFrame Final ---
print("\nPre-construyendo la estructura del DataFrame final...")
seasons_order = ['Verano', 'Otono', 'Invierno', 'Primavera']
hours_in_day = 24
rows_per_year = len(seasons_order) * hours_in_day

# Mapeo de meses a estaciones para la selección de días típicos
season_month_map = {
    12: 'Verano', 1: 'Verano', 2: 'Verano',
    3: 'Otono', 4: 'Otono', 5: 'Otono',
    6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
    9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
}
# Crear un índice de tiempo de un solo año para agrupar
yearly_index = pd.to_datetime(pd.date_range(start='2025-01-01', periods=n_hours_per_year, freq='h'))

# Crear las columnas base
df_final = pd.DataFrame({
    'Anio': np.repeat(np.arange(1, NUM_ANIOS + 1), rows_per_year),
    'Estacion': np.tile(np.repeat(seasons_order, hours_in_day), NUM_ANIOS),
    'Hora': np.tile(np.arange(1, hours_in_day + 1), NUM_ANIOS * len(seasons_order))
})

# --- 5. Generación de Escenarios para Días Típicos ---
try:
    # Calcular parámetros una sola vez
    alpha, beta_p = get_beta_params(mean_availability, variance)
    lambda_inversores = (1 / MTBF_inversor_horas) * n_hours_per_year * num_inverters
    lambda_hvac = (1 / MTBF_hvac_container_horas) * n_hours_per_year * num_containers
    
    print(f"\nParámetros Beta calculados: alpha={alpha:.2f}, beta={beta_p:.2f}")
    print(f"Tasa de falla anual esperada para inversores (λ): {lambda_inversores:.2f}")
    print(f"Tasa de falla anual esperada para HVAC (λ): {lambda_hvac:.2f}")

    # Bucle principal para generar cada escenario
    for i in tqdm(range(1, NUM_ESCENARIOS + 1), desc="Generando Escenarios"):
        scenario_column_data = []
        # Bucle anidado para cada año dentro del escenario
        for year in range(NUM_ANIOS):
            # Generar datos para un año completo
            availability_full_year = beta.rvs(alpha, beta_p, size=n_hours_per_year)
            availability_full_year[availability_full_year > 1] = 1

            # Simular y aplicar fallas para ESE año
            num_fallas_inv = np.random.poisson(lam=lambda_inversores)
            num_fallas_hvac = np.random.poisson(lam=lambda_hvac)
            horas_falla_inv = np.random.choice(np.arange(n_hours_per_year), size=num_fallas_inv, replace=False)
            horas_falla_hvac = np.random.choice(np.arange(n_hours_per_year), size=num_fallas_hvac, replace=False)
            
            availability_full_year[horas_falla_inv] -= (1 / num_inverters)
            availability_full_year[horas_falla_hvac] -= (1 / num_containers)
            availability_full_year[availability_full_year < 0] = 0
            
            # --- LÓGICA MEJORADA: CALCULAR EL DÍA TÍPICO PROMEDIO ---
            # Crear un DataFrame temporal para el año actual
            df_year = pd.DataFrame({
                'availability': availability_full_year,
                'season': yearly_index.month.map(season_month_map),
                'hour': yearly_index.hour
            })
            
            # Agrupar por estación y hora, y calcular la media
            # Esto calcula el perfil horario esperado para cada estación
            avg_seasonal_profiles = df_year.groupby(['season', 'hour'])['availability'].mean().unstack(level='season')

            # Añadir los perfiles del día típico para este año al resultado del escenario
            for season in seasons_order:
                # Los valores de la hora van de 0 a 23, se ajustan al formato final
                typical_day_values = avg_seasonal_profiles[season].values
                scenario_column_data.extend(typical_day_values)
        
        # Añadir la columna completa del escenario al DataFrame final
        df_final[f'Escenario_{i}'] = scenario_column_data

    # --- 6. OPTIMIZACIÓN Y GUARDADO ---
    print("\nOptimizando y guardando el archivo final...")
    scenario_cols = [f'Escenario_{i}' for i in range(1, NUM_ESCENARIOS + 1)]
    df_final[scenario_cols] = df_final[scenario_cols].round(5)

    directorio_destino = r'D:\2025\FACULTAD\Trabajo final integrador\Confiabilidad BESS'
    nombre_archivo = 'escenarios_disponibilidad_bess_dias_tipicos.csv'
    ruta_completa = os.path.join(directorio_destino, nombre_archivo)
    
    os.makedirs(directorio_destino, exist_ok=True)
    # Se añade sep=';' para que Excel lo reconozca correctamente como separador de columnas.
    df_final.to_csv(ruta_completa, index=False, sep=';')

    print(f"\nProceso completado. Archivo guardado en: '{ruta_completa}'")

except (ValueError, KeyError) as e:
    print(f"\nHa ocurrido un error: {e}")