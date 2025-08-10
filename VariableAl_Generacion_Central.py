# -*- coding: utf-8 -*-
"""
Análisis de Generación Hidroeléctrica y Generación de Escenarios

Este script realiza un análisis de los datos históricos de generación de una
central hidroeléctrica, ajusta un Modelo de Mezcla Gaussiana (GMM) para cada
estación y luego genera múltiples escenarios de días típicos en un formato CSV
específico.

El proceso incluye:
1. Carga y consolidación de datos horarios desde archivos .MDB.
2. Limpieza de datos y segmentación por estaciones.
3. Ajuste de un Modelo de Mezcla Gaussiana (GMM) para cada estación.
4. Generación de N escenarios para M años, basados en perfiles de días típicos.
   - El día típico de cada estación se genera directamente como una muestra de 24 horas
     del GMM correspondiente.
5. Exportación de los resultados a un archivo CSV con separador de punto y coma.
"""
import os
import glob
import warnings
import pandas as pd
import pypyodbc as odbc
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

# --- Configuración Inicial ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')

class AjustadorGeneradorHidro:
    """
    Clase para cargar, procesar y ajustar un Modelo de Mezcla Gaussiana (GMM) a los
    datos de generación hidroeléctrica de una central específica.
    """
    def __init__(self, data_path, plant_name, plant_capacity_mw):
        self.raw_data_path = data_path
        self.plant_name = plant_name
        self.plant_capacity = plant_capacity_mw
        self.full_data = pd.DataFrame()
        self.seasonal_data = {}
        self.best_fits = {}
        # Mapeo de nombres de estaciones para la salida
        self.season_name_map = {
            'Summer': 'Verano', 'Autumn': 'Otono',
            'Winter': 'Invierno', 'Spring': 'Primavera'
        }
        print(f"Ajustador inicializado para la central: {self.plant_name}")

    def cargar_y_preparar_datos(self):
        print("\n--- Iniciando Carga y Preparación de Datos ---")
        mdb_files = glob.glob(os.path.join(self.raw_data_path, 'PO*.MDB'))
        if not mdb_files:
            print(f"Error: No se encontraron archivos .MDB en la ruta: {self.raw_data_path}")
            return False

        all_months_data = []
        print(f"Encontrados {len(mdb_files)} archivos. Procesando...")
        for file_path in tqdm(mdb_files, desc="Cargando archivos MDB"):
            df_month = self._leer_mdb(file_path)
            if not df_month.empty:
                all_months_data.append(df_month)

        if not all_months_data:
            print("Error: No se pudieron leer datos de ningún archivo.")
            return False

        self.full_data = pd.concat(all_months_data, ignore_index=True)
        print(f"Datos consolidados: {self.full_data.shape[0]} registros horarios cargados.")

        self._limpiar_y_transformar()
        self._segmentar_por_estacion()
        print("--- Carga y Preparación de Datos Finalizada ---")
        return True

    def _leer_mdb(self, file_path):
        DRIVER_NAME = 'Microsoft Access Driver (*.mdb, *.accdb)'
        conn_str = f"DRIVER={{{DRIVER_NAME}}};DBQ={file_path};"
        query = f"SELECT FECHA, HORA, ENERGIA FROM VALORES_GENERADORES WHERE GRUPO = '{self.plant_name}'"
        try:
            with odbc.connect(conn_str) as connection:
                df = pd.read_sql(query, connection)
                if not df.empty:
                    df.columns = [col.upper() for col in df.columns]
                return df
        except odbc.Error as ex:
            sqlstate = ex.args[0]
            if 'SQLSTATE=HY000' in str(sqlstate):
                 print(f"\nAdvertencia: No se pudo leer la tabla de {os.path.basename(file_path)}. Puede que no contenga datos para '{self.plant_name}'.")
            else:
                 print(f"\nError al conectar o leer el archivo {os.path.basename(file_path)}: {ex}")
            return pd.DataFrame()

    def _limpiar_y_transformar(self):
        print("Limpiando y transformando datos...")
        df = self.full_data
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        df['HORA'] = df['HORA'] - 1
        df['timestamp'] = df.apply(lambda row: row['FECHA'] + pd.to_timedelta(row['HORA'], unit='h'), axis=1)
        df = df.set_index('timestamp').sort_index()
        df['ENERGIA'] = pd.to_numeric(df['ENERGIA'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['ENERGIA'], inplace=True)
        df = df[df['ENERGIA'] >= 0]
        df = df[df['ENERGIA'] <= self.plant_capacity]
        print(f"Registros eliminados por limpieza: {initial_rows - len(df)}")
        self.full_data = df[['ENERGIA']]

    def _segmentar_por_estacion(self):
        print("Segmentando datos por estación...")
        df = self.full_data
        # Usa el mapeo interno para los nombres en inglés
        seasons_map_eng = { 12: 'Summer', 1: 'Summer', 2: 'Summer', 3: 'Autumn', 4: 'Autumn', 5: 'Autumn', 6: 'Winter', 7: 'Winter', 8: 'Winter', 9: 'Spring', 10: 'Spring', 11: 'Spring' }
        df['Season'] = df.index.month.map(seasons_map_eng)
        for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
            self.seasonal_data[season] = df[df['Season'] == season]
            print(f"  - {self.season_name_map[season]}: {len(self.seasonal_data[season])} registros.")

    def realizar_ajuste_estacional(self, component_range=list(range(2, 9)), manual_selection=None):
        if manual_selection is None:
            manual_selection = {}

        print("\n--- Iniciando Ajuste Adaptativo de GMM por Estación ---")
        for season, data_df in self.seasonal_data.items():
            print(f"\nAnalizando estación: {self.season_name_map[season]}")
            data_series = data_df['ENERGIA'].dropna()
            
            if data_series.nunique() < 2:
                print(f"  -> Advertencia: Datos constantes o vacíos. Se omite el ajuste.")
                continue

            data_reshaped = data_series.values.reshape(-1, 1)
            best_gmm = None

            if season in manual_selection:
                n_components = manual_selection[season]
                print(f"  -> Usando selección manual: {n_components} componentes.")
                try:
                    # Usamos un random_state fijo aquí solo para que el *ajuste* sea repetible
                    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
                    best_gmm = gmm.fit(data_reshaped)
                except Exception as e:
                    print(f"    Error al ajustar con {n_components} componentes: {e}")
            else:
                # Búsqueda automática si no hay selección manual
                print(" -> No se proveyó selección manual. Buscando el mejor número de componentes vía BIC...")
                bic_scores, models = [], []
                for n_components in tqdm(component_range, desc=f"Ajustando {season}"):
                    try:
                        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
                        gmm.fit(data_reshaped)
                        bic_scores.append(gmm.bic(data_reshaped))
                        models.append(gmm)
                    except Exception:
                        bic_scores.append(np.inf)
                        models.append(None)
                
                if not any(np.isfinite(bic_scores)):
                    print("  -> Falló el ajuste para todos los componentes.")
                    continue
                
                best_index = np.argmin(bic_scores)
                best_gmm = models[best_index]

            if best_gmm:
                self.best_fits[season] = best_gmm
                print(f"Ajuste de GMM para {self.season_name_map[season]} completado.")
                print(f"  -> Modelo final con {best_gmm.n_components} componentes.")
        
        print("\n--- Ajuste de GMM Adaptativo Finalizado ---")

# --- Bloque Principal de Ejecución ---
if __name__ == '__main__':
    # --- Parámetros de Configuración ---
    DATA_DIRECTORY_PATH = r"D:\2025\FACULTAD\Trabajo final integrador\Centrales\DatosGen"
    PLANT_NAME = 'ULLUHI'
    PLANT_CAPACITY_MW = 45.0
    
    # Parámetros para la generación de escenarios
    NUM_ESCENARIOS = 1000
    NUM_ANIOS = 15

    # --- Ejecución del Ajustador ---
    ajustador = AjustadorGeneradorHidro(DATA_DIRECTORY_PATH, PLANT_NAME, PLANT_CAPACITY_MW)
    
    if ajustador.cargar_y_preparar_datos():
        # Define tus elecciones de componentes aquí después de un análisis previo
        season_component_choices = {
             'Autumn': 4,
             'Spring': 4,
             'Summer': 5,
             'Winter': 5
        }
        ajustador.realizar_ajuste_estacional(manual_selection=season_component_choices)

        # --- Generación de Escenarios en Formato CMO ---
        if not ajustador.best_fits:
            print("\nNo se encontraron ajustes de distribución válidos. Se omite la generación de escenarios.")
        else:
            print("\n--- Iniciando Generación de Escenarios de Generación Hidroeléctrica ---")
            
            # Pre-construcción del DataFrame final
            seasons_order_eng = ['Summer', 'Autumn', 'Winter', 'Spring']
            seasons_order_esp = [ajustador.season_name_map[s] for s in seasons_order_eng]
            hours_in_day = 24
            rows_per_year = len(seasons_order_esp) * hours_in_day

            df_final = pd.DataFrame({
                'Anio': np.repeat(np.arange(1, NUM_ANIOS + 1), rows_per_year),
                'Estacion': np.tile(np.repeat(seasons_order_esp, hours_in_day), NUM_ANIOS),
                'Hora': np.tile(np.arange(1, hours_in_day + 1), NUM_ANIOS * len(seasons_order_esp))
            })

            # Bucle principal para generar cada escenario
            for i in tqdm(range(1, NUM_ESCENARIOS + 1), desc="Generando Escenarios"):
                scenario_column_data = []
                # Bucle anidado para cada año dentro del escenario
                for year in range(NUM_ANIOS):
                    # Generar un día típico para cada estación en este año
                    for season_eng in seasons_order_eng:
                        original_gmm = ajustador.best_fits[season_eng]
                        
                        # --- LÓGICA CORREGIDA ---
                        # 1. Crear una nueva instancia de GMM para asegurar un estado aleatorio nuevo
                        sampling_gmm = GaussianMixture(
                            n_components=original_gmm.n_components,
                            covariance_type=original_gmm.covariance_type,
                            random_state=None  # Clave: Usa un estado aleatorio diferente cada vez
                        )
                        
                        # 2. Copiar los parámetros del modelo ya ajustado
                        sampling_gmm.weights_ = original_gmm.weights_
                        sampling_gmm.means_ = original_gmm.means_
                        sampling_gmm.covariances_ = original_gmm.covariances_
                        sampling_gmm.precisions_cholesky_ = original_gmm.precisions_cholesky_
                        
                        # 3. Generar la muestra desde la nueva instancia
                        samples, _ = sampling_gmm.sample(n_samples=hours_in_day)
                        
                        # Aplicar restricciones físicas
                        samples = np.clip(samples, 0, ajustador.plant_capacity)
                        
                        # El perfil ya es el día típico
                        typical_day_profile = samples.flatten()
                        
                        scenario_column_data.extend(typical_day_profile)
                
                # Añadir la columna completa del escenario al DataFrame
                df_final[f'Escenario_{i}'] = scenario_column_data

            # --- Guardado del Archivo Final ---
            print("\nOptimizando y guardando el archivo final...")
            scenario_cols = [f'Escenario_{i}' for i in range(1, NUM_ESCENARIOS + 1)]
            df_final[scenario_cols] = df_final[scenario_cols].round(5)

            directorio_destino = r'D:\2025\FACULTAD\Trabajo final integrador\Centrales\Python distribución prob'
            nombre_archivo = 'escenarios_generacion_hidro.csv'
            ruta_completa = os.path.join(directorio_destino, nombre_archivo)
            
            os.makedirs(directorio_destino, exist_ok=True)
            df_final.to_csv(ruta_completa, index=False, sep=';', decimal='.')

            print(f"\nProceso completado. Archivo guardado en: '{ruta_completa}'")