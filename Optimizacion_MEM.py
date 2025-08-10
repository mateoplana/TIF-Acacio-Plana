# -*- coding: utf-8 -*-
"""
HydroBoost: Modelo de Optimización Anual por Estaciones con Ciclo Externo

Este script implementa el modelo de optimización para la operación conjunta
de una central hidroeléctrica y un BESS.

"""

# Importar las librerías necesarias
import pandas as pd
from pyomo.environ import *
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from io import BytesIO
import numpy as np
import numpy_financial as npf

# --- CÓDIGO PARA CAMBIAR EL DIRECTORIO DE TRABAJO AUTOMÁTICAMENTE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# ---(nombre excel parametros)
excel_file_name = 'parametros_mercado_mem.xlsx'
# --- CONFIGURACIÓN DE LAS RUTAS DE ARCHIVOS DE SALIDA ---
output_dir = os.path.join(script_dir, 'resultados_optimizacion')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# SECCIÓN DE CONFIGURACIÓN DEL CICLO DE OPTIMIZACIÓN EXTERNA
# ==============================================================================
potencias_mw_a_probar = [1, 20, 40, 75] 
duraciones_horas_a_probar = [4, 6, 8] 
TASA_DESCUENTO = 0.10
VIDA_UTIL_ANOS = 15
VIDA_UTIL_SUBESTACION_ANOS = 40
TASA_DEGRADACION_ANUAL_CAPACIDAD = 0.025
dias_por_estacion = 91
# ==============================================================================
# DATOS DE ENTRADA (ESTÁTICOS)
# ==============================================================================
P_hidro_max_central = 45.0
FACTOR_CAUDAL_POTENCIA = 0.6962
costo_om_hidro_fijo = 100000

hydrology_data_example = {
    'caudales_verano': [44.067, 42.415, 42.056, 42.013, 42.243, 42.458, 42.803, 42.846, 42.717, 36.052, 31.341, 30.623, 30.364, 30.235, 30.867, 34.400, 43.334, 49.123, 51.536, 52.369, 52.613, 52.900, 52.312, 51.033],
    'caudales_otono': [11.993, 9.882, 9.408, 9.293, 9.322, 9.322, 9.351, 9.307, 9.293, 8.719, 8.417, 8.417, 8.273, 8.216, 8.158, 8.374, 14.967, 18.428, 19.994, 19.979, 19.807, 19.649, 19.577, 19.261],
    'caudales_invierno': [2.959, 2.901, 2.887, 2.887, 2.887, 2.887, 2.887, 2.873, 2.873, 2.873, 2.873, 3.074, 3.074, 3.074, 3.088, 3.059, 3.059, 3.074, 3.074, 3.074, 3.074, 3.174, 3.174, 3.174],
    'caudales_primavera': [30.178, 29.775, 29.603, 29.718, 29.761, 29.847, 29.589, 29.574, 27.951, 16.130, 11.993, 12.094, 12.137, 12.195, 12.238, 12.209, 12.726, 21.373, 33.366, 42.817, 47.026, 48.318, 48.318, 45.733]
}

COSTO_POR_CICLO_BESS = 20

INCENTIVOS_POR_ESTACION = {
    'Verano': range(14, 17),
    'Otoño': range(16, 19),
    'Invierno': range(18, 21),
    'Primavera': range(16, 19)
}
premio_por_descarga_pico = 5

BESS_PARAMETROS_BASE = {
    'eta_C': 0.97,
    'eta_D': 0.97,
    'soc_min': 0.10,
    'soc_max': 0.90,
}

potencia_red_max = 120

# ==============================================================================
# DEFINICIÓN DE FUNCIONES DEL MODELO Y CICLO EXTERNO
# ==============================================================================

def cargar_datos_mercado_estacional(excel_file_name):
    """Carga los precios horarios desde la 'Hoja 2'"""
    try:
        df_precios = pd.read_excel(excel_file_name, sheet_name='Hoja 2')
    except Exception as e:
        print(f"Error al leer la 'Hoja 2' del archivo Excel: {e}")
        return None, None
    seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    all_seasonal_data = {}
    for season_name in seasons:
        matching_columns = [col for col in df_precios.columns if col.lower() == season_name.lower()]
        if not matching_columns: continue
        actual_season_col = matching_columns[0]
        all_seasonal_data[season_name] = {'lambda_E': df_precios[actual_season_col].dropna().tolist()}
    try:
        df_fa = pd.read_excel(excel_file_name, sheet_name='FA')
        fa_anual_dict = {}
        for index, row in df_fa.iterrows():
            periodo_str = str(row['Años'])
            factor = row['Factor']
            if '-' in periodo_str:
                inicio, fin = map(int, periodo_str.split('-'))
                for anio in range(inicio, fin + 1):
                    fa_anual_dict[anio] = factor
            else:
                anio = int(float(periodo_str))
                fa_anual_dict[anio] = factor
    except Exception as e:
        print(f"❌ Error inesperado al leer o procesar la hoja 'FA': {e}")
        return None, None
    return all_seasonal_data, fa_anual_dict

def definir_parametros_y_variables(model, bess_data, market_data, fixed_hydro_gen):
    """Define los parámetros y variables del modelo simplificado"""
    model.T = Set(initialize=range(1, 25))
    model.lambda_E = Param(model.T, initialize={t: market_data['lambda_E'][t-1] for t in model.T})
    model.P_C_max = Param(initialize=bess_data['P_C_max'])
    model.P_D_max = Param(initialize=bess_data['P_D_max'])
    model.E_max = Param(initialize=bess_data['E_max'])
    model.eta_C = Param(initialize=bess_data['eta_C'])
    model.eta_D = Param(initialize=bess_data['eta_D'])
    model.soc_min_abs = Param(initialize=bess_data['soc_min'] * bess_data['E_max'])
    model.soc_max_abs = Param(initialize=bess_data['soc_max'] * bess_data['E_max'])
    model.p_H_total_fixed = Param(model.T, initialize={t: fixed_hydro_gen[t-1] for t in model.T})
    model.p_HG_t = Var(model.T, within=NonNegativeReals)
    model.p_HB_t = Var(model.T, within=NonNegativeReals)
    model.p_D_bht = Var(model.T, within=NonNegativeReals)
    model.soc_bht = Var(model.T, within=NonNegativeReals, bounds=(bess_data['soc_min'] * bess_data['E_max'], bess_data['soc_max'] * bess_data['E_max']))
    model.u_C_bht = Var(model.T, within=Binary)
    model.u_D_bht = Var(model.T, within=Binary)
    model.p_GB_t = Var(model.T, within=NonNegativeReals)
    model.start_C_bht = Var(model.T, within=Binary)
    model.start_D_bht = Var(model.T, within=Binary)
    model.soc_initial_abs = Var(within=NonNegativeReals, bounds=(bess_data['soc_min'] * bess_data['E_max'], bess_data['soc_min'] * bess_data['E_max'] * 1.01))
    return model

def definir_restricciones_simplificadas(model):
    """Define las restricciones del modelo simplificado"""
    model.constraints = ConstraintList()
    for t in model.T:
        model.constraints.add(model.p_H_total_fixed[t] == model.p_HG_t[t] + model.p_HB_t[t])
        soc_anterior = model.soc_bht[t-1] * 0.99 if t > 1 else model.soc_initial_abs
        model.constraints.add(model.soc_bht[t] == soc_anterior + (model.p_HB_t[t] + model.p_GB_t[t]) * model.eta_C - (model.p_D_bht[t] / model.eta_D))
        model.constraints.add(model.soc_bht[t] >= model.soc_min_abs)
        model.constraints.add(model.soc_bht[t] <= model.soc_max_abs)
        model.constraints.add(model.p_HB_t[t] + model.p_GB_t[t] <= model.P_C_max * model.u_C_bht[t])
        model.constraints.add(model.p_D_bht[t] <= model.P_D_max * model.u_D_bht[t])
        model.constraints.add(model.u_C_bht[t] + model.u_D_bht[t] <= 1)
        model.constraints.add(model.p_GB_t[t] <= potencia_red_max)
        model.constraints.add(model.p_D_bht[t] + model.p_HG_t[t] <= potencia_red_max)
        if t > 1:
            model.constraints.add(model.start_C_bht[t] >= model.u_C_bht[t] - model.u_C_bht[t-1])
            model.constraints.add(model.start_D_bht[t] >= model.u_D_bht[t] - model.u_D_bht[t-1])
        else:
            model.constraints.add(model.start_C_bht[t] >= model.u_C_bht[t])
            model.constraints.add(model.start_D_bht[t] >= model.u_D_bht[t])
    model.constraints.add(sum((model.start_C_bht[t] + model.start_D_bht[t]) for t in model.T) <= 2)
    model.constraints.add(model.soc_bht[24] == model.soc_bht[1])
    return model

def definir_objetivo_simplificado(model, params_mercado, horas_incentivo_estacion):
    """Define la función objetivo del modelo simplificado"""
    ingreso_energia_hidro_red_usd = sum(model.p_HG_t[t] * model.lambda_E[t] for t in model.T)
    precio_fijo_venta_bess = params_mercado.get('PRECIO_ENERGIA_SUMINISTRADA_BESS')
    ingreso_energia_bess_red_usd = sum(model.p_D_bht[t] * precio_fijo_venta_bess for t in model.T)
    beneficio_incentivo_pico_usd = sum(model.p_D_bht[t] * premio_por_descarga_pico for t in model.T if t in horas_incentivo_estacion)
    precio_fijo_penalizacion_perdidas = params_mercado.get('COSTO_ENERGIA_PERDIDAS_BESS')
    costo_fijo_carga_red = (1 - model.eta_C) + (1 - model.eta_D)
    costo_compra_energia_bess_usd = sum(model.p_GB_t[t] * costo_fijo_carga_red * precio_fijo_penalizacion_perdidas for t in model.T)
    costo_ciclos_bess_usd = sum((model.start_C_bht[t] + model.start_D_bht[t])/2 * COSTO_POR_CICLO_BESS for t in model.T)
    beneficio_variable_diario_usd_expr = (ingreso_energia_hidro_red_usd + ingreso_energia_bess_red_usd + beneficio_incentivo_pico_usd - costo_compra_energia_bess_usd - costo_ciclos_bess_usd)
    model.objective = Objective(expr=beneficio_variable_diario_usd_expr, sense=maximize)
    return model

def calcular_capex_bess_no_lineal(potencia_mw, energia_mwh):
    """Calcula el CAPEX diferenciando los costos del BESS y de la subestación."""
    if potencia_mw == 0: return 0, 0
    costo_potencia = (-0.581594 * potencia_mw**2 + 135.811907 * potencia_mw + 1467.210091) * 1000
    capex_subestacion = costo_potencia
    costo_energia = (energia_mwh / 2) * 254603
    ajuste = (energia_mwh - 2 * potencia_mw) * 25539
    capex_bess = costo_energia - ajuste
    return capex_bess, capex_subestacion

def generar_graficos_operacion(df_horario, P_bess_mw, E_bess_mwh, season_name, output_path):
    """Genera y guarda gráficos de operación estacional."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Gráfica de Flujos de Potencia
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Hidro_Total_Fija_MW'], label='Generación Hidro Total (Fija)', color='cyan', linestyle='-')
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Hidro_a_Red_MW'], label='Hidro a Red (MW)', color='blue', linestyle='--')
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Hidro_a_BESS_MW'], label='Hidro a BESS (MW)', color='purple', linestyle='--')
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Red_a_BESS_MW'], label='Red a BESS (MW)', color='red', drawstyle='steps-post')
    ax1.plot(df_horario['Hora'], df_horario['Potencia_BESS_a_Red_MW'], label='BESS a Red (MW)', color='green', drawstyle='steps-post')
    ax1.set_ylabel('Potencia (MW)')
    ax1.set_title(f'Flujos de Potencia para BESS {P_bess_mw}MW-{E_bess_mwh}MWh - {season_name}')
    ax1.legend()
    ax1.grid(True)

    # Gráfica del SOC del BESS
    ax2.plot(df_horario['Hora'], df_horario['SOC_MWh'], label='SOC BESS (MWh)', color='orange')
    ax2.set_xlabel('Hora del Día')
    ax2.set_ylabel('SOC (MWh)')
    ax2.set_ylim(bottom=0)
    ax2.set_title(f'Estado de Carga (SOC) del BESS')
    ax2.legend()
    ax2.grid(True)

    plt.xticks(range(1, 25))
    plt.tight_layout()
    plot_filename = f"Grafico_Operacion_P{int(P_bess_mw)}MW_E{int(E_bess_mwh)}MWh_{season_name}.png"
    plot_filepath = os.path.join(output_path, plot_filename)
    plt.savefig(plot_filepath)
    plt.close(fig)
    print(f"    -> Gráfico de operación para '{season_name}' guardado en: {plot_filepath}")

def ejecutar_optimizacion_anual_simplificada(bess_data, all_seasonal_data, params_mercado, excel_writer=None, output_path_graficos=None):
    """Ejecuta la optimización para un año completo, por estaciones."""
    total_annual_operating_profit = 0
    total_annual_hydro_profit = 0
    seasonal_dfs = {}
    season_to_caudal_key = {'Verano': 'caudales_verano', 'Otoño': 'caudales_otono', 'Invierno': 'caudales_invierno', 'Primavera': 'caudales_primavera'}
    estaciones_simuladas = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    for season_name in estaciones_simuladas:
        horas_incentivo_actual = INCENTIVOS_POR_ESTACION[season_name]
        caudales_horarios = hydrology_data_example[season_to_caudal_key[season_name]]
        fixed_hydro_generation = [min(caudal * FACTOR_CAUDAL_POTENCIA, P_hidro_max_central) for caudal in caudales_horarios]
        market_data_for_season = all_seasonal_data[season_name]
        model_instance = ConcreteModel(name=f"HydroBoost_{season_name}_Simplified")
        model_instance = definir_parametros_y_variables(model_instance, bess_data, market_data_for_season, fixed_hydro_generation)
        model_instance = definir_objetivo_simplificado(model_instance, params_mercado, horas_incentivo_actual)
        model_instance = definir_restricciones_simplificadas(model_instance)
        solver = SolverFactory('cbc')
        results = solver.solve(model_instance, tee=False)
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]):
            daily_variable_profit_usd = value(model_instance.objective)
            daily_hydro_profit_usd = sum(value(model_instance.p_HG_t[t]) * value(model_instance.lambda_E[t]) for t in model_instance.T)
            total_annual_hydro_profit += daily_hydro_profit_usd * dias_por_estacion
            costo_ciclos_del_dia_usd = sum((value(model_instance.start_C_bht[t]) + value(model_instance.start_D_bht[t]))/2 * COSTO_POR_CICLO_BESS for t in model_instance.T)
            premio_pico_del_dia_usd = sum(value(model_instance.p_D_bht[t]) * premio_por_descarga_pico for t in model_instance.T if t in horas_incentivo_actual)
            beneficio_diario_para_reporte = daily_variable_profit_usd + costo_ciclos_del_dia_usd - premio_pico_del_dia_usd
            total_annual_operating_profit += beneficio_diario_para_reporte * dias_por_estacion
            if excel_writer or output_path_graficos:
                df_horario = pd.DataFrame({
                    'Hora': [t for t in model_instance.T],
                    'Potencia_Hidro_Total_Fija_MW': [value(model_instance.p_H_total_fixed[t]) for t in model_instance.T],
                    'Potencia_Hidro_a_Red_MW': [value(model_instance.p_HG_t[t]) for t in model_instance.T],
                    'Potencia_Hidro_a_BESS_MW': [value(model_instance.p_HB_t[t]) for t in model_instance.T],
                    'Potencia_Red_a_BESS_MW': [value(model_instance.p_GB_t[t]) for t in model_instance.T],
                    'Potencia_BESS_a_Red_MW': [value(model_instance.p_D_bht[t]) for t in model_instance.T],
                    'SOC_MWh': [value(model_instance.soc_bht[t]) for t in model_instance.T],
                    'Precio_Spot_USD_MWh': [value(model_instance.lambda_E[t]) for t in model_instance.T]
                })
                seasonal_dfs[season_name] = df_horario
                if excel_writer:
                    df_horario.to_excel(excel_writer, sheet_name=season_name, index=False)
                if output_path_graficos:
                    generar_graficos_operacion(df_horario, bess_data['P_D_max'], bess_data['E_max'], season_name, output_path_graficos)
        else:
            print(f"   -> Advertencia: El modelo para {season_name} fue {results.solver.termination_condition}. Beneficio = 0.")
    return total_annual_operating_profit, total_annual_hydro_profit, seasonal_dfs

def formatear_y_autoajustar_hojas(writer, dfs_dict, workbook):
    """Aplica formato de número y autoajusta el ancho de las columnas para un diccionario de DataFrames."""
    currency_format = workbook.add_format({'num_format': '$#,##0.00'})
    percent_format = workbook.add_format({'num_format': '0.00%'})
    
    for sheet_name, df in dfs_dict.items():
        worksheet = writer.sheets[sheet_name]
        for col_num, col_name in enumerate(df.columns):
            max_len = max(df[col_name].astype(str).map(len).max(), len(str(col_name))) + 2
            if '(USD)' in col_name:
                worksheet.set_column(col_num, col_num, max_len, currency_format)
            elif 'TIR' in col_name:
                worksheet.set_column(col_num, col_num, max_len, percent_format)
            else:
                worksheet.set_column(col_num, col_num, max_len)

def crear_grafico_superficie_3d(writer, df_resultados, sheet_name, z_column, title):
    """Crea un gráfico de superficie 3D, marca el máximo y lo inserta en Excel."""
    worksheet = writer.sheets[sheet_name]
    
    df_plot = df_resultados[['Potencia_MW', 'Energia_MWh', z_column]].dropna()
    if df_plot.shape[0] < 4:
        print(f"  Advertencia: No hay suficientes datos ({df_plot.shape[0]}) para crear el gráfico de superficie 3D para {title}.")
        return

    x = df_plot['Potencia_MW']
    y = df_plot['Energia_MWh']
    z = df_plot[z_column]

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='cubic')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=f'{z_column} (%)')
    
    # --- Encontrar y marcar el punto máximo ---
    max_z_idx = z.idxmax()
    max_x = x.loc[max_z_idx]
    max_y = y.loc[max_z_idx]
    max_z_val = z.loc[max_z_idx]
    
    ax.scatter(max_x, max_y, max_z_val, c='red', marker='*', s=200, edgecolor='black', depthshade=True, label=f'Máximo: {max_z_val:.2%}')

    ax.text(max_x, max_y, max_z_val * 0.9, f'Máximo ({max_x} MW, {max_y} MWh)', color='black', fontweight='bold', ha='center', va='top')
    
    ax.set_xlabel('Potencia (MW)')
    ax.set_ylabel('Energía (MWh)')
    ax.set_zlabel(z_column)
    ax.set_title(title)
    
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    
    worksheet.insert_image('J2', f'{z_column}_surface_plot.png', {'image_data': image_buffer})
    print(f"  Gráfico de superficie 3D para '{title}' insertado en la hoja '{sheet_name}'.")

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    ruta_excel = os.path.join(script_dir, excel_file_name)
    all_seasonal_data, fa_anual_dict = cargar_datos_mercado_estacional(ruta_excel)
    if all_seasonal_data is None: exit()
    try:
        df_params = pd.read_excel(ruta_excel, sheet_name='Hoja 1')
        params_mercado = df_params.set_index('Parametro').to_dict()['Valor']
    except Exception as e:
        print(f"Error fatal al leer 'Hoja 1': {e}")
        exit()
    resultados_finales = []
    print("\n" + "="*80)
    print("INICIANDO ANÁLISIS DE CONFIGURACIONES")
    print("="*80)

    for potencia in potencias_mw_a_probar:
        for duracion in duraciones_horas_a_probar:
            energia = potencia * duracion
            print(f"\n--- Analizando BESS: {potencia} MW / {energia} MWh ({duracion}h) ---")
            bess_data_actual = BESS_PARAMETROS_BASE.copy()
            bess_data_actual.update({'P_C_max': potencia, 'P_D_max': potencia, 'E_max': energia})
            beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, _ = ejecutar_optimizacion_anual_simplificada(bess_data_actual, all_seasonal_data, params_mercado)
            beneficio_op_anual_solo_bess = beneficio_op_anual_combinado - beneficio_op_anual_solo_hidro
            capex_bess, capex_subestacion = calcular_capex_bess_no_lineal(potencia, energia)
            capex_total = capex_bess + capex_subestacion
            valor_residual_subestacion = capex_subestacion * (VIDA_UTIL_SUBESTACION_ANOS - VIDA_UTIL_ANOS) / VIDA_UTIL_SUBESTACION_ANOS if VIDA_UTIL_SUBESTACION_ANOS > VIDA_UTIL_ANOS else 0
            
            flujos_combinado = [-capex_total]
            costo_opex_sub_BESS = (-0.581594 * potencia**2 + 135.811907 * potencia + 1467.210091) * 1000 * 0.01 + potencia * 8000
            ingreso_potencia_hidro_anual = params_mercado.get('HIDRO_REM_POT_PRECIO_BASE', 0) * P_hidro_max_central * 12
            valor_ofertado_bess = params_mercado.get('VALOR_OFERTADO_BESS', 0)
            for anio in range(1, VIDA_UTIL_ANOS + 1):
                factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
                beneficio_op_degradado_bess = beneficio_op_anual_solo_bess * factor_degradacion
                beneficio_total_anual_degradado = beneficio_op_anual_solo_hidro + beneficio_op_degradado_bess
                fa_del_anio = fa_anual_dict.get(anio, 1.0)
                ingreso_potencia_bess_anual = valor_ofertado_bess * fa_del_anio * potencia * 12
                costo_opex_anual_total = costo_om_hidro_fijo + costo_opex_sub_BESS
                flujo_anual = (beneficio_total_anual_degradado + ingreso_potencia_hidro_anual + ingreso_potencia_bess_anual - costo_opex_anual_total)
                if anio == VIDA_UTIL_ANOS: flujo_anual += valor_residual_subestacion
                flujos_combinado.append(flujo_anual)
            
            flujos_bess = [-capex_total]
            for anio in range(1, VIDA_UTIL_ANOS + 1):
                factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
                beneficio_op_degradado_bess = beneficio_op_anual_solo_bess * factor_degradacion
                fa_del_anio = fa_anual_dict.get(anio, 1.0)
                ingreso_potencia_bess_anual = valor_ofertado_bess * fa_del_anio * potencia * 12
                flujo_anual = (beneficio_op_degradado_bess + ingreso_potencia_bess_anual - costo_opex_sub_BESS)
                if anio == VIDA_UTIL_ANOS: flujo_anual += valor_residual_subestacion
                flujos_bess.append(flujo_anual)

            try:
                tir_combinado = npf.irr(flujos_combinado)
            except Exception:
                tir_combinado = np.nan
            try:
                tir_bess = npf.irr(flujos_bess)
            except Exception:
                tir_bess = np.nan

            van_combinado = npf.npv(TASA_DESCUENTO, flujos_combinado)
            van_bess = npf.npv(TASA_DESCUENTO, flujos_bess)
            
            print(f"   -> VAN (Solo BESS): ${van_bess:,.2f} | TIR (Solo BESS): {tir_bess:.2%}" if pd.notna(tir_bess) else f"   -> VAN (Solo BESS): ${van_bess:,.2f} | TIR (Solo BESS): N/A")
            resultados_finales.append({'Potencia_MW': potencia, 'Energia_MWh': energia, 'VAN_Combinado_USD': van_combinado, 'TIR_Combinado': tir_combinado, 'VAN_BESS_USD': van_bess, 'TIR_BESS': tir_bess})

    if not resultados_finales:
        print("No se completó ninguna simulación. No se puede generar el reporte.")
        exit()
    df_resultados_completos = pd.DataFrame(resultados_finales)
    optimo_idx = df_resultados_completos['VAN_BESS_USD'].idxmax()
    configuracion_optima = df_resultados_completos.loc[optimo_idx]
    print("\n" + "="*80)
    print("🏆 CONFIGURACIÓN ÓPTIMA ENCONTRADA (Basado en VAN del BESS) 🏆")
    print(f"   - Potencia: {configuracion_optima['Potencia_MW']:.0f} MW, Energía: {configuracion_optima['Energia_MWh']:.0f} MWh")
    tir_optima_str = f"{configuracion_optima['TIR_BESS']:.2%}" if pd.notna(configuracion_optima['TIR_BESS']) else "N/A"
    print(f"   - VAN (Solo BESS): ${configuracion_optima['VAN_BESS_USD']:,.2f}, TIR (Solo BESS): {tir_optima_str}")
    print("="*80)

    print("\nGenerando reporte detallado y gráficos para la configuración óptima...")
    opt_potencia = configuracion_optima['Potencia_MW']
    opt_energia = configuracion_optima['Energia_MWh']
    excel_optimo_filename = f"Reporte_Optimo_P{int(opt_potencia)}MW_E{int(opt_energia)}MWh.xlsx"
    excel_optimo_filepath = os.path.join(output_dir, excel_optimo_filename)

    with pd.ExcelWriter(excel_optimo_filepath, engine='xlsxwriter') as writer:
        bess_data_optima = BESS_PARAMETROS_BASE.copy()
        bess_data_optima.update({'P_C_max': opt_potencia, 'P_D_max': opt_potencia, 'E_max': opt_energia})
        beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, seasonal_dfs = ejecutar_optimizacion_anual_simplificada(bess_data_optima, all_seasonal_data, params_mercado, excel_writer=writer, output_path_graficos=output_dir)
        print("  Hojas de operación estacional y gráficos generados.")

        beneficio_op_anual_solo_bess = beneficio_op_anual_combinado - beneficio_op_anual_solo_hidro
        capex_bess, capex_subestacion = calcular_capex_bess_no_lineal(opt_potencia, opt_energia)
        capex_total = capex_bess + capex_subestacion
        valor_residual_subestacion = capex_subestacion * (VIDA_UTIL_SUBESTACION_ANOS - VIDA_UTIL_ANOS) / VIDA_UTIL_SUBESTACION_ANOS if VIDA_UTIL_SUBESTACION_ANOS > VIDA_UTIL_ANOS else 0
        costo_opex_sub_BESS = (-0.581594 * opt_potencia**2 + 135.811907 * opt_potencia + 1267.210091) * 1000 * 0.01 + opt_potencia * 8000
        ingreso_potencia_hidro_anual = params_mercado.get('HIDRO_REM_POT_PRECIO_BASE', 0) * P_hidro_max_central * 12
        valor_ofertado_bess = params_mercado.get('VALOR_OFERTADO_BESS', 0)

        detalle_flujo_combinado = []
        flujos_combinado_calc = [-capex_total]
        for anio in range(1, VIDA_UTIL_ANOS + 1):
            factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
            beneficio_op_degradado_bess = beneficio_op_anual_solo_bess * factor_degradacion
            beneficio_total_anual_degradado = beneficio_op_anual_solo_hidro + beneficio_op_degradado_bess
            fa_del_anio = fa_anual_dict.get(anio, 1.0)
            ingreso_potencia_bess_anual = valor_ofertado_bess * fa_del_anio * opt_potencia * 12
            costo_opex_anual_total = costo_om_hidro_fijo + costo_opex_sub_BESS
            valor_residual_del_anio = valor_residual_subestacion if anio == VIDA_UTIL_ANOS else 0
            flujo_anual = (beneficio_total_anual_degradado + ingreso_potencia_hidro_anual + ingreso_potencia_bess_anual - costo_opex_anual_total + valor_residual_del_anio)
            flujos_combinado_calc.append(flujo_anual)
            try: tir_acumulada = npf.irr(flujos_combinado_calc)
            except Exception: tir_acumulada = np.nan
            van_acumulado = npf.npv(TASA_DESCUENTO, flujos_combinado_calc)
            detalle_flujo_combinado.append({'Año': anio, 'Beneficio Operativo (USD)': beneficio_total_anual_degradado, 'Ingreso Potencia Hidro (USD)': ingreso_potencia_hidro_anual, 'Ingreso Potencia BESS (USD)': ingreso_potencia_bess_anual, 'Costo OPEX Total (USD)': costo_opex_anual_total, 'Valor Residual Subestación (USD)': valor_residual_del_anio, 'Flujo de Caja Neto (USD)': flujo_anual, 'VAN Acumulado (USD)': van_acumulado, 'TIR Acumulada': f"{tir_acumulada:.2%}" if pd.notna(tir_acumulada) else "N/A"})
        df_flujo_combinado = pd.DataFrame(detalle_flujo_combinado)
        
        detalle_flujo_bess = []
        flujos_bess_calc = [-capex_total]
        for anio in range(1, VIDA_UTIL_ANOS + 1):
            factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
            beneficio_op_degradado_bess = beneficio_op_anual_solo_bess * factor_degradacion
            fa_del_anio = fa_anual_dict.get(anio, 1.0)
            ingreso_potencia_bess_anual = valor_ofertado_bess * fa_del_anio * opt_potencia * 12
            valor_residual_del_anio = valor_residual_subestacion if anio == VIDA_UTIL_ANOS else 0
            flujo_anual = (beneficio_op_degradado_bess + ingreso_potencia_bess_anual - costo_opex_sub_BESS + valor_residual_del_anio)
            flujos_bess_calc.append(flujo_anual)
            try: tir_acumulada = npf.irr(flujos_bess_calc)
            except Exception: tir_acumulada = np.nan
            van_acumulado = npf.npv(TASA_DESCUENTO, flujos_bess_calc)
            detalle_flujo_bess.append({'Año': anio, 'Beneficio Operativo BESS (USD)': beneficio_op_degradado_bess, 'Ingreso Potencia BESS (USD)': ingreso_potencia_bess_anual, 'Costo OPEX BESS (USD)': costo_opex_sub_BESS, 'Valor Residual Subestación (USD)': valor_residual_del_anio, 'Flujo de Caja Neto (USD)': flujo_anual, 'VAN Acumulado (USD)': van_acumulado, 'TIR Acumulada': f"{tir_acumulada:.2%}" if pd.notna(tir_acumulada) else "N/A"})
        df_flujo_bess = pd.DataFrame(detalle_flujo_bess)
        
        tir_combinado_str = f"{configuracion_optima['TIR_Combinado']:.2%}" if pd.notna(configuracion_optima['TIR_Combinado']) else "N/A"
        tir_bess_str = f"{configuracion_optima['TIR_BESS']:.2%}" if pd.notna(configuracion_optima['TIR_BESS']) else "N/A"
        df_resumen_financiero = pd.DataFrame({'Métrica': ['Potencia (MW)', 'Energía (MWh)', 'CAPEX BESS (USD)', 'CAPEX Subestación (USD)', 'CAPEX Total (USD)', 'VAN Combinado (USD)', 'TIR Combinado', 'VAN BESS (USD)', 'TIR BESS'], 'Valor': [opt_potencia, opt_energia, capex_bess, capex_subestacion, capex_total, configuracion_optima['VAN_Combinado_USD'], tir_combinado_str, configuracion_optima['VAN_BESS_USD'], tir_bess_str]})
        
        df_resultados_completos_str = df_resultados_completos.copy()
        df_resultados_completos_str['TIR_Combinado'] = df_resultados_completos_str['TIR_Combinado'].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')
        df_resultados_completos_str['TIR_BESS'] = df_resultados_completos_str['TIR_BESS'].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')
        
        df_flujo_combinado.to_excel(writer, sheet_name='Flujo_Caja_Combinado', index=False)
        df_flujo_bess.to_excel(writer, sheet_name='Flujo_Caja_BESS', index=False)
        df_resumen_financiero.to_excel(writer, sheet_name='Resumen_Financiero', index=False)
        df_resultados_completos_str.to_excel(writer, sheet_name='Resumen_Configuraciones', index=False)
        print("  Hojas de datos generadas.")

        workbook  = writer.book
        dfs_to_format = {**seasonal_dfs, 'Flujo_Caja_Combinado': df_flujo_combinado, 'Flujo_Caja_BESS': df_flujo_bess, 'Resumen_Financiero': df_resumen_financiero, 'Resumen_Configuraciones': df_resultados_completos_str}
        formatear_y_autoajustar_hojas(writer, dfs_to_format, workbook)
        print("  Columnas autoajustadas y formato aplicado.")

        # --- Crear y añadir el gráfico de superficie 3D ---
        crear_grafico_superficie_3d(writer, df_resultados_completos, 'Resumen_Configuraciones', 'TIR_BESS', 'Superficie de TIR del BESS')

    print(f"\n✅ Reporte final y completo para la configuración óptima guardado en: {excel_optimo_filepath}")