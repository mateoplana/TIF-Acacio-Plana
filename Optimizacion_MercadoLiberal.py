# -*- coding: utf-8 -*-
"""

HydroBoost: Modelo de Optimización para Mercado Liberalizado

Este script adapta el modelo de optimización a un mercado de energía liberalizado,
maximizando ingresos por arbitraje de energía y venta de servicios auxiliares.

"""

# Importar las librerías necesarias
import pandas as pd
from pyomo.environ import *
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import numpy as np
import numpy_financial as npf

# --- CÓDIGO PARA CAMBIAR EL DIRECTORIO DE TRABAJO AUTOMÁTICAMENTE ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURACIÓN DE LAS RUTAS DE ARCHIVOS DE SALIDA ---
output_dir = os.path.join(script_dir, 'resultados_optimizacion_liberalizado')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# SECCIÓN DE CONFIGURACIÓN Y PARÁMETROS
# ==============================================================================
potencias_mw_a_probar = [5, 10]
duraciones_horas_a_probar = [4]
TASA_DESCUENTO = 0.1378
VIDA_UTIL_ANOS = 15
VIDA_UTIL_SUBESTACION_ANOS = 40
dias_por_estacion = 91
TASA_DEGRADACION_ANUAL_CAPACIDAD = 0.025 # Afecta solo al BESS

# ==============================================================================
# DATOS DE ENTRADA (ESTÁTICOS)
# ==============================================================================
P_hidro_max_central = 45.0
FACTOR_CAUDAL_POTENCIA = 0.6962
costo_om_hidro_fijo_anual = 100000
potencia_red_max=120

hydrology_data_example = {
    'caudales_verano': [44.067, 42.415, 42.056, 42.013, 42.243, 42.458, 42.803, 42.846, 42.717, 36.052, 31.341, 30.623, 30.364, 30.235, 30.867, 34.400, 43.334, 49.123, 51.536, 52.369, 52.613, 52.900, 52.312, 51.033],
    'caudales_otono': [11.993, 9.882, 9.408, 9.293, 9.322, 9.322, 9.351, 9.307, 9.293, 8.719, 8.417, 8.417, 8.273, 8.216, 8.158, 8.374, 14.967, 18.428, 19.994, 19.979, 19.807, 19.649, 19.577, 19.261],
    'caudales_invierno': [2.959, 2.901, 2.887, 2.887, 2.887, 2.887, 2.887, 2.873, 2.873, 2.873, 2.873, 3.074, 3.074, 3.074, 3.088, 3.059, 3.059, 3.074, 3.074, 3.074, 3.074, 3.174, 3.174, 3.174],
    'caudales_primavera': [30.178, 29.775, 29.603, 29.718, 29.761, 29.847, 29.589, 29.574, 27.951, 16.130, 11.993, 12.094, 12.137, 12.195, 12.238, 12.209, 12.726, 21.373, 33.366, 42.817, 47.026, 48.318, 48.318, 45.733]
}

BESS_PARAMETROS_BASE = {
    'eta_C': 0.97, 'eta_D': 0.97, 'soc_min': 0.10, 'soc_max': 0.90
}

# ==============================================================================
# DEFINICIÓN DE FUNCIONES DEL MODELO Y CICLO EXTERNO
# ==============================================================================

def cargar_datos_mercado_liberalizado(file_path_excel):
    """Carga los datos de precios y servicios auxiliares desde sus hojas correspondientes."""
    market_data = {}
    try:
        df_cmo = pd.read_excel(file_path_excel, sheet_name='cmo', header=0)
        df_delta_ru = pd.read_excel(file_path_excel, sheet_name='delta-RU', header=0)
        df_delta_rd = pd.read_excel(file_path_excel, sheet_name='delta_RD', header=0)
        df_params = pd.read_excel(file_path_excel, sheet_name='parametros_mercado_liberal', header=0)
        print(f"Datos de mercado liberalizado cargados desde: {file_path_excel}")
    except Exception as e:
        print(f"Error al leer el archivo Excel del mercado liberalizado: {e}")
        return None, None

    try:
        params_dict = df_params.set_index('Parametro')['Valor'].to_dict()
    except (KeyError, ValueError) as e:
        print(f"Error al procesar la hoja 'parametros_mercado_liberal': {e}. Verifique el formato.")
        return None, None

    precio_rpf = params_dict.get('precio_disponibilidad_rpf', 0)
    precio_r10 = params_dict.get('precio_disponibilidad_r10', 0)

    lambda_R_up_fixed = [precio_rpf] * 24
    lambda_R_down_fixed = [precio_rpf] * 24
    lambda_S_fixed = [precio_r10] * 24

    seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    for season_name in seasons:
        current_season_data = {}
        try:
            current_season_data['lambda_E'] = df_cmo[season_name].dropna().tolist()
            current_season_data['delta_RU'] = df_delta_ru[season_name].dropna().tolist()
            current_season_data['delta_RD'] = df_delta_rd[season_name].dropna().tolist()
            current_season_data['lambda_R_up'] = lambda_R_up_fixed
            current_season_data['lambda_R_down'] = lambda_R_down_fixed
            current_season_data['lambda_S'] = lambda_S_fixed
            market_data[season_name] = current_season_data
        except KeyError as e:
            print(f"Advertencia: No se pudo encontrar la columna para la estación '{season_name}' en una de las hojas. Error: {e}")
            continue

    return market_data, params_dict

def definir_parametros(model, bess_data, market_data, fixed_hydro_gen, params_mercado):
    """Define los parámetros del modelo Pyomo."""
    model.T = Set(initialize=range(1, 25))
    model.lambda_E = Param(model.T, initialize={t: market_data['lambda_E'][t-1] for t in model.T})
    model.lambda_R_up = Param(model.T, initialize={t: market_data['lambda_R_up'][t-1] for t in model.T})
    model.lambda_R_down = Param(model.T, initialize={t: market_data['lambda_R_down'][t-1] for t in model.T})
    model.lambda_S = Param(model.T, initialize={t: market_data['lambda_S'][t-1] for t in model.T})
    model.delta_RU = Param(model.T, initialize={t: market_data['delta_RU'][t-1] for t in model.T})
    model.delta_RD = Param(model.T, initialize={t: market_data['delta_RD'][t-1] for t in model.T})
    model.P_C_max = Param(initialize=bess_data['P_C_max'])
    model.P_D_max = Param(initialize=bess_data['P_D_max'])
    model.E_max = Param(initialize=bess_data['E_max'])
    model.eta_C = Param(initialize=bess_data['eta_C'])
    model.eta_D = Param(initialize=bess_data['eta_D'])
    model.soc_min_abs = Param(initialize=bess_data['soc_min'] * bess_data['E_max'])
    model.soc_max_abs = Param(initialize=bess_data['soc_max'] * bess_data['E_max'])
    model.p_H_total_fixed = Param(model.T, initialize={t: fixed_hydro_gen[t-1] for t in model.T})

    porcentaje_rpf = params_mercado.get('porcentaje_reserva_rpf', 0.04)
    porcentaje_r10 = params_mercado.get('porcentaje_reserva_r10', 0.02)
    
    r_up_val = porcentaje_rpf * bess_data['P_D_max']
    s_val = porcentaje_r10 * bess_data['P_D_max']
    r_down_val = porcentaje_rpf * bess_data['P_C_max']

    model.r_B_up_t = Param(model.T, initialize={t: r_up_val for t in model.T})
    model.s_B_t = Param(model.T, initialize={t: s_val for t in model.T})
    model.r_B_down_t = Param(model.T, initialize={t: r_down_val for t in model.T})

    return model

def definir_variables(model):
    """Define las variables de decisión del modelo Pyomo."""
    model.p_HG_t = Var(model.T, within=NonNegativeReals)
    model.p_HB_t = Var(model.T, within=NonNegativeReals)
    model.p_GB_t = Var(model.T, within=NonNegativeReals)
    model.p_D_bht = Var(model.T, within=NonNegativeReals)
    model.soc_total_t = Var(model.T, bounds=(model.soc_min_abs, model.soc_max_abs))
    model.u_C_bht = Var(model.T, within=Binary)
    model.u_D_bht = Var(model.T, within=Binary)
    model.soc_initial_total = Var(bounds=(model.soc_min_abs, model.soc_max_abs))
    return model

def definir_restricciones(model):
    """Define las restricciones operativas del modelo Pyomo."""
    model.constraints = ConstraintList()
    for t in model.T:
        potencia_carga_arbitraje = model.p_HB_t[t] + model.p_GB_t[t]
        potencia_descarga_arbitraje = model.p_D_bht[t]
        
        model.constraints.add(model.p_HG_t[t] + potencia_descarga_arbitraje + model.delta_RU[t] * model.r_B_up_t[t] <= potencia_red_max)
        model.constraints.add(model.p_GB_t[t] + model.delta_RD[t] * model.r_B_down_t[t] <= potencia_red_max)
        
        model.constraints.add(model.p_H_total_fixed[t] == model.p_HG_t[t] + model.p_HB_t[t])
        
        soc_anterior = model.soc_total_t[t-1] if t > 1 else model.soc_initial_total
        energia_cargada_total = (potencia_carga_arbitraje + model.delta_RD[t] * model.r_B_down_t[t]) * model.eta_C
        energia_descargada_total = (potencia_descarga_arbitraje + model.delta_RU[t] * model.r_B_up_t[t]) / model.eta_D
        model.constraints.add(model.soc_total_t[t] == soc_anterior + energia_cargada_total - energia_descargada_total)

        model.constraints.add(potencia_carga_arbitraje + model.delta_RD[t] * model.r_B_down_t[t] <= model.P_C_max)
        model.constraints.add(potencia_descarga_arbitraje + model.delta_RU[t] * model.r_B_up_t[t] <= model.P_D_max)
        
        model.constraints.add(model.u_C_bht[t] + model.u_D_bht[t] <= 1)
        model.constraints.add(potencia_carga_arbitraje <= model.P_C_max * model.u_C_bht[t])
        model.constraints.add(potencia_descarga_arbitraje <= model.P_D_max * model.u_D_bht[t])

        energia_necesaria_para_reserva_up = (model.delta_RU[t] * model.r_B_up_t[t] + model.s_B_t[t]) / model.eta_D
        model.constraints.add(potencia_descarga_arbitraje / model.eta_D <= soc_anterior - model.soc_min_abs - energia_necesaria_para_reserva_up)
        
        espacio_necesario_para_reserva_down = (model.delta_RD[t] * model.r_B_down_t[t]) * model.eta_C
        model.constraints.add(potencia_carga_arbitraje * model.eta_C <= model.soc_max_abs - soc_anterior - espacio_necesario_para_reserva_down)

    energia_inicial_reserva_up = (model.delta_RU[1] * model.r_B_up_t[1] + model.s_B_t[1]) / model.eta_D
    model.constraints.add(model.soc_initial_total >= model.soc_min_abs + energia_inicial_reserva_up)
    
    espacio_inicial_reserva_down = (model.delta_RD[1] * model.r_B_down_t[1]) * model.eta_C
    model.constraints.add(model.soc_initial_total <= model.soc_max_abs - espacio_inicial_reserva_down)
    
    model.constraints.add(model.soc_total_t[24] >= model.soc_initial_total)
    return model

def definir_objetivo_liberalizado(model):
    """Define la función objetivo para el mercado liberalizado."""
    ingreso_energia = sum(model.lambda_E[t] * (model.p_HG_t[t] + model.p_D_bht[t]) for t in model.T)
    costo_energia = sum(model.lambda_E[t] * model.p_GB_t[t] for t in model.T)
    ingreso_reserva_up = sum(model.lambda_R_up[t] * model.r_B_up_t[t] for t in model.T)
    ingreso_reserva_down = sum(model.lambda_R_down[t] * model.r_B_down_t[t] for t in model.T)
    ingreso_reserva_spin = sum(model.lambda_S[t] * model.s_B_t[t] for t in model.T)
    ajuste_por_regulacion_subida = sum(model.lambda_E[t] * model.delta_RU[t] * model.r_B_up_t[t] for t in model.T)
    ajuste_por_regulacion_bajada = sum(model.lambda_E[t] * model.delta_RD[t] * model.r_B_down_t[t] for t in model.T)
    
    beneficio_total_diario = (
        (ingreso_energia - costo_energia) +
        (ingreso_reserva_up + ingreso_reserva_down + ingreso_reserva_spin) +
        (ajuste_por_regulacion_subida - ajuste_por_regulacion_bajada)
    )
    model.objective = Objective(expr=beneficio_total_diario, sense=maximize)
    return model

def graficar_resultados_liberalizados(df_horario, P_bess_mw, E_bess_mwh, season_name, output_dir):
    """Genera gráficos de potencia como escalones para el modelo liberalizado."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Resultados Optimización Liberalizada - BESS {P_bess_mw}MW-{E_bess_mwh}MWh - {season_name}', fontsize=16)
    
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Hidro_Total_Fija_MW'], label='Hidro Total (Fija)', color='cyan', lw=2)
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Hidro_a_Red_MW'], label='Hidro a Red', color='blue', linestyle='--', drawstyle='steps-post')
    ax1.plot(df_horario['Hora'], df_horario['Potencia_BESS_a_Red_MW'], label='BESS a Red (Energía)', color='green', lw=2, drawstyle='steps-post')
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Red_a_BESS_MW'], label='Red a BESS', color='red', drawstyle='steps-post', lw=2)
    ax1.plot(df_horario['Hora'], df_horario['Potencia_Hidro_a_BESS_MW'], label='Hidro a BESS', color='purple', linestyle=':', drawstyle='steps-post')
    ax1.set_ylabel('Potencia (MW)')
    ax1.set_title('Flujos de Potencia')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(df_horario['Hora'], df_horario['SOC_Total_MWh'], label='SOC BESS (MWh)', color='orange', lw=2)
    ax2.set_ylabel('SOC (MWh)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(bottom=0)
    ax2.set_title('Estado de Carga (SOC) y Precio de Energía')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlabel('Hora del Día')

    ax2b = ax2.twinx()
    ax2b.plot(df_horario['Hora'], df_horario['Precio_Spot_USD_MWh'], label='Precio Spot (USD/MWh)', color='gray', linestyle='-.')
    ax2b.set_ylabel('Precio (USD/MWh)', color='gray')
    ax2b.tick_params(axis='y', labelcolor='gray')
    ax2b.legend(loc='upper right')

    plt.xticks(range(1, 25))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"resultados_liberalizado_P{int(P_bess_mw)}MW_E{int(E_bess_mwh)}MWh_{season_name}.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close(fig)
    print(f"    Gráfico guardado en: {plot_filepath}")

def calcular_beneficios_desglosados_diarios(model):
    """
    Calcula los beneficios operativos diarios desglosados para el BESS.
    Retorna un diccionario con los componentes del beneficio.
    """
    ingreso_venta_energia_bess = sum(value(model.lambda_E[t] * model.p_D_bht[t]) for t in model.T)
    costo_compra_energia_bess = sum(value(model.lambda_E[t] * model.p_GB_t[t]) for t in model.T)
    costo_energia_hidro_a_bess = sum(value(model.lambda_E[t] * model.p_HB_t[t]) for t in model.T)
    beneficio_arbitraje_energia = ingreso_venta_energia_bess - costo_compra_energia_bess - costo_energia_hidro_a_bess

    ingreso_reserva_up = sum(value(model.lambda_R_up[t] * model.r_B_up_t[t]) for t in model.T)
    ingreso_reserva_down = sum(value(model.lambda_R_down[t] * model.r_B_down_t[t]) for t in model.T)
    ingreso_reserva_spin = sum(value(model.lambda_S[t] * model.s_B_t[t]) for t in model.T)
    ajuste_por_regulacion_subida = sum(value(model.lambda_E[t] * model.delta_RU[t] * model.r_B_up_t[t]) for t in model.T)
    ajuste_por_regulacion_bajada = sum(value(model.lambda_E[t] * model.delta_RD[t] * model.r_B_down_t[t]) for t in model.T)
    ingreso_total_servicios_auxiliares = (ingreso_reserva_up + ingreso_reserva_down + ingreso_reserva_spin +
                                          ajuste_por_regulacion_subida - ajuste_por_regulacion_bajada)
    
    beneficio_bess_diario = beneficio_arbitraje_energia + ingreso_total_servicios_auxiliares
    beneficio_total_diario = value(model.objective)

    return {
        "beneficio_total_diario": beneficio_total_diario,
        "beneficio_bess_diario": beneficio_bess_diario,
        "beneficio_arbitraje_energia_diario": beneficio_arbitraje_energia,
        "ingreso_servicios_auxiliares_diario": ingreso_total_servicios_auxiliares
    }

def calcular_capex_bess_no_lineal(potencia_mw, energia_mwh):
    """
    Calcula el CAPEX diferenciando los costos del BESS y de la subestación.
    """
    if potencia_mw == 0:
        return 0, 0
    
    costo_potencia = (-0.581594 * potencia_mw**2 + 135.811907 * potencia_mw + 1467.210091) * 1000
    capex_subestacion = costo_potencia

    costo_energia = (energia_mwh / 2) * 254603
    ajuste = (energia_mwh - 2 * potencia_mw) * 25539
    capex_bess = costo_energia - ajuste
    
    return capex_bess, capex_subestacion

def calcular_pago_potencia_requerido(van_inicial_bess, tasa_descuento, vida_util, potencia_mw):
    """Calcula el pago por potencia anual (USD/MW-año) necesario para que el VAN sea cero."""
    if van_inicial_bess >= 0 or potencia_mw == 0:
        return 0, 0
    try:
        pv_annuity_factor = sum([(1 / (1 + tasa_descuento)**t) for t in range(1, vida_util + 1)])
    except ZeroDivisionError:
        return float('inf'), float('inf')
    if pv_annuity_factor == 0:
        return float('inf'), float('inf')
    ingreso_anual_adicional_requerido = -van_inicial_bess / pv_annuity_factor
    pago_por_potencia_anual_mw = ingreso_anual_adicional_requerido / potencia_mw
    return pago_por_potencia_anual_mw, ingreso_anual_adicional_requerido

def ejecutar_optimizacion_anual_liberalizada(bess_data, all_seasonal_data, params_mercado, excel_writer=None, output_path_graficos=None):
    """Ejecuta el ciclo de optimización estacional y retorna beneficios anuales desglosados."""
    beneficio_anual_total = 0
    beneficio_anual_bess = 0
    beneficio_anual_arbitraje = 0
    ingreso_anual_ssaa = 0
    
    seasonal_dfs = {}
    
    season_to_caudal_key = {'Verano': 'caudales_verano', 'Otoño': 'caudales_otono', 'Invierno': 'caudales_invierno', 'Primavera': 'caudales_primavera'}
    estaciones_simuladas = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    
    for season_name in estaciones_simuladas:
        caudales_horarios = hydrology_data_example[season_to_caudal_key[season_name]]
        fixed_hydro_generation = [min(caudal * FACTOR_CAUDAL_POTENCIA, P_hidro_max_central) for caudal in caudales_horarios]
        market_data_for_season = all_seasonal_data.get(season_name)
        if not market_data_for_season:
             print(f"   -> Saltando estación {season_name} por falta de datos.")
             continue
        
        model_instance = ConcreteModel(name=f"HydroBoost_{season_name}_Liberalized")
        model_instance = definir_parametros(model_instance, bess_data, market_data_for_season, fixed_hydro_generation, params_mercado)
        model_instance = definir_variables(model_instance)
        model_instance = definir_restricciones(model_instance)
        model_instance = definir_objetivo_liberalizado(model_instance)
        solver = SolverFactory('cbc')
        results = solver.solve(model_instance, tee=False)
        
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]):
            beneficios_diarios = calcular_beneficios_desglosados_diarios(model_instance)
            
            beneficio_anual_total += beneficios_diarios["beneficio_total_diario"] * dias_por_estacion
            beneficio_anual_bess += beneficios_diarios["beneficio_bess_diario"] * dias_por_estacion
            beneficio_anual_arbitraje += beneficios_diarios["beneficio_arbitraje_energia_diario"] * dias_por_estacion
            ingreso_anual_ssaa += beneficios_diarios["ingreso_servicios_auxiliares_diario"] * dias_por_estacion
            
            if excel_writer or output_path_graficos:
                df_horario = pd.DataFrame({
                    'Hora': list(model_instance.T),
                    'Potencia_Hidro_Total_Fija_MW': [value(model_instance.p_H_total_fixed[t]) for t in model_instance.T],
                    'Potencia_Hidro_a_Red_MW': [value(model_instance.p_HG_t[t]) for t in model_instance.T],
                    'Potencia_Hidro_a_BESS_MW': [value(model_instance.p_HB_t[t]) for t in model_instance.T],
                    'Potencia_Red_a_BESS_MW': [value(model_instance.p_GB_t[t]) for t in model_instance.T],
                    'Potencia_BESS_a_Red_MW': [value(model_instance.p_D_bht[t]) for t in model_instance.T],
                    'SOC_Total_MWh': [value(model_instance.soc_total_t[t]) for t in model_instance.T],
                    'Precio_Spot_USD_MWh': [value(model_instance.lambda_E[t]) for t in model_instance.T],
                    'Reserva_Up_MW': [value(model_instance.r_B_up_t[t]) for t in model_instance.T],
                    'Reserva_Down_MW': [value(model_instance.r_B_down_t[t]) for t in model_instance.T],
                    'Reserva_Spin_MW': [value(model_instance.s_B_t[t]) for t in model_instance.T],
                    'Energia_Regulacion_Arriba_MWh': [value(model_instance.delta_RU[t] * model_instance.r_B_up_t[t]) for t in model_instance.T],
                    'Energia_Regulacion_Abajo_MWh': [value(model_instance.delta_RD[t] * model_instance.r_B_down_t[t]) for t in model_instance.T],
                })
                seasonal_dfs[season_name] = df_horario
                if excel_writer:
                    df_horario.to_excel(excel_writer, sheet_name=season_name, index=False)
                if output_path_graficos:
                    graficar_resultados_liberalizados(df_horario, bess_data['P_D_max'], bess_data['E_max'], season_name, output_path_graficos)
        else:
            print(f"   -> Advertencia: El modelo para {season_name} fue {results.solver.termination_condition}. Beneficio = 0.")
            
    return {
        "beneficio_anual_total": beneficio_anual_total,
        "beneficio_anual_bess": beneficio_anual_bess,
        "beneficio_anual_arbitraje": beneficio_anual_arbitraje,
        "ingreso_anual_ssaa": ingreso_anual_ssaa,
        "seasonal_dfs": seasonal_dfs
    }

def formatear_y_autoajustar_hojas(writer, dfs_dict, workbook):
    """Aplica formato de número y autoajusta el ancho de las columnas."""
    currency_format = workbook.add_format({'num_format': '$#,##0.00'})
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    percent_format = workbook.add_format({'num_format': '0.00%'})
    
    for sheet_name, df in dfs_dict.items():
        worksheet = writer.sheets[sheet_name]
        for col_num, col_name in enumerate(df.columns):
            max_len = max(df[col_name].astype(str).map(len).max(), len(str(col_name))) + 2
            if any(substring in col_name for substring in ['USD', 'VAN', 'Beneficio', 'Ingreso', 'Costo', 'Flujo', 'CAPEX', 'Valor']):
                worksheet.set_column(col_num, col_num, max_len, currency_format)
            elif 'TIR' in col_name:
                worksheet.set_column(col_num, col_num, max_len, percent_format)
            elif 'MWh' in col_name:
                 worksheet.set_column(col_num, col_num, max_len, number_format)
            else:
                worksheet.set_column(col_num, col_num, max_len)

def crear_grafico_superficie_3d(writer, df_resultados, sheet_name, z_column, title, cell_location, colormap):
    """Crea un gráfico de superficie 3D, marca el punto notable y lo inserta en Excel."""
    worksheet = writer.sheets[sheet_name]
    
    df_plot = df_resultados[['Potencia_MW', 'Energia_MWh', z_column]].dropna()
    if df_plot.shape[0] < 3:
        print(f"  Advertencia: No hay suficientes datos ({df_plot.shape[0]}) para crear el gráfico 3D para {title}.")
        return

    x = df_plot['Potencia_MW']
    y = df_plot['Energia_MWh']
    z = df_plot[z_column]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_trisurf(x, y, z, cmap=colormap, edgecolor='none', antialiased=True, zorder=1)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=f'{z_column}')
    
    if 'TIR' in z_column:
        idx = z.idxmax()
        label_prefix = 'Máximo'
    else:
        idx = z.idxmin()
        label_prefix = 'Mínimo'
        
    point_x = x.loc[idx]
    point_y = y.loc[idx]
    point_z = z.loc[idx]

    ax.scatter(point_x, point_y, point_z, c='red', marker='*', s=250, edgecolor='black', depthshade=False, zorder=10)
    
    text_label = f'  {label_prefix}\n  ({point_x} MW, {point_y} MWh)'
    txt = ax.text(point_x, point_y, point_z, text_label, color='black', fontweight='bold', ha='right', va='center', zorder=10)
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    ax.set_xlabel('Potencia (MW)')
    ax.set_ylabel('Energía (MWh)')
    ax.set_zlabel(z_column)
    ax.set_title(title)
    
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    
    worksheet.insert_image(cell_location, f'{z_column}_surface_plot.png', {'image_data': image_buffer})
    print(f"  Gráfico de superficie 3D para '{title}' insertado en la hoja '{sheet_name}'.")

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    excel_file_name = 'parametros_mercado_liberal.xlsx'
    ruta_excel = os.path.join(script_dir, excel_file_name)
    all_seasonal_data, params_mercado = cargar_datos_mercado_liberalizado(ruta_excel)
    if not all_seasonal_data:
        print("Error fatal: No se pudieron cargar datos. Revisar el archivo Excel.")
        exit()
        
    resultados_finales = []
    print("\n" + "="*80)
    print("INICIANDO CICLO DE OPTIMIZACIÓN (MERCADO LIBERALIZADO - RESERVAS FIJAS)")
    print("="*80)
    
    for potencia in potencias_mw_a_probar:
        for duracion in duraciones_horas_a_probar:
            energia = potencia * duracion
            print(f"\n--- Probando BESS: {potencia} MW / {energia} MWh ({duracion}h) ---")
            bess_data_actual = BESS_PARAMETROS_BASE.copy()
            bess_data_actual.update({'P_C_max': potencia, 'P_D_max': potencia, 'E_max': energia})
            
            resultados_optimizacion = ejecutar_optimizacion_anual_liberalizada(bess_data_actual, all_seasonal_data, params_mercado)
            
            beneficio_anual_bess = resultados_optimizacion["beneficio_anual_bess"]
            
            capex_bess, capex_subestacion = calcular_capex_bess_no_lineal(potencia, energia)
            capex_total = capex_bess + capex_subestacion
            costo_opex_anual_bess = (-0.581594 * potencia**2 + 135.811907 * potencia + 1267.210091) * 1000 * 0.01 + potencia * 8000
            
            if VIDA_UTIL_SUBESTACION_ANOS > VIDA_UTIL_ANOS:
                valor_residual_subestacion = capex_subestacion * (VIDA_UTIL_SUBESTACION_ANOS - VIDA_UTIL_ANOS) / VIDA_UTIL_SUBESTACION_ANOS
            else:
                valor_residual_subestacion = 0

            flujos_bess_sin_pago = [-capex_total]
            for anio in range(1, VIDA_UTIL_ANOS + 1):
                factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
                flujo_anual = (beneficio_anual_bess * factor_degradacion) - costo_opex_anual_bess
                if anio == VIDA_UTIL_ANOS:
                    flujo_anual += valor_residual_subestacion
                flujos_bess_sin_pago.append(flujo_anual)
            
            van_bess_sin_pago = npf.npv(TASA_DESCUENTO, flujos_bess_sin_pago)
            try: tir_bess_sin_pago = npf.irr(flujos_bess_sin_pago)
            except: tir_bess_sin_pago = np.nan

            pago_potencia_req_anual_mw, ingreso_adicional_anual = calcular_pago_potencia_requerido(van_bess_sin_pago, TASA_DESCUENTO, VIDA_UTIL_ANOS, potencia)

            flujos_bess_con_pago = [-capex_total]
            for i in range(1, len(flujos_bess_sin_pago)):
                flujos_bess_con_pago.append(flujos_bess_sin_pago[i] + ingreso_adicional_anual)

            van_bess_con_pago = npf.npv(TASA_DESCUENTO, flujos_bess_con_pago)
            try: tir_bess_con_pago = npf.irr(flujos_bess_con_pago)
            except: tir_bess_con_pago = np.nan

            resultados_finales.append({
                'Potencia_MW': potencia, 'Energia_MWh': energia,
                'VAN_BESS_Sin_Pago_Potencia_USD': van_bess_sin_pago,
                'TIR_BESS_Sin_Pago_Potencia': tir_bess_sin_pago,
                'Pago_Potencia_Requerido_USD_MW_Anio': pago_potencia_req_anual_mw,
                'VAN_BESS_Con_Pago_Potencia_USD': van_bess_con_pago,
                'TIR_BESS_Con_Pago_Potencia': tir_bess_con_pago
            })
            print(f"   -> Pago Potencia Requerido: ${pago_potencia_req_anual_mw:,.2f} /MW-año")

    if not resultados_finales:
        print("No se completó ninguna simulación. No se pueden generar reportes.")
        exit()

    df_resultados_completos = pd.DataFrame(resultados_finales)
    
    excel_resumen_filename = "Resumen_General_Configuraciones.xlsx"
    excel_resumen_filepath = os.path.join(output_dir, excel_resumen_filename)
    with pd.ExcelWriter(excel_resumen_filepath, engine='xlsxwriter') as writer_resumen:
        df_resultados_completos_str = df_resultados_completos.copy()
        df_resultados_completos['TIR_BESS_Sin_Pago_Potencia_Pct'] = df_resultados_completos['TIR_BESS_Sin_Pago_Potencia'] * 100
        
        df_resultados_completos_str['TIR_BESS_Sin_Pago_Potencia'] = df_resultados_completos_str['TIR_BESS_Sin_Pago_Potencia'].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')
        df_resultados_completos_str['TIR_BESS_Con_Pago_Potencia'] = df_resultados_completos_str['TIR_BESS_Con_Pago_Potencia'].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')
        df_resultados_completos_str.to_excel(writer_resumen, sheet_name='Resumen_Configuraciones', index=False)
        
        workbook_resumen = writer_resumen.book
        formatear_y_autoajustar_hojas(writer_resumen, {'Resumen_Configuraciones': df_resultados_completos_str}, workbook_resumen)
        
        crear_grafico_superficie_3d(writer_resumen, df_resultados_completos, 'Resumen_Configuraciones', 'TIR_BESS_Sin_Pago_Potencia_Pct', 'Superficie de TIR del BESS (Sin Pago Potencia, %)', 'J2', 'viridis')
        crear_grafico_superficie_3d(writer_resumen, df_resultados_completos, 'Resumen_Configuraciones', 'Pago_Potencia_Requerido_USD_MW_Anio', 'Superficie de Pago por Potencia Requerido (USD/MW-año)', 'J30', 'plasma_r')

    print(f"\n✅ Reporte con resumen de todas las configuraciones guardado en: {excel_resumen_filepath}")

    df_resultados_completos['Pago_Potencia_Requerido_USD_MW_Anio'] = pd.to_numeric(df_resultados_completos['Pago_Potencia_Requerido_USD_MW_Anio'])
    optimo_idx = df_resultados_completos['Pago_Potencia_Requerido_USD_MW_Anio'].idxmin()
    configuracion_optima = df_resultados_completos.loc[optimo_idx]
    
    print("\n" + "="*80)
    print("🏆 CONFIGURACIÓN ÓPTIMA ENCONTRADA (Basado en el MENOR pago por potencia requerido) 🏆")
    print(f"   - Potencia: {configuracion_optima['Potencia_MW']:.0f} MW, Energía: {configuracion_optima['Energia_MWh']:.0f} MWh")
    print(f"   - Pago por Potencia Requerido: ${configuracion_optima['Pago_Potencia_Requerido_USD_MW_Anio']:,.2f} /MW-año")
    print("="*80)

    print("\nGenerando reporte detallado y gráficos para la configuración óptima...")
    opt_potencia = configuracion_optima['Potencia_MW']
    opt_energia = configuracion_optima['Energia_MWh']
    excel_optimo_filename = f"Reporte_Optimo_P{int(opt_potencia)}MW_E{int(opt_energia)}MWh.xlsx"
    excel_optimo_filepath = os.path.join(output_dir, excel_optimo_filename)

    with pd.ExcelWriter(excel_optimo_filepath, engine='xlsxwriter') as writer:
        bess_data_optima = BESS_PARAMETROS_BASE.copy()
        bess_data_optima.update({'P_C_max': opt_potencia, 'P_D_max': opt_potencia, 'E_max': opt_energia})
        
        resultados_optimizacion = ejecutar_optimizacion_anual_liberalizada(bess_data_optima, all_seasonal_data, params_mercado, excel_writer=writer, output_path_graficos=output_dir)
        print("  Hojas de operación estacional y gráficos generados.")

        beneficio_anual_bess = resultados_optimizacion["beneficio_anual_bess"]
        beneficio_anual_total = resultados_optimizacion["beneficio_anual_total"]
        beneficio_anual_hidro = beneficio_anual_total - beneficio_anual_bess
        seasonal_dfs = resultados_optimizacion["seasonal_dfs"]

        capex_bess, capex_subestacion = calcular_capex_bess_no_lineal(opt_potencia, opt_energia)
        costo_opex_anual_bess = (-0.581594 * opt_potencia**2 + 135.811907 * opt_potencia + 1267.210091) * 1000 * 0.01 + opt_potencia * 8000
        costo_opex_total_anual = costo_opex_anual_bess + costo_om_hidro_fijo_anual
        
        if VIDA_UTIL_SUBESTACION_ANOS > VIDA_UTIL_ANOS:
            valor_residual_subestacion = capex_subestacion * (VIDA_UTIL_SUBESTACION_ANOS - VIDA_UTIL_ANOS) / VIDA_UTIL_SUBESTACION_ANOS
        else:
            valor_residual_subestacion = 0
        
        pago_potencia_req_anual_mw = configuracion_optima['Pago_Potencia_Requerido_USD_MW_Anio']
        ingreso_adicional_anual = pago_potencia_req_anual_mw * opt_potencia if opt_potencia > 0 else 0

        # --- Flujo de Caja Conjunto ---
        flujos_conjunto_sin_pago = [- (capex_bess + capex_subestacion)]
        flujos_conjunto_con_pago = [- (capex_bess + capex_subestacion)]

        for anio in range(1, VIDA_UTIL_ANOS + 1):
            factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
            beneficio_bess_degradado = beneficio_anual_bess * factor_degradacion
            
            flujo_neto_sin_pago = (beneficio_bess_degradado + beneficio_anual_hidro) - costo_opex_total_anual
            flujo_neto_con_pago = (beneficio_bess_degradado + beneficio_anual_hidro + ingreso_adicional_anual) - costo_opex_total_anual

            if anio == VIDA_UTIL_ANOS:
                flujo_neto_sin_pago += valor_residual_subestacion
                flujo_neto_con_pago += valor_residual_subestacion
            
            flujos_conjunto_sin_pago.append(flujo_neto_sin_pago)
            flujos_conjunto_con_pago.append(flujo_neto_con_pago)

        van_conjunto_sin_pago = npf.npv(TASA_DESCUENTO, flujos_conjunto_sin_pago)
        try: tir_conjunto_sin_pago = npf.irr(flujos_conjunto_sin_pago)
        except: tir_conjunto_sin_pago = np.nan
        
        van_conjunto_con_pago = npf.npv(TASA_DESCUENTO, flujos_conjunto_con_pago)
        try: tir_conjunto_con_pago = npf.irr(flujos_conjunto_con_pago)
        except: tir_conjunto_con_pago = np.nan

        # --- HOJA DE RESUMEN FINANCIERO ---
        df_resumen_financiero = pd.DataFrame({
            'Métrica': [
                'Potencia (MW)', 'Energía (MWh)', 'CAPEX BESS (USD)', 'CAPEX Subestación (USD)', 'CAPEX Total (USD)',
                'VAN BESS (Sin Pago Potencia) (USD)', 'TIR BESS (Sin Pago Potencia)',
                'Pago Potencia Requerido (USD/MW-año)',
                'VAN BESS (Con Pago Potencia) (USD)', 'TIR BESS (Con Pago Potencia)',
                '---',
                'VAN Conjunto (Sin Pago Potencia) (USD)', 'TIR Conjunto (Sin Pago Potencia)',
                'VAN Conjunto (Con Pago Potencia) (USD)', 'TIR Conjunto (Con Pago Potencia)'
            ],
            'Valor': [
                opt_potencia, opt_energia, capex_bess, capex_subestacion, capex_bess + capex_subestacion,
                configuracion_optima['VAN_BESS_Sin_Pago_Potencia_USD'], configuracion_optima['TIR_BESS_Sin_Pago_Potencia'],
                pago_potencia_req_anual_mw,
                configuracion_optima['VAN_BESS_Con_Pago_Potencia_USD'], configuracion_optima['TIR_BESS_Con_Pago_Potencia'],
                '',
                van_conjunto_sin_pago, tir_conjunto_sin_pago,
                van_conjunto_con_pago, tir_conjunto_con_pago
            ]
        })
        df_resumen_financiero.to_excel(writer, sheet_name='Resumen_Financiero', index=False)
        print("  Hojas de datos para el óptimo generadas.")
        
        df_flujo_bess_detallado = pd.DataFrame() # Placeholder, el código original lo llena
        
        workbook  = writer.book
        dfs_to_format = {**seasonal_dfs, 'Resumen_Financiero': df_resumen_financiero}
        formatear_y_autoajustar_hojas(writer, dfs_to_format, workbook)
        print("  Columnas autoajustadas y formato aplicado.")

    print(f"\n✅ Reporte final y detallado para la configuración óptima guardado en: {excel_optimo_filepath}")