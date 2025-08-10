# -*- coding: utf-8 -*-
"""
HydroBoost: Modelo de Optimización Anual con Análisis de Sensibilidad (v8 - Final)

Este script realiza un análisis de sensibilidad sobre las variables clave del
proyecto, manteniendo la configuración óptima del BESS como fija.

"""

# Importar las librerías necesarias
import pandas as pd
from pyomo.environ import *
import os
import numpy as np
import numpy_financial as npf

# --- CÓDIGO PARA CAMBIAR EL DIRECTORIO DE TRABAJO AUTOMÁTICAMENTE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_name = 'parametros_mercado_mem.xlsx'
output_dir = os.path.join(script_dir, 'resultados_optimizacion')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# SECCIÓN DE CONFIGURACIÓN FIJA (POST-OPTIMIZACIÓN DE TAMAÑO)
# ==============================================================================
POTENCIA_BESS_MW = 75
DURACION_BESS_HORAS = 4
ENERGIA_BESS_MWH = POTENCIA_BESS_MW * DURACION_BESS_HORAS

TASA_DESCUENTO_BASE = 0.10
VIDA_UTIL_ANOS = 15
VIDA_UTIL_SUBESTACION_ANOS = 40
TASA_DEGRADACION_ANUAL_CAPACIDAD = 0.025
dias_por_estacion = 91

# ==============================================================================
# SECCIÓN DE CONFIGURACIÓN DEL ANÁLISIS DE SENSIBILIDAD
# ==============================================================================
# Para CAPEX, OPEX e Ingresos
SENSITIVITY_RANGE_PERCENT = 25
SENSITIVITY_STEPS = 5

# Rango absoluto para la Tasa de Descuento
DISCOUNT_RATE_RANGE_ABSOLUTE = [0.08, 0.10, 0.12, 0.14, 0.16]


# ==============================================================================
# DATOS DE ENTRADA (ESTÁTICOS)
# ==============================================================================
P_hidro_max_central = 45.0
FACTOR_CAUDAL_POTENCIA = 0.6962

hydrology_data_example = {
    'caudales_verano': [44.067, 42.415, 42.056, 42.013, 42.243, 42.458, 42.803, 42.846, 42.717, 36.052, 31.341, 30.623, 30.364, 30.235, 30.867, 34.400, 43.334, 49.123, 51.536, 52.369, 52.613, 52.900, 52.312, 51.033],
    'caudales_otono': [11.993, 9.882, 9.408, 9.293, 9.322, 9.322, 9.351, 9.307, 9.293, 8.719, 8.417, 8.417, 8.273, 8.216, 8.158, 8.374, 14.967, 18.428, 19.994, 19.979, 19.807, 19.649, 19.577, 19.261],
    'caudales_invierno': [2.959, 2.901, 2.887, 2.887, 2.887, 2.887, 2.887, 2.873, 2.873, 2.873, 2.873, 3.074, 3.074, 3.074, 3.088, 3.059, 3.059, 3.074, 3.074, 3.074, 3.074, 3.174, 3.174, 3.174],
    'caudales_primavera': [30.178, 29.775, 29.603, 29.718, 29.761, 29.847, 29.589, 29.574, 27.951, 16.130, 11.993, 12.094, 12.137, 12.195, 12.238, 12.209, 12.726, 21.373, 33.366, 42.817, 47.026, 48.318, 48.318, 45.733]
}

COSTO_POR_CICLO_BESS = 20
INCENTIVOS_POR_ESTACION = {
    'Verano': range(14, 17), 'Otoño': range(16, 19),
    'Invierno': range(18, 21), 'Primavera': range(16, 19)
}
premio_por_descarga_pico = 5

BESS_PARAMETROS_BASE = {
    'eta_C': 0.97, 'eta_D': 0.97,
    'soc_min': 0.10, 'soc_max': 0.90,
}
potencia_red_max = 120

# ==============================================================================
# DEFINICIÓN DE FUNCIONES DEL MODELO Y ANÁLISIS
# ==============================================================================

def autoajustar_ancho_columna(df, worksheet):
    """Ajusta el ancho de las columnas de una hoja de cálculo específica
    basado en el contenido del DataFrame."""
    for idx, col in enumerate(df.columns):
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),
            len(str(series.name))
        )) + 2
        worksheet.set_column(idx, idx, max_len)

def cargar_datos_mercado(excel_file_name):
    """Carga todos los datos necesarios del mercado desde el archivo Excel."""
    try:
        df_precios = pd.read_excel(excel_file_name, sheet_name='Hoja 2')
        seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
        all_seasonal_data = {}
        for season_name in seasons:
            col_name = next((col for col in df_precios.columns if col.lower() == season_name.lower()), None)
            if col_name:
                all_seasonal_data[season_name] = {'lambda_E': df_precios[col_name].dropna().tolist()}
            else:
                raise ValueError(f"No se encontró la columna de precios para '{season_name}'")

        df_fa = pd.read_excel(excel_file_name, sheet_name='FA')
        fa_anual_dict = {}
        for _, row in df_fa.iterrows():
            periodo_str, factor = str(row['Años']), row['Factor']
            if '-' in periodo_str:
                inicio, fin = map(int, periodo_str.split('-'))
                for anio in range(inicio, fin + 1): fa_anual_dict[anio] = factor
            else:
                fa_anual_dict[int(float(periodo_str))] = factor

        df_params = pd.read_excel(excel_file_name, sheet_name='Hoja 1')
        params_mercado = df_params.set_index('Parametro').to_dict()['Valor']

        print(f"Datos de mercado cargados exitosamente desde '{excel_file_name}'.")
        return all_seasonal_data, fa_anual_dict, params_mercado

    except Exception as e:
        print(f"❌ Error crítico al cargar datos desde Excel: {e}")
        exit()


def definir_parametros_y_variables(model, bess_data, market_data, fixed_hydro_gen):
    """Define los parámetros y variables del modelo de optimización."""
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
    model.soc_bht = Var(model.T, within=NonNegativeReals, bounds=(model.soc_min_abs, model.soc_max_abs))
    model.u_C_bht = Var(model.T, within=Binary)
    model.u_D_bht = Var(model.T, within=Binary)
    model.p_GB_t = Var(model.T, within=NonNegativeReals)
    model.start_C_bht = Var(model.T, within=Binary)
    model.start_D_bht = Var(model.T, within=Binary)
    model.soc_initial_abs = Var(within=NonNegativeReals, bounds=(model.soc_min_abs, model.soc_min_abs * 1.01))
    return model

def definir_restricciones(model):
    """Define las restricciones del modelo de optimización."""
    model.constraints = ConstraintList()
    for t in model.T:
        model.constraints.add(model.p_H_total_fixed[t] == model.p_HG_t[t] + model.p_HB_t[t])
        soc_anterior = model.soc_bht[t-1] * 0.99 if t > 1 else model.soc_initial_abs
        model.constraints.add(model.soc_bht[t] == soc_anterior + (model.p_HB_t[t] + model.p_GB_t[t]) * model.eta_C - (model.p_D_bht[t] / model.eta_D))
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

def definir_objetivo(model, params_mercado, horas_incentivo_estacion):
    """Define la función objetivo del modelo de optimización."""
    ingreso_hidro = sum(model.p_HG_t[t] * model.lambda_E[t] for t in model.T)
    ingreso_bess = sum(model.p_D_bht[t] * params_mercado['PRECIO_ENERGIA_SUMINISTRADA_BESS'] for t in model.T)
    incentivo = sum(model.p_D_bht[t] * premio_por_descarga_pico for t in model.T if t in horas_incentivo_estacion)
    costo_perdidas = sum(model.p_GB_t[t] * ((1 - model.eta_C) + (1 - model.eta_D)) * params_mercado['COSTO_ENERGIA_PERDIDAS_BESS'] for t in model.T)
    costo_ciclos = sum((model.start_C_bht[t] + model.start_D_bht[t]) / 2 * COSTO_POR_CICLO_BESS for t in model.T)
    model.objective = Objective(expr=ingreso_hidro + ingreso_bess + incentivo - costo_perdidas - costo_ciclos, sense=maximize)
    return model

def ejecutar_optimizacion_estacional(bess_data, all_seasonal_data, params_mercado, excel_writer=None):
    """Ejecuta la optimización para todas las estaciones y calcula los beneficios anuales."""
    total_annual_operating_profit = 0
    total_annual_hydro_profit = 0
    season_to_caudal_key = {'Verano': 'caudales_verano', 'Otoño': 'caudales_otono', 'Invierno': 'caudales_invierno', 'Primavera': 'caudales_primavera'}
    for season_name in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
        caudales = hydrology_data_example[season_to_caudal_key[season_name]]
        fixed_hydro_gen = [min(c * FACTOR_CAUDAL_POTENCIA, P_hidro_max_central) for c in caudales]
        model = ConcreteModel(name=f"Hydro_{season_name}")
        model = definir_parametros_y_variables(model, bess_data, all_seasonal_data[season_name], fixed_hydro_gen)
        model = definir_objetivo(model, params_mercado, INCENTIVOS_POR_ESTACION[season_name])
        model = definir_restricciones(model)
        solver = SolverFactory('cbc')
        results = solver.solve(model, tee=False)
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]):
            daily_profit = value(model.objective)
            daily_hydro_profit = sum(value(model.p_HG_t[t]) * value(model.lambda_E[t]) for t in model.T)
            costo_ciclos_del_dia_usd = sum((value(model.start_C_bht[t]) + value(model.start_D_bht[t])) / 2 * COSTO_POR_CICLO_BESS for t in model.T)
            premio_pico_del_dia_usd = sum(value(model.p_D_bht[t]) * premio_por_descarga_pico for t in model.T if t in INCENTIVOS_POR_ESTACION[season_name])
            beneficio_diario_para_reporte = daily_profit + costo_ciclos_del_dia_usd - premio_pico_del_dia_usd
            total_annual_operating_profit += beneficio_diario_para_reporte * dias_por_estacion
            total_annual_hydro_profit += daily_hydro_profit * dias_por_estacion
            if excel_writer:
                df_horario = pd.DataFrame({'Hora': list(range(1, 25)), 'Potencia_Hidro_a_Red_MW': [value(model.p_HG_t[t]) for t in model.T], 'Potencia_BESS_a_Red_MW': [value(model.p_D_bht[t]) for t in model.T], 'SOC_MWh': [value(model.soc_bht[t]) for t in model.T], 'Precio_Spot_USD_MWh': [value(model.lambda_E[t]) for t in model.T]})
                df_horario.to_excel(excel_writer, sheet_name=season_name, index=False)
                autoajustar_ancho_columna(df_horario, excel_writer.sheets[season_name])
        else:
            print(f"   -> Advertencia: El modelo para {season_name} no encontró una solución óptima.")
    return total_annual_operating_profit, total_annual_hydro_profit

def calcular_capex_bess_no_lineal(potencia_mw, energia_mwh, capex_multiplier=1.0):
    """Calcula el CAPEX para el BESS y la subestación, aplicando un multiplicador."""
    if potencia_mw == 0:
        return 0, 0
    costo_potencia = (-0.581594 * potencia_mw**2 + 135.811907 * potencia_mw + 1467.210091) * 1000
    capex_subestacion = costo_potencia * capex_multiplier
    costo_energia = (energia_mwh / 2) * 254603
    ajuste = (energia_mwh - 2 * potencia_mw) * 25539
    capex_bess = (costo_energia - ajuste) * capex_multiplier
    return capex_bess, capex_subestacion

def calcular_metricas_financieras(potencia_mw, energia_mwh, beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, fa_anual_dict, params_mercado, costo_om_hidro, tasa_descuento, capex_multiplier=1.0):
    """Calcula el VAN, TIR y flujos de caja detallados para los escenarios combinado y solo BESS."""
    capex_bess, capex_subestacion = calcular_capex_bess_no_lineal(potencia_mw, energia_mwh, capex_multiplier)
    capex_total = capex_bess + capex_subestacion
    if VIDA_UTIL_SUBESTACION_ANOS > VIDA_UTIL_ANOS:
        valor_residual_subestacion = capex_subestacion * (VIDA_UTIL_SUBESTACION_ANOS - VIDA_UTIL_ANOS) / VIDA_UTIL_SUBESTACION_ANOS
    else:
        valor_residual_subestacion = 0
    beneficio_op_anual_solo_bess = beneficio_op_anual_combinado - beneficio_op_anual_solo_hidro
    costo_opex_sub_BESS = (-0.581594 * potencia_mw**2 + 135.811907 * potencia_mw + 1267.210091) * 1000 * 0.01 + potencia_mw * 8000
    ingreso_potencia_hidro_anual = params_mercado['HIDRO_REM_POT_PRECIO_BASE'] * P_hidro_max_central * 12
    valor_ofertado_bess = params_mercado['VALOR_OFERTADO_BESS']
    flujos_combinado = [-capex_total]
    flujos_bess = [-capex_total]
    detalle_flujo_combinado = []
    detalle_flujo_bess = []
    for anio in range(1, VIDA_UTIL_ANOS + 1):
        factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
        beneficio_op_degradado_bess = beneficio_op_anual_solo_bess * factor_degradacion
        fa_del_anio = fa_anual_dict.get(anio, 1.0)
        ingreso_potencia_bess_anual = valor_ofertado_bess * fa_del_anio * potencia_mw * 12
        costo_opex_anual_total = costo_om_hidro + costo_opex_sub_BESS
        beneficio_total_anual_degradado = beneficio_op_anual_solo_hidro + beneficio_op_degradado_bess
        flujo_anual_combinado = (beneficio_total_anual_degradado + ingreso_potencia_hidro_anual + ingreso_potencia_bess_anual - costo_opex_anual_total)
        flujo_anual_bess = (beneficio_op_degradado_bess + ingreso_potencia_bess_anual - costo_opex_sub_BESS)
        valor_residual_del_anio = 0
        if anio == VIDA_UTIL_ANOS:
            flujo_anual_combinado += valor_residual_subestacion
            flujo_anual_bess += valor_residual_subestacion
            valor_residual_del_anio = valor_residual_subestacion
        flujos_combinado.append(flujo_anual_combinado)
        van_acumulado_comb = npf.npv(tasa_descuento, flujos_combinado)
        tir_acumulada_comb = npf.irr(flujos_combinado)
        detalle_flujo_combinado.append({'Año': anio, 'Beneficio Operativo (USD)': beneficio_total_anual_degradado, 'Ingreso Potencia Hidro (USD)': ingreso_potencia_hidro_anual, 'Ingreso Potencia BESS (USD)': ingreso_potencia_bess_anual, 'Costo OPEX Total (USD)': costo_opex_anual_total, 'Valor Residual Subestación (USD)': valor_residual_del_anio, 'Flujo de Caja Neto (USD)': flujo_anual_combinado, 'VAN Acumulado (USD)': van_acumulado_comb, 'TIR Acumulada': tir_acumulada_comb})
        flujos_bess.append(flujo_anual_bess)
        van_acumulado_bess = npf.npv(tasa_descuento, flujos_bess)
        tir_acumulada_bess = npf.irr(flujos_bess)
        detalle_flujo_bess.append({'Año': anio, 'Beneficio Operativo BESS (USD)': beneficio_op_degradado_bess, 'Ingreso Potencia BESS (USD)': ingreso_potencia_bess_anual, 'Costo OPEX BESS (USD)': costo_opex_sub_BESS, 'Valor Residual Subestación (USD)': valor_residual_del_anio, 'Flujo de Caja Neto (USD)': flujo_anual_bess, 'VAN Acumulado (USD)': van_acumulado_bess, 'TIR Acumulada': tir_acumulada_bess})
    van_combinado_final = npf.npv(tasa_descuento, flujos_combinado)
    tir_combinado_final = npf.irr(flujos_combinado)
    van_bess_final = npf.npv(tasa_descuento, flujos_bess)
    tir_bess_final = npf.irr(flujos_bess)
    df_flujo_combinado = pd.DataFrame(detalle_flujo_combinado)
    df_flujo_bess = pd.DataFrame(detalle_flujo_bess)
    return van_combinado_final, tir_combinado_final, van_bess_final, tir_bess_final, capex_bess, capex_subestacion, df_flujo_combinado, df_flujo_bess

def agregar_grafico_a_excel(writer, sheet_name, x_range, van_comb_range, tir_comb_range, van_bess_range, tir_bess_range, title, x_label, start_cell):
    """Añade un gráfico de sensibilidad con doble eje Y (VAN y TIR) a una hoja de Excel."""
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    van_chart = workbook.add_chart({'type': 'line'})
    van_chart.set_title({'name': title})
    van_chart.set_x_axis({'name': x_label, 'major_gridlines': {'visible': True}})
    van_chart.set_y_axis({'name': 'VAN (USD)', 'major_gridlines': {'visible': True}})
    van_chart.set_size({'width': 720, 'height': 432})
    van_chart.add_series({'name': f"='{sheet_name}'!$B$1", 'categories': f"='{sheet_name}'!{x_range}", 'values': f"='{sheet_name}'!{van_comb_range}"})
    van_chart.add_series({'name': f"='{sheet_name}'!$D$1", 'categories': f"='{sheet_name}'!{x_range}", 'values': f"='{sheet_name}'!{van_bess_range}"})
    tir_chart = workbook.add_chart({'type': 'line'})
    tir_chart.add_series({'name': f"='{sheet_name}'!$C$1", 'categories': f"='{sheet_name}'!{x_range}", 'values': f"='{sheet_name}'!{tir_comb_range}", 'y2_axis': True})
    tir_chart.add_series({'name': f"='{sheet_name}'!$E$1", 'categories': f"='{sheet_name}'!{x_range}", 'values': f"='{sheet_name}'!{tir_bess_range}", 'y2_axis': True})
    van_chart.combine(tir_chart)
    van_chart.set_y2_axis({'name': 'TIR (%)', 'num_format': '0.0%'})
    worksheet.insert_chart(start_cell, van_chart)


# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    ruta_excel = os.path.join(script_dir, excel_file_name)
    all_seasonal_data, fa_anual_dict, params_mercado_base = cargar_datos_mercado(ruta_excel)

    bess_data_config = BESS_PARAMETROS_BASE.copy()
    bess_data_config.update({'P_C_max': POTENCIA_BESS_MW, 'P_D_max': POTENCIA_BESS_MW, 'E_max': ENERGIA_BESS_MWH})

    excel_output_filename = f"Analisis_Sensibilidad_P{POTENCIA_BESS_MW}MW_E{ENERGIA_BESS_MWH}.xlsx"
    excel_output_filepath = os.path.join(output_dir, excel_output_filename)

    with pd.ExcelWriter(excel_output_filepath, engine='xlsxwriter') as writer:
        print("\n" + "="*80 + f"\nEjecutando simulación base para BESS: {POTENCIA_BESS_MW} MW / {ENERGIA_BESS_MWH} MWh\n" + "="*80)
        beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro = ejecutar_optimizacion_estacional(bess_data_config, all_seasonal_data, params_mercado_base, writer)
        costo_om_hidro_base = params_mercado_base.get('COSTO_OM_HIDRO_FIJO', 100000)
        
        van_comb_base, tir_comb_base, van_bess_base, tir_bess_base, capex_bess_base, capex_sub_base, df_flujo_comb, df_flujo_bess = calcular_metricas_financieras(
            POTENCIA_BESS_MW, ENERGIA_BESS_MWH, beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, 
            fa_anual_dict, params_mercado_base, costo_om_hidro_base, TASA_DESCUENTO_BASE
        )
        
        capex_total_base = capex_bess_base + capex_sub_base
        df_resumen_base = pd.DataFrame({'Métrica': ['VAN Combinado (USD)', 'TIR Combinado', 'VAN BESS (USD)', 'TIR BESS', 'CAPEX BESS (USD)', 'CAPEX Subestación (USD)', 'CAPEX Total (USD)'], 'Valor': [van_comb_base, f"{tir_comb_base:.2%}", van_bess_base, f"{tir_bess_base:.2%}", capex_bess_base, capex_sub_base, capex_total_base]})
        df_resumen_base.to_excel(writer, sheet_name='Resumen_Caso_Base', index=False)
        autoajustar_ancho_columna(df_resumen_base, writer.sheets['Resumen_Caso_Base'])
        
        df_flujo_comb.to_excel(writer, sheet_name='Flujo_Caja_Combinado', index=False)
        autoajustar_ancho_columna(df_flujo_comb, writer.sheets['Flujo_Caja_Combinado'])
        
        df_flujo_bess.to_excel(writer, sheet_name='Flujo_Caja_BESS', index=False)
        autoajustar_ancho_columna(df_flujo_bess, writer.sheets['Flujo_Caja_BESS'])
        
        print("Caso base y flujos de caja detallados guardados en Excel.")
        print(f"  -> VAN BESS (Base): ${van_bess_base:,.2f}")

        print("\n" + "="*80 + "\nINICIANDO ANÁLISIS DE SENSIBILIDAD\n" + "="*80)
        sensibilidad_results = {}
        variacion_min = 1 - (SENSITIVITY_RANGE_PERCENT / 100)
        variacion_max = 1 + (SENSITIVITY_RANGE_PERCENT / 100)
        variacion_porcentual = np.linspace(variacion_min, variacion_max, SENSITIVITY_STEPS)

        # --- ANÁLISIS DE SENSIBILIDAD ---
        print("Analizando sensibilidad a: CAPEX del BESS...")
        resultados_capex = []
        for mult in variacion_porcentual:
            van_c, tir_c, van_b, tir_b, _, _, _, _ = calcular_metricas_financieras(POTENCIA_BESS_MW, ENERGIA_BESS_MWH, beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, fa_anual_dict, params_mercado_base, costo_om_hidro_base, TASA_DESCUENTO_BASE, capex_multiplier=mult)
            resultados_capex.append({'Variacion_CAPEX': f"{mult:.0%}", 'VAN_Combinado': van_c, 'TIR_Combinado': tir_c, 'VAN_BESS': van_b, 'TIR_BESS': tir_b})
        sensibilidad_results['CAPEX'] = pd.DataFrame(resultados_capex)

        print("Analizando sensibilidad a: VALOR_OFERTADO_BESS...")
        resultados_valor_ofertado = []
        valor_ofertado_base = params_mercado_base['VALOR_OFERTADO_BESS']
        for mult in variacion_porcentual:
            params_modificados = params_mercado_base.copy()
            params_modificados['VALOR_OFERTADO_BESS'] = valor_ofertado_base * mult
            van_c, tir_c, van_b, tir_b, _, _, _, _ = calcular_metricas_financieras(POTENCIA_BESS_MW, ENERGIA_BESS_MWH, beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, fa_anual_dict, params_modificados, costo_om_hidro_base, TASA_DESCUENTO_BASE)
            resultados_valor_ofertado.append({'Valor_Ofertado_BESS': params_modificados['VALOR_OFERTADO_BESS'], 'VAN_Combinado': van_c, 'TIR_Combinado': tir_c, 'VAN_BESS': van_b, 'TIR_BESS': tir_b})
        sensibilidad_results['VALOR_OFERTADO_BESS'] = pd.DataFrame(resultados_valor_ofertado)

        print("Analizando sensibilidad a: Costo O&M Hidroeléctrica...")
        resultados_om_hidro = []
        for mult in variacion_porcentual:
            costo_om_modificado = costo_om_hidro_base * mult
            van_c, tir_c, van_b, tir_b, _, _, _, _ = calcular_metricas_financieras(POTENCIA_BESS_MW, ENERGIA_BESS_MWH, beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, fa_anual_dict, params_mercado_base, costo_om_modificado, TASA_DESCUENTO_BASE)
            resultados_om_hidro.append({'Costo_OM_Hidro': costo_om_modificado, 'VAN_Combinado': van_c, 'TIR_Combinado': tir_c, 'VAN_BESS': van_b, 'TIR_BESS': tir_b})
        sensibilidad_results['OM_HIDRO'] = pd.DataFrame(resultados_om_hidro)
        
        print("Analizando sensibilidad a: Tasa de Descuento...")
        resultados_tasa_desc = []
        for tasa_modificada in DISCOUNT_RATE_RANGE_ABSOLUTE:
            van_c, tir_c, van_b, tir_b, _, _, _, _ = calcular_metricas_financieras(POTENCIA_BESS_MW, ENERGIA_BESS_MWH, beneficio_op_anual_combinado, beneficio_op_anual_solo_hidro, fa_anual_dict, params_mercado_base, costo_om_hidro_base, tasa_modificada)
            resultados_tasa_desc.append({'Tasa_Descuento': f"{tasa_modificada:.1%}", 'VAN_Combinado': van_c, 'TIR_Combinado': tir_c, 'VAN_BESS': van_b, 'TIR_BESS': tir_b})
        sensibilidad_results['TASA_DESC'] = pd.DataFrame(resultados_tasa_desc)

        # --- GUARDAR RESULTADOS DE SENSIBILIDAD Y GRÁFICOS EN EXCEL ---
        sheet_name = 'Analisis_Sensibilidad'
        
        # CAPEX
        df_capex = sensibilidad_results['CAPEX']
        df_capex.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        agregar_grafico_a_excel(writer, sheet_name, f'A2:A{1+SENSITIVITY_STEPS}', f'B2:B{1+SENSITIVITY_STEPS}', f'C2:C{1+SENSITIVITY_STEPS}', f'D2:D{1+SENSITIVITY_STEPS}', f'E2:E{1+SENSITIVITY_STEPS}', 'Sensibilidad al CAPEX del BESS', 'Variación del CAPEX', 'G2')
        
        # VALOR OFERTADO
        start_row = len(df_capex) + 3
        df_valor = sensibilidad_results['VALOR_OFERTADO_BESS']
        df_valor.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        agregar_grafico_a_excel(writer, sheet_name, f'A{start_row+2}:A{start_row+1+SENSITIVITY_STEPS}', f'B{start_row+2}:B{start_row+1+SENSITIVITY_STEPS}', f'C{start_row+2}:C{start_row+1+SENSITIVITY_STEPS}', f'D{start_row+2}:D{start_row+1+SENSITIVITY_STEPS}', f'E{start_row+2}:E{start_row+1+SENSITIVITY_STEPS}', 'Sensibilidad al Valor Ofertado del BESS', 'Valor Ofertado (USD/MW-mes)', 'G25')

        # O&M HIDRO
        start_row += len(df_valor) + 3
        df_om = sensibilidad_results['OM_HIDRO']
        df_om.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        agregar_grafico_a_excel(writer, sheet_name, f'A{start_row+2}:A{start_row+1+SENSITIVITY_STEPS}', f'B{start_row+2}:B{start_row+1+SENSITIVITY_STEPS}', f'C{start_row+2}:C{start_row+1+SENSITIVITY_STEPS}', f'D{start_row+2}:D{start_row+1+SENSITIVITY_STEPS}', f'E{start_row+2}:E{start_row+1+SENSITIVITY_STEPS}', 'Sensibilidad al Costo O&M Hidro', 'Costo O&M Anual (USD)', 'G48')

        # TASA DE DESCUENTO
        start_row += len(df_om) + 3
        df_tasa = sensibilidad_results['TASA_DESC']
        df_tasa.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        agregar_grafico_a_excel(writer, sheet_name, f'A{start_row+2}:A{start_row+len(DISCOUNT_RATE_RANGE_ABSOLUTE)+1}', f'B{start_row+2}:B{start_row+len(DISCOUNT_RATE_RANGE_ABSOLUTE)+1}', f'C{start_row+2}:C{start_row+len(DISCOUNT_RATE_RANGE_ABSOLUTE)+1}', f'D{start_row+2}:D{start_row+len(DISCOUNT_RATE_RANGE_ABSOLUTE)+1}', f'E{start_row+2}:E{start_row+len(DISCOUNT_RATE_RANGE_ABSOLUTE)+1}', 'Sensibilidad a la Tasa de Descuento', 'Tasa de Descuento', 'G71')

        autoajustar_ancho_columna(pd.concat([df_capex, df_valor, df_om, df_tasa]), writer.sheets[sheet_name])

        print(f"\nAnálisis de sensibilidad completado. Resultados guardados en '{excel_output_filename}'.")
        print("\n" + "="*80 + "\nPROCESO FINALIZADO\n" + "="*80)