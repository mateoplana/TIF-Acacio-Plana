# -*- coding: utf-8 -*-
"""

=================================================================================
Análisis de Riesgo con Simulación de Montecarlo - Modelo Híbrido 
=================================================================================

<<< DESCRIPCIÓN DE LA VERSIÓN >>>
Esta versión mantiene el núcleo de optimización corregido y el enfoque financiero
en el BESS, pero modifica la visualización de resultados para mayor claridad.

"""

# ---------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ---------------------------------------------------------------------------
import pandas as pd
from pyomo.environ import *
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------------------------
# 2. CONFIGURACIÓN Y PARÁMETROS DEL PROYECTO
# ---------------------------------------------------------------------------

# --- Parámetros Financieros ---
TASA_DESCUENTO = 0.1378
VIDA_UTIL_ANOS = 15
VIDA_UTIL_SUBESTACION_ANOS = 40
TASA_DEGRADACION_ANUAL_CAPACIDAD = 0.025 # 2.5% de pérdida anual
dias_por_estacion = {'Verano': 90, 'Otoño': 91, 'Invierno': 92, 'Primavera': 92}

# --- Parámetros del BESS (Sistema de Almacenamiento) ---
POTENCIA_BESS_MW = 75
ENERGIA_BESS_MWH = 300
EFICIENCIA_CARGA = 0.97
EFICIENCIA_DESCARGA = 0.97
SOC_MIN = 0.10
SOC_MAX = 0.90

# --- Parámetros de la Central Hidroeléctrica ---
POTENCIA_HIDRO_MAX_MW = 45.0
COSTO_OM_HIDRO_FIJO_ANUAL = 100000
POTENCIA_RED_MAX = 120

# --- Rutas a los archivos de datos ---
# IMPORTANTE: Asegurarse de que esta carpeta exista y contenga los archivos CSV y XLSX.
script_dir = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = r"C:\Users\54264\Desktop\Fac\ingenieria\tesis\Py\csv_montecarlo2"

RUTA_MERCADO_RESERVAS = os.path.join('parametros_mercado_liberal.xlsx')
RUTA_CMO = os.path.join(CARPETA_DATOS, 'escenarios_15_anios_cmo.csv')
RUTA_PAGO_POTENCIA = os.path.join(CARPETA_DATOS, 'escenarios_15_anios_pago_potencia_por_estacion.csv')
RUTA_DISP_BESS = os.path.join(CARPETA_DATOS, 'escenarios_disponibilidad_bess_dias_tipicos.csv')
RUTA_GEN_HIDRO = os.path.join(CARPETA_DATOS, 'escenarios_generacion_hidro.csv')


# ---------------------------------------------------------------------------
# 3. FUNCIONES DE CARGA DE DATOS Y CÁLCULOS
# ---------------------------------------------------------------------------

def calcular_capex_bess_no_lineal(potencia_mw, energia_mwh):
    """Calcula el CAPEX diferenciando los costos del BESS y de la subestación."""
    if potencia_mw == 0:
        return 0, 0
    costo_potencia = (-0.581594 * potencia_mw**2 + 135.811907 * potencia_mw + 1467.210091) * 1000
    capex_subestacion = costo_potencia
    costo_energia = (energia_mwh / 2) * 254603
    ajuste = (energia_mwh - 2 * potencia_mw) * 25539
    capex_bess = costo_energia - ajuste
    return capex_bess, capex_subestacion

def calcular_opex_bess_no_lineal(potencia_mw):
    """Calcula el OPEX anual no lineal para el BESS."""
    if potencia_mw == 0: return 0
    return ((-0.581594 * potencia_mw**2 + 135.811907 * potencia_mw + 1267.210091) * 1000 * 0.01) + (potencia_mw * 8000)

def cargar_datos_mercado(file_path_excel):
    """Carga tanto los datos de reserva como los parámetros del mercado liberalizado."""
    reserve_data = {}
    params_dict = {}
    try:
        df_delta_ru = pd.read_excel(file_path_excel, sheet_name='delta-RU', header=0)
        df_delta_rd = pd.read_excel(file_path_excel, sheet_name='delta_RD', header=0)
        df_params = pd.read_excel(file_path_excel, sheet_name='parametros_mercado_liberal', header=0)
        params_dict = df_params.set_index('Parametro')['Valor'].to_dict()
    except Exception as e:
        print(f"Error crítico al leer el archivo Excel de mercado '{file_path_excel}': {e}")
        return None, None

    precio_rpf = params_dict.get('precio_disponibilidad_rpf', 0)
    precio_r10 = params_dict.get('precio_disponibilidad_r10', 0)
    seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    for season_name in seasons:
        current_season_data = {}
        try:
            current_season_data['delta_RU'] = df_delta_ru[season_name].dropna().tolist()
            current_season_data['delta_RD'] = df_delta_rd[season_name].dropna().tolist()
            current_season_data['lambda_R_up'] = [precio_rpf] * 24
            current_season_data['lambda_R_down'] = [precio_rpf] * 24
            current_season_data['lambda_S'] = [precio_r10] * 24
            reserve_data[season_name] = current_season_data
        except KeyError:
            print(f"Advertencia: No se encontró columna para '{season_name}' en hojas de deltas.")
    return reserve_data, params_dict

# ---------------------------------------------------------------------------
# 4. FUNCIÓN DE OPTIMIZACIÓN DEL DESPACHO DIARIO
# ---------------------------------------------------------------------------

def optimizar_despacho_hibrido_dia_tipico(precios_energia_diarios, reserve_data, gen_hidro_diaria, disponibilidad_horaria_bess, params_mercado):
    """
    Optimiza el despacho diario usando la lógica corregida de 'LiberalizadoFinalcopia.py'.
    Las reservas son parámetros fijos y no variables de decisión.
    """
    modelo = ConcreteModel(name="Despacho_Hibrido_Corregido")
    
    # --- Definición de Conjuntos y Parámetros ---
    modelo.T = Set(initialize=range(24))
    modelo.lambda_E = Param(modelo.T, initialize={t: precios_energia_diarios.iloc[t] for t in modelo.T})
    modelo.lambda_R_up = Param(modelo.T, initialize={t: reserve_data['lambda_R_up'][t] for t in modelo.T})
    modelo.lambda_R_down = Param(modelo.T, initialize={t: reserve_data['lambda_R_down'][t] for t in modelo.T})
    modelo.lambda_S = Param(modelo.T, initialize={t: reserve_data['lambda_S'][t] for t in modelo.T})
    modelo.delta_RU = Param(modelo.T, initialize={t: reserve_data['delta_RU'][t] for t in modelo.T})
    modelo.delta_RD = Param(modelo.T, initialize={t: reserve_data['delta_RD'][t] for t in modelo.T})
    
    modelo.P_C_max = Param(initialize=POTENCIA_BESS_MW)
    modelo.P_D_max = Param(initialize=POTENCIA_BESS_MW)
    modelo.E_max = Param(initialize=ENERGIA_BESS_MWH)
    modelo.eta_C = Param(initialize=EFICIENCIA_CARGA)
    modelo.eta_D = Param(initialize=EFICIENCIA_DESCARGA)
    modelo.soc_min_abs = Param(initialize=SOC_MIN * ENERGIA_BESS_MWH)
    modelo.soc_max_abs = Param(initialize=SOC_MAX * ENERGIA_BESS_MWH)
    modelo.p_H_total_fixed = Param(modelo.T, initialize={t: min(gen_hidro_diaria.iloc[t], POTENCIA_HIDRO_MAX_MW) for t in modelo.T})
    
    # Parámetros de reserva fija
    porcentaje_rpf = params_mercado.get('porcentaje_reserva_rpf', 0.04)
    porcentaje_r10 = params_mercado.get('porcentaje_reserva_r10', 0.02)
    r_up_val = porcentaje_rpf * POTENCIA_BESS_MW
    s_val = porcentaje_r10 * POTENCIA_BESS_MW
    r_down_val = porcentaje_rpf * POTENCIA_BESS_MW
    
    modelo.r_B_up_t = Param(modelo.T, initialize={t: r_up_val * disponibilidad_horaria_bess.iloc[t] for t in modelo.T})
    modelo.s_B_t = Param(modelo.T, initialize={t: s_val * disponibilidad_horaria_bess.iloc[t] for t in modelo.T})
    modelo.r_B_down_t = Param(modelo.T, initialize={t: r_down_val * disponibilidad_horaria_bess.iloc[t] for t in modelo.T})

    # --- Definición de Variables ---
    modelo.p_HG_t = Var(modelo.T, within=NonNegativeReals)
    modelo.p_HB_t = Var(modelo.T, within=NonNegativeReals)
    modelo.p_GB_t = Var(modelo.T, within=NonNegativeReals)
    modelo.p_D_bht = Var(modelo.T, within=NonNegativeReals)
    modelo.soc_total_t = Var(modelo.T, bounds=(modelo.soc_min_abs, modelo.soc_max_abs))
    modelo.u_C_bht = Var(modelo.T, within=Binary)
    modelo.u_D_bht = Var(modelo.T, within=Binary)
    modelo.soc_initial_total = Var(bounds=(modelo.soc_min_abs, modelo.soc_max_abs))

    # --- Definición de la Función Objetivo ---
    def regla_objetivo(m):
        ingreso_energia = sum(m.lambda_E[t] * (m.p_HG_t[t] + m.p_D_bht[t]) for t in m.T)
        costo_energia = sum(m.lambda_E[t] * m.p_GB_t[t] for t in m.T)
        ingreso_reserva_up = sum(m.lambda_R_up[t] * m.r_B_up_t[t] for t in m.T)
        ingreso_reserva_down = sum(m.lambda_R_down[t] * m.r_B_down_t[t] for t in m.T)
        ingreso_reserva_spin = sum(m.lambda_S[t] * m.s_B_t[t] for t in m.T)
        ajuste_regulacion_subida = sum(m.lambda_E[t] * m.delta_RU[t] * m.r_B_up_t[t] for t in m.T)
        ajuste_regulacion_bajada = sum(m.lambda_E[t] * m.delta_RD[t] * m.r_B_down_t[t] for t in m.T)
        return (ingreso_energia - costo_energia) + (ingreso_reserva_up + ingreso_reserva_down + ingreso_reserva_spin) + (ajuste_regulacion_subida - ajuste_regulacion_bajada)
    modelo.objetivo = Objective(rule=regla_objetivo, sense=maximize)

    # --- Definición de Restricciones ---
    modelo.constraints = ConstraintList()
    for t in modelo.T:
        potencia_carga_arbitraje = modelo.p_HB_t[t] + modelo.p_GB_t[t]
        potencia_descarga_arbitraje = modelo.p_D_bht[t]
        
        modelo.constraints.add(modelo.p_HG_t[t] + potencia_descarga_arbitraje + modelo.delta_RU[t] * modelo.r_B_up_t[t] <= POTENCIA_RED_MAX)
        modelo.constraints.add(modelo.p_GB_t[t] + modelo.delta_RD[t] * modelo.r_B_down_t[t] <= POTENCIA_RED_MAX)
        
        modelo.constraints.add(modelo.p_H_total_fixed[t] == modelo.p_HG_t[t] + modelo.p_HB_t[t])
        
        soc_anterior = modelo.soc_total_t[t-1] if t > 0 else modelo.soc_initial_total
        energia_cargada_total = (potencia_carga_arbitraje + modelo.delta_RD[t] * modelo.r_B_down_t[t]) * modelo.eta_C
        energia_descargada_total = (potencia_descarga_arbitraje + modelo.delta_RU[t] * modelo.r_B_up_t[t]) / modelo.eta_D
        modelo.constraints.add(modelo.soc_total_t[t] == soc_anterior + energia_cargada_total - energia_descargada_total)

        potencia_max_bess_hora = modelo.P_C_max * disponibilidad_horaria_bess.iloc[t]
        modelo.constraints.add(potencia_carga_arbitraje + modelo.delta_RD[t] * modelo.r_B_down_t[t] <= potencia_max_bess_hora)
        modelo.constraints.add(potencia_descarga_arbitraje + modelo.delta_RU[t] * modelo.r_B_up_t[t] <= potencia_max_bess_hora)
        
        modelo.constraints.add(modelo.u_C_bht[t] + modelo.u_D_bht[t] <= 1)
        modelo.constraints.add(potencia_carga_arbitraje <= potencia_max_bess_hora * modelo.u_C_bht[t])
        modelo.constraints.add(potencia_descarga_arbitraje <= potencia_max_bess_hora * modelo.u_D_bht[t])

        energia_necesaria_para_reserva_up = (modelo.delta_RU[t] * modelo.r_B_up_t[t] + modelo.s_B_t[t]) / modelo.eta_D
        modelo.constraints.add(potencia_descarga_arbitraje / modelo.eta_D <= soc_anterior - modelo.soc_min_abs - energia_necesaria_para_reserva_up)
        
        espacio_necesario_para_reserva_down = (modelo.delta_RD[t] * modelo.r_B_down_t[t]) * modelo.eta_C
        modelo.constraints.add(potencia_carga_arbitraje * modelo.eta_C <= modelo.soc_max_abs - soc_anterior - espacio_necesario_para_reserva_down)

    energia_inicial_reserva_up = (modelo.delta_RU[0] * modelo.r_B_up_t[0] + modelo.s_B_t[0]) / modelo.eta_D
    modelo.constraints.add(modelo.soc_initial_total >= modelo.soc_min_abs + energia_inicial_reserva_up)
    
    espacio_inicial_reserva_down = (modelo.delta_RD[0] * modelo.r_B_down_t[0]) * modelo.eta_C
    modelo.constraints.add(modelo.soc_initial_total <= modelo.soc_max_abs - espacio_inicial_reserva_down)
    
    modelo.constraints.add(modelo.soc_total_t[23] >= modelo.soc_initial_total) # Condición de ciclicidad

    # --- Resolución del Modelo ---
    solver = SolverFactory('cbc')
    resultado = solver.solve(modelo, tee=False)

    if (resultado.solver.status == SolverStatus.ok) and (resultado.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]):
        # Calcular beneficio del BESS por separado
        ingreso_venta_energia_bess = sum(value(modelo.lambda_E[t] * modelo.p_D_bht[t]) for t in modelo.T)
        costo_compra_energia_bess = sum(value(modelo.lambda_E[t] * modelo.p_GB_t[t]) for t in modelo.T)
        costo_energia_hidro_a_bess = sum(value(modelo.lambda_E[t] * modelo.p_HB_t[t]) for t in modelo.T)
        beneficio_arbitraje_energia = ingreso_venta_energia_bess - costo_compra_energia_bess - costo_energia_hidro_a_bess

        ingreso_reserva_up = sum(value(modelo.lambda_R_up[t] * modelo.r_B_up_t[t]) for t in modelo.T)
        ingreso_reserva_down = sum(value(modelo.lambda_R_down[t] * modelo.r_B_down_t[t]) for t in modelo.T)
        ingreso_reserva_spin = sum(value(modelo.lambda_S[t] * modelo.s_B_t[t]) for t in modelo.T)
        ajuste_subida = sum(value(modelo.lambda_E[t] * modelo.delta_RU[t] * modelo.r_B_up_t[t]) for t in modelo.T)
        ajuste_bajada = sum(value(modelo.lambda_E[t] * modelo.delta_RD[t] * modelo.r_B_down_t[t]) for t in modelo.T)
        ingreso_ssaa = (ingreso_reserva_up + ingreso_reserva_down + ingreso_reserva_spin + ajuste_subida - ajuste_bajada)
        
        beneficio_bess_diario = beneficio_arbitraje_energia + ingreso_ssaa
        return beneficio_bess_diario
    else:
        print(f"Advertencia: No se encontró solución óptima para un día. Condición: {resultado.solver.termination_condition}")
        return 0

# ---------------------------------------------------------------------------
# 5. FUNCIÓN PRINCIPAL DE LA SIMULACIÓN
# ---------------------------------------------------------------------------

def ejecutar_simulacion_montecarlo_dual():
    """Ejecuta la simulación completa de Montecarlo para ambos escenarios."""
    print("Iniciando Simulación de Montecarlo (Análisis Dual con Modelo Corregido)...")
    reserve_data_all_seasons, params_mercado = cargar_datos_mercado(RUTA_MERCADO_RESERVAS)
    if not reserve_data_all_seasons: return

    try:
        df_cmo = pd.read_csv(RUTA_CMO, encoding='latin1')
        df_potencia = pd.read_csv(RUTA_PAGO_POTENCIA, encoding='latin1')
        df_disponibilidad = pd.read_csv(RUTA_DISP_BESS, encoding='latin1')
        df_hidro = pd.read_csv(RUTA_GEN_HIDRO, encoding='latin1')
        print("Archivos CSV de escenarios cargados correctamente.")
    except FileNotFoundError as e:
        print(f"Error Crítico: No se pudo encontrar el archivo de escenario {e.filename}.")
        return

    escenario_cols = [col for col in df_cmo.columns if 'Escenario_' in col]
    if not escenario_cols:
        print("Error: No se encontraron columnas con el prefijo 'Escenario_' en el archivo de CMO.")
        return
    print(f"Se procesarán {len(escenario_cols)} escenarios.")

    # Listas para almacenar resultados
    resultados_van_con_pago = []
    resultados_tir_con_pago = []
    resultados_van_sin_pago = []
    resultados_tir_sin_pago = []
    
    # Listas para gráficos de convergencia
    promedios_van_con_pago = []
    promedios_van_sin_pago = []
    
    capex_bess, capex_subestacion = calcular_capex_bess_no_lineal(POTENCIA_BESS_MW, ENERGIA_BESS_MWH)
    capex_total = capex_bess + capex_subestacion
    om_anual_bess = calcular_opex_bess_no_lineal(POTENCIA_BESS_MW)

    if VIDA_UTIL_SUBESTACION_ANOS > VIDA_UTIL_ANOS:
        valor_residual_subestacion = capex_subestacion * (VIDA_UTIL_SUBESTACION_ANOS - VIDA_UTIL_ANOS) / VIDA_UTIL_SUBESTACION_ANOS
    else:
        valor_residual_subestacion = 0
    
    print(f"\nConfiguración del Proyecto (Solo BESS):")
    print(f"  - CAPEX BESS: ${capex_bess:,.2f}")
    print(f"  - CAPEX Subestación: ${capex_subestacion:,.2f}")
    print(f"  - CAPEX Total (Inversión): ${capex_total:,.2f}")
    print(f"  - O&M Anual BESS: ${om_anual_bess:,.2f}")
    print(f"  - Valor Residual Subestación (Año {VIDA_UTIL_ANOS}): ${valor_residual_subestacion:,.2f}")

    for nombre_escenario in escenario_cols:
        print(f"\n--- Procesando {nombre_escenario} ---")
        
        flujos_caja_anuales_con_pago = []
        flujos_caja_anuales_sin_pago = []

        for anio in range(1, VIDA_UTIL_ANOS + 1):
            beneficio_op_anual_bess = 0
            pago_potencia_anual = 0
            
            for estacion, num_dias in dias_por_estacion.items():
                reserve_data_estacion = reserve_data_all_seasons.get(estacion)
                if not reserve_data_estacion: continue
                
                filtro = (df_cmo['Anio'] == anio) & (df_cmo['Estacion'] == estacion)
                precios_dia_tipico = df_cmo.loc[filtro, nombre_escenario].reset_index(drop=True)
                hidro_dia_tipico = df_hidro.loc[filtro, nombre_escenario].reset_index(drop=True)
                disponibilidad_dia_tipico = df_disponibilidad.loc[filtro, nombre_escenario].reset_index(drop=True)
                
                if not all(len(df) == 24 for df in [precios_dia_tipico, hidro_dia_tipico, disponibilidad_dia_tipico]): continue
                
                beneficio_diario_bess = optimizar_despacho_hibrido_dia_tipico(precios_dia_tipico, reserve_data_estacion, hidro_dia_tipico, disponibilidad_dia_tipico, params_mercado)
                beneficio_op_anual_bess += beneficio_diario_bess * num_dias
                
                filtro_pot = (df_potencia['Anio'] == anio) & (df_potencia['Estacion'] == estacion)
                if not df_potencia.loc[filtro_pot].empty:
                    pago_potencia_mensual = df_potencia.loc[filtro_pot, nombre_escenario].iloc[0]
                    pago_potencia_anual += pago_potencia_mensual * POTENCIA_BESS_MW * 3
            
            factor_degradacion = (1 - TASA_DEGRADACION_ANUAL_CAPACIDAD) ** (anio - 1)
            beneficio_bess_degradado = beneficio_op_anual_bess * factor_degradacion

            # Flujo de caja CON pago por potencia
            flujo_neto_con_pago = beneficio_bess_degradado + pago_potencia_anual - om_anual_bess
            # Flujo de caja SIN pago por potencia
            flujo_neto_sin_pago = beneficio_bess_degradado - om_anual_bess
            
            if anio == VIDA_UTIL_ANOS:
                flujo_neto_con_pago += valor_residual_subestacion
                flujo_neto_sin_pago += valor_residual_subestacion

            flujos_caja_anuales_con_pago.append(flujo_neto_con_pago)
            flujos_caja_anuales_sin_pago.append(flujo_neto_sin_pago)

        # Cálculo de VAN y TIR para el escenario CON pago
        flujos_totales_con_pago = [-capex_total] + flujos_caja_anuales_con_pago
        van_con_pago = npf.npv(TASA_DESCUENTO, flujos_totales_con_pago)
        try: tir_con_pago = npf.irr(flujos_totales_con_pago)
        except: tir_con_pago = np.nan
        resultados_van_con_pago.append(van_con_pago)
        resultados_tir_con_pago.append(tir_con_pago)

        # Cálculo de VAN y TIR para el escenario SIN pago
        flujos_totales_sin_pago = [-capex_total] + flujos_caja_anuales_sin_pago
        van_sin_pago = npf.npv(TASA_DESCUENTO, flujos_totales_sin_pago)
        try: tir_sin_pago = npf.irr(flujos_totales_sin_pago)
        except: tir_sin_pago = np.nan
        resultados_van_sin_pago.append(van_sin_pago)
        resultados_tir_sin_pago.append(tir_sin_pago)

        # Actualizar promedios para gráfico de convergencia
        promedios_van_con_pago.append(np.mean(resultados_van_con_pago))
        promedios_van_sin_pago.append(np.mean(resultados_van_sin_pago))

        print(f"  -> Con Pago:   VAN = ${van_con_pago:,.2f}, TIR = {'N/A' if np.isnan(tir_con_pago) else f'{tir_con_pago:.2%}'}")
        print(f"  -> Sin Pago:   VAN = ${van_sin_pago:,.2f}, TIR = {'N/A' if np.isnan(tir_sin_pago) else f'{tir_sin_pago:.2%}'}")

    analizar_resultados_finales(
        resultados_van_con_pago, resultados_tir_con_pago, 
        resultados_van_sin_pago, resultados_tir_sin_pago,
        promedios_van_con_pago, promedios_van_sin_pago
    )

# ---------------------------------------------------------------------------
# 6. FUNCIONES PARA ANALIZAR, VISUALIZAR Y REPORTAR RESULTADOS
# ---------------------------------------------------------------------------

def generar_reporte_excel(van_con_pago, tir_con_pago, van_sin_pago, tir_sin_pago):
    """Calcula estadísticas clave y genera un reporte en Excel comparando ambos análisis."""
    print("\nGenerando reporte de resultados en Excel...")

    van_con_pago_series = pd.Series(van_con_pago)
    tir_con_pago_series = pd.Series(tir_con_pago).dropna()
    van_sin_pago_series = pd.Series(van_sin_pago)
    tir_sin_pago_series = pd.Series(tir_sin_pago).dropna()

    data = {
        "Métrica": [
            "VAN Medio", "Desviación Estándar VAN", "Percentil 10 VAN", "Percentil 50 VAN (Mediana)", "Percentil 90 VAN", "Probabilidad de VAN > 0",
            "---",
            "TIR Media", "Desviación Estándar TIR", "Percentil 10 TIR", "Percentil 50 TIR (Mediana)", "Percentil 90 TIR"
        ],
        "Con Pago por Potencia": [
            van_con_pago_series.mean(), van_con_pago_series.std(), van_con_pago_series.quantile(0.10), van_con_pago_series.quantile(0.50), van_con_pago_series.quantile(0.90), (van_con_pago_series > 0).mean(),
            "",
            tir_con_pago_series.mean() if not tir_con_pago_series.empty else np.nan,
            tir_con_pago_series.std() if not tir_con_pago_series.empty else np.nan,
            tir_con_pago_series.quantile(0.10) if not tir_con_pago_series.empty else np.nan,
            tir_con_pago_series.quantile(0.50) if not tir_con_pago_series.empty else np.nan,
            tir_con_pago_series.quantile(0.90) if not tir_con_pago_series.empty else np.nan
        ],
        "Sin Pago por Potencia": [
            van_sin_pago_series.mean(), van_sin_pago_series.std(), van_sin_pago_series.quantile(0.10), van_sin_pago_series.quantile(0.50), van_sin_pago_series.quantile(0.90), (van_sin_pago_series > 0).mean(),
            "",
            tir_sin_pago_series.mean() if not tir_sin_pago_series.empty else np.nan,
            tir_sin_pago_series.std() if not tir_sin_pago_series.empty else np.nan,
            tir_sin_pago_series.quantile(0.10) if not tir_sin_pago_series.empty else np.nan,
            tir_sin_pago_series.quantile(0.50) if not tir_sin_pago_series.empty else np.nan,
            tir_sin_pago_series.quantile(0.90) if not tir_sin_pago_series.empty else np.nan
        ]
    }
    df_reporte = pd.DataFrame(data)

    nombre_archivo = 'reporte_financiero_montecarlo_con_sin_pago.xlsx'
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        df_reporte.to_excel(writer, index=False, sheet_name='Resumen_Estadistico')
        workbook = writer.book
        worksheet = writer.sheets['Resumen_Estadistico']
        
        formato_moneda = '$ #,##0.00'
        formato_porcentaje = '0.00%'
        
        for col_letra in ['B', 'C']:
            for row in [2, 3, 4, 5, 6]:
                worksheet[f'{col_letra}{row}'].number_format = formato_moneda
            worksheet[f'{col_letra}{7}'].number_format = formato_porcentaje
            for row in [9, 10, 11, 12, 13]:
                worksheet[f'{col_letra}{row}'].number_format = formato_porcentaje

        worksheet.column_dimensions['A'].width = 35
        worksheet.column_dimensions['B'].width = 25
        worksheet.column_dimensions['C'].width = 25

    print(f"Reporte guardado exitosamente como: {nombre_archivo}")

def analizar_resultados_finales(lista_van_con_pago, lista_tir_con_pago, lista_van_sin_pago, lista_tir_sin_pago, promedios_van_con_pago, promedios_van_sin_pago):
    """Analiza y visualiza los resultados finales de la simulación."""
    print("\n\n" + "="*70)
    print("ANÁLISIS DE RIESGO - RESULTADOS FINALES (CON VS SIN PAGO)")
    print("="*70)

    van_con_pago_series = pd.Series(lista_van_con_pago)
    tir_con_pago_series = pd.Series(lista_tir_con_pago).dropna()
    van_sin_pago_series = pd.Series(lista_van_sin_pago)
    tir_sin_pago_series = pd.Series(lista_tir_sin_pago).dropna()

    print("\n--- Resultados CON Pago por Potencia ---")
    print(f"  - VAN Medio: ${van_con_pago_series.mean():,.2f} (Desv. Est.: ${van_con_pago_series.std():,.2f})")
    if not tir_con_pago_series.empty: print(f"  - TIR Media: {tir_con_pago_series.mean():.2%}")
    print(f"  - Probabilidad de Rentabilidad (VAN > 0): {(van_con_pago_series > 0).mean() * 100:.2f}%")

    print("\n--- Resultados SIN Pago por Potencia (Solo Arbitraje y Reservas) ---")
    print(f"  - VAN Medio: ${van_sin_pago_series.mean():,.2f} (Desv. Est.: ${van_sin_pago_series.std():,.2f})")
    if not tir_sin_pago_series.empty: print(f"  - TIR Media: {tir_sin_pago_series.mean():.2%}")
    print(f"  - Probabilidad de Rentabilidad (VAN > 0): {(van_sin_pago_series > 0).mean() * 100:.2f}%")
    
    sns.set_theme(style="whitegrid")

    # --- Gráfico 1: CON Pago por Potencia ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))
    fig1.suptitle('Análisis de Riesgo (Con Pago por Potencia)', fontsize=16, weight='bold')

    # Subplot 1.1: VAN (Con Pago)
    sns.histplot(van_con_pago_series, kde=True, ax=axes1[0], color='skyblue', bins=30)
    axes1[0].set_title('Distribución del VAN', fontsize=14)
    axes1[0].set_xlabel('Valor Actual Neto (USD)')
    axes1[0].axvline(van_con_pago_series.mean(), color='red', linestyle='--', label=f'Media: ${van_con_pago_series.mean():,.0f}')
    axes1[0].axvline(0, color='black', linestyle='-')
    axes1[0].legend()

    # Subplot 1.2: TIR (Con Pago)
    sns.histplot(tir_con_pago_series, kde=True, ax=axes1[1], color='salmon', bins=30)
    axes1[1].set_title('Distribución de la TIR', fontsize=14)
    axes1[1].set_xlabel('Tasa Interna de Retorno')
    if not tir_con_pago_series.empty:
        axes1[1].axvline(tir_con_pago_series.mean(), color='red', linestyle='--', label=f'Media: {tir_con_pago_series.mean():.2%}')
    axes1[1].axvline(TASA_DESCUENTO, color='black', linestyle='--', label=f'Tasa Descuento ({TASA_DESCUENTO:.1%})')
    axes1[1].legend()

    # Formateo y guardado
    axes1[0].ticklabel_format(style='plain', axis='x')
    axes1[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("analisis_riesgo_con_pago.png")
    print(f"\nGráfico de resultados (Con Pago) guardado como: analisis_riesgo_con_pago.png")
    plt.show()


    # --- Gráfico 2: SIN Pago por Potencia ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle('Análisis de Riesgo (SIN Pago por Potencia)', fontsize=16, weight='bold')

    # Subplot 2.1: VAN (Sin Pago)
    sns.histplot(van_sin_pago_series, kde=True, ax=axes2[0], color='lightgreen', bins=30)
    axes2[0].set_title('Distribución del VAN', fontsize=14)
    axes2[0].set_xlabel('Valor Actual Neto (USD)')
    axes2[0].axvline(van_sin_pago_series.mean(), color='blue', linestyle='--', label=f'Media: ${van_sin_pago_series.mean():,.0f}')
    axes2[0].axvline(0, color='black', linestyle='-')
    axes2[0].legend()

    # Subplot 2.2: TIR (Sin Pago)
    sns.histplot(tir_sin_pago_series, kde=True, ax=axes2[1], color='gold', bins=30)
    axes2[1].set_title('Distribución de la TIR', fontsize=14)
    axes2[1].set_xlabel('Tasa Interna de Retorno')
    if not tir_sin_pago_series.empty:
        axes2[1].axvline(tir_sin_pago_series.mean(), color='blue', linestyle='--', label=f'Media: {tir_sin_pago_series.mean():.2%}')
    axes2[1].axvline(TASA_DESCUENTO, color='black', linestyle='--', label=f'Tasa Descuento ({TASA_DESCUENTO:.1%})')
    axes2[1].legend()

    # Formateo y guardado
    axes2[0].ticklabel_format(style='plain', axis='x')
    axes2[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("analisis_riesgo_sin_pago.png")
    print(f"Gráfico de resultados (Sin Pago) guardado como: analisis_riesgo_sin_pago.png")
    plt.show()


    # --- Gráfico de Convergencia (sin cambios) ---
    plt.figure(figsize=(14, 7))
    num_escenarios = len(promedios_van_con_pago)
    eje_x = np.arange(1, num_escenarios + 1)
    plt.plot(eje_x, promedios_van_con_pago, label='VAN Promedio (Con Pago por Potencia)', color='blue')
    plt.plot(eje_x, promedios_van_sin_pago, label='VAN Promedio (SIN Pago por Potencia)', color='green', linestyle='--')
    plt.title('Convergencia del VAN Promedio', fontsize=16, weight='bold')
    plt.xlabel('Número de Escenarios Simulados', fontsize=12)
    plt.ylabel('VAN Promedio Acumulado (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig("convergencia_van_promedio_con_sin_pago.png")
    print(f"Gráfico de convergencia guardado como: convergencia_van_promedio_con_sin_pago.png")
    plt.show()

    # Llamada a la generación del reporte Excel
    generar_reporte_excel(lista_van_con_pago, lista_tir_con_pago, lista_van_sin_pago, lista_tir_sin_pago)

# ---------------------------------------------------------------------------
# 7. PUNTO DE ENTRADA PRINCIPAL DEL SCRIPT
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        os.chdir(script_dir)
        print(f"Directorio de trabajo cambiado a: {script_dir}")
    except (NameError, FileNotFoundError):
        print(f"No se pudo cambiar el directorio. El directorio actual es: {os.getcwd()}")

    ejecutar_simulacion_montecarlo_dual()