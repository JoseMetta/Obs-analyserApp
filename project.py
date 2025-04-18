import streamlit as st
import os
import subprocess
import numpy as np
import gnsspy as gp
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from streamlit_option_menu import option_menu

# Configuración de la página	
st.set_page_config(page_title="OBS-ANALYSER", page_icon="📡", layout="wide")

# =============== Configurar el menú lateral ==============
with st.sidebar:
    selected = option_menu(
        menu_title="Menú",
        options=["OBS-GNSS_ANALYSER", "GNSS_Graph"],
        icons=["arrow-right-circle", "arrow-right-circle"],
        menu_icon="cast",
        default_index=0,
    )



print("Python executable in use:", sys.executable)


# ====================================== Query Selection ===========================================

#if selected == "OBS-GNSS_ANALYSER":

def run_obs_analyses():
        st.title("Conversión de UBX a RINEX Observación con convbin")
        st.markdown("""
        Esta aplicación utiliza **convbin.exe** para convertir un archivo UBX a formato RINEX de observación.
        """)

        usar_nombres_automaticos = st.checkbox("Usar nombres automáticos para archivos de salida", value=True)

        uploaded_file = st.file_uploader("Sube tu archivo UBX", type=["ubx"])
        destination_path = st.text_input("Ruta para guardar el archivo (sin caracteres especiales):", "")

        if uploaded_file is not None:
            file_base = os.path.splitext(uploaded_file.name)[0]

            if usar_nombres_automaticos:
                obs_file = f"{file_base}.obs"
                nav_file = f"{file_base}.nav"
                st.info(f"Nombres generados automáticamente:\n📄 OBS: `{obs_file}`\n📄 NAV: `{nav_file}`")
            else:
                obs_file = st.text_input("Archivo de salida (obs):", f"{file_base}.obs")
                nav_file = st.text_input("Archivo de navegación (nav):", f"{file_base}.nav")

            if destination_path.strip() == "":
                st.error("⚠️ Debes ingresar una ruta válida para guardar los archivos.")
                return None

            os.makedirs(destination_path, exist_ok=True)
            input_path = os.path.join(destination_path, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Archivo UBX guardado en: {input_path}")

            if st.button("Convertir a RINEX Observación"):
                cwd = os.path.dirname(input_path)
                st.write("Directorio de trabajo:", cwd)

                command = [
                    "convbin.exe", "-r", "ubx", "-hm", "-ho", "-hr", "-ha",
                    "-v", "3.02", "-od", "-oi", "-ot", "-ol", "-os",
                    "-halfc", "-trace", "level", "2",
                    "-n", nav_file, "-o", obs_file,
                    input_path
                ]
                st.write("Ejecutando el siguiente comando:")
                st.code(" ".join(command))

                try:
                    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
                    st.success("¡Conversión exitosa!")
                    obs_file_path = os.path.join(cwd, obs_file)
                    nav_file_path = os.path.join(cwd, nav_file)

                    if os.path.exists(obs_file_path):
                        with open(obs_file_path, "rb") as f:
                            st.download_button("Descargar archivo OBS", f, file_name=obs_file)
                    else:
                        st.error("No se encontró el archivo de salida OBS.")



                    obs_file_path = os.path.join(destination_path, obs_file)
                    if os.path.exists(obs_file_path):
                        station = gp.read_obsFile(obs_file_path)
                        df = station.observation
                        st.dataframe(df)

                        # Filtrar solo las observaciones del sistema GPS
                        df_gps = df[df["SYSTEM"] == "GPS"]
                        df_glonass = df[df["SYSTEM"] == "GLONASS"]
                        df_galileo = df[df["SYSTEM"] == "GALILEO"]
                        df_beidou = df[df["SYSTEM"] == "COMPASS"]
                        df_sbas = df[df["SYSTEM"] == "SBAS"]

                        df_gps = df_gps.reset_index()
                        df_glonass = df_glonass.reset_index()
                        df_galileo = df_galileo.reset_index()
                        df_beidou = df_beidou.reset_index()
                        df_sbas = df_sbas.reset_index()

                        # Selecciona solo las columnas deseadas
                        df_gps = df_gps[['Epoch','SV','C1C', 'L1C', 'D1C', 'S1C', 'C2X', 'L2X', 'D2X', 'S2X']]
                        df_glonass = df_glonass[['Epoch','SV','C1C', 'L1C', 'D1C', 'S1C', 'C2C', 'L2C', 'D2C', 'S2C']]
                        df_galileo = df_galileo[['Epoch','SV','C1X', 'L1X', 'D1X', 'S1X', 'C7X', 'L7X', 'D7X', 'S7X']]
                        df_beidou = df_beidou[['Epoch','SV','C1I', 'L1I', 'D1I', 'S1I']]    #'L7I' 'C7I', 'D7I', 'S7I'
                        df_sbas = df_sbas[['Epoch','SV','C1C', 'L1C', 'D1C', 'S1C']]

                        # Crear una nueva columna "FREQ" inicialmente llena de 0
                        df_gps['FREQ'] = 0

                        # Iterar sobre las filas del DataFrame
                        for index, row in df_gps.iterrows():
                            if pd.notna(row['L1C']) and pd.isna(row['L2X']):
                                df_gps.at[index, 'FREQ'] = 1
                            elif pd.notna(row['L1C']) and pd.notna(row['L2X']):
                                df_gps.at[index, 'FREQ'] = 2

                        print(df_gps.head(10))  

                                    
                        CLIGHT = 299792458.0
                        FREQ1 = 1575.42e6
                        FREQ2 = 1227.60e6

                        def read_rinex_obs_fixed_patch(rinex_path):
                            obs_data = []
                            with open(rinex_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()

                            gps_obs_types = []
                            header_end = 0
                            for i, line in enumerate(lines):
                                if "SYS / # / OBS TYPES" in line and line.startswith("G"):
                                    parts = line[6:60].split()
                                    gps_obs_types.extend(parts)
                                    while len(gps_obs_types) < int(line[3:6]):
                                        i += 1
                                        gps_obs_types.extend(lines[i][6:60].split())
                                if "END OF HEADER" in line:
                                    header_end = i + 1
                                    break

                            try:
                                idx_C1C = gps_obs_types.index("C1C")
                                idx_L1C = gps_obs_types.index("L1C")
                                idx_C2X = gps_obs_types.index("C2X")
                                idx_L2X = gps_obs_types.index("L2X")
                            except ValueError:
                                return []

                            i = header_end
                            while i < len(lines):
                                line = lines[i]
                                if not line.startswith('>'):
                                    i += 1
                                    continue
                                time_str = line[2:29].strip()
                                num_sats = int(line[32:35])
                                i += 1

                                for _ in range(num_sats):
                                    if i >= len(lines): break
                                    sat_line = lines[i]
                                    sat_id = sat_line[0:3]
                                    if not sat_id.startswith("G"):
                                        i += 1
                                        continue

                                    n_obs = len(gps_obs_types)
                                    obs_lines = sat_line[3:].rstrip('\n')
                                    obs_vals = [obs_lines[j:j+16] for j in range(0, len(obs_lines), 16)]

                                    while len(obs_vals) < n_obs:
                                        i += 1
                                        extra = lines[i].rstrip('\n')
                                        obs_vals.extend([extra[j:j+16] for j in range(0, len(extra), 16)])

                                    def get_val(idx):
                                        if idx >= len(obs_vals):
                                            return None, 0
                                        val_str = obs_vals[idx][:14].replace('D', 'E').strip()
                                        lli_str = obs_vals[idx][14:15].strip()
                                        if not val_str:
                                            return None, 0
                                        try:
                                            val = float(val_str)
                                        except ValueError:
                                            return None, 0
                                        lli = int(lli_str) if lli_str.isdigit() else 0
                                        return val, lli

                                    P1, _ = get_val(idx_C1C)
                                    L1, LLI1 = get_val(idx_L1C)
                                    P2, _ = get_val(idx_C2X)
                                    L2, LLI2 = get_val(idx_L2X)

                                    if None not in (P1, L1, P2, L2):
                                        obs_data.append((time_str, sat_id, P1, L1, LLI1, P2, L2, LLI2))
                                    i += 1

                            return obs_data

                        def compute_mp_rtklib_like(obs_data):
                            obs_data.sort(key=lambda x: (x[1], x[0]))
                            mp_results = {}
                            for time, sat, P1, L1, LLI1, P2, L2, LLI2 in obs_data:
                                if sat not in mp_results:
                                    mp_results[sat] = []
                                if any(v == 0.0 for v in (L1, L2, P1, P2)):
                                    mp_results[sat].append((time, None, None))
                                    continue
                                I = -CLIGHT * ((L1 / FREQ1) - (L2 / FREQ2)) / (1.0 - (FREQ1/FREQ2)**2)
                                MP1 = P1 - (CLIGHT * L1 / FREQ1) - 2.0 * I
                                MP2 = P2 - (CLIGHT * L2 / FREQ2) - 2.0 * ((FREQ1/FREQ2)**2) * I
                                mp_results[sat].append([time, MP1, MP2, LLI1, LLI2])

                            for sat, data in mp_results.items():
                                B1 = B2 = 0.0
                                n1 = n2 = 0
                                arc_start = 0
                                for i in range(len(data)):
                                    time, MP1, MP2, LLI1, LLI2 = data[i]
                                    if (LLI1 & 1) or (LLI2 & 1) or (n1 > 0 and MP1 is not None and abs(MP1 - B1) > 5.0) or (n2 > 0 and MP2 is not None and abs(MP2 - B2) > 5.0):
                                        for j in range(arc_start, i):
                                            if data[j][1] is not None:
                                                data[j][1] -= B1
                                            if data[j][2] is not None:
                                                data[j][2] -= B2
                                        B1 = B2 = 0.0
                                        n1 = n2 = 0
                                        arc_start = i
                                    if MP1 is not None:
                                        n1 += 1
                                        B1 += (MP1 - B1) / n1
                                    if MP2 is not None:
                                        n2 += 1
                                        B2 += (MP2 - B2) / n2
                                for j in range(arc_start, len(data)):
                                    if data[j][1] is not None:
                                        data[j][1] -= B1
                                    if data[j][2] is not None:
                                        data[j][2] -= B2
                                mp_results[sat] = [(t, mp1, mp2) for t, mp1, mp2, _, _ in data]

                            return mp_results

                        obs_data = read_rinex_obs_fixed_patch(obs_file_path)
                        mp_results = compute_mp_rtklib_like(obs_data)

                        df_mp = pd.DataFrame([
                            {"sat": sat, "time": t, "MP1": round(mp1, 4) if mp1 is not None else None, "MP2": round(mp2, 4) if mp2 is not None else None}
                            for sat in sorted(mp_results.keys())
                            for t, mp1, mp2 in mp_results[sat]
                        ])
                        df_mp["time"] = pd.to_datetime(df_mp["time"], format="%Y %m %d %H %M %S.%f")

                        st.write(f"DataFrame shape: {df.shape}")
                        observable_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                        st.write("Observable columns:", observable_columns)

                        diferencia = df['epoch'].max() - df['epoch'].min()
                        if hasattr(diferencia, "total_seconds"):
                            segundos_totales = diferencia.total_seconds()
                            horas = int(segundos_totales // 3600)
                            minutos = int((segundos_totales % 3600) // 60)
                            st.write(f"Tiempo aproximado de grabación: {horas}h {minutos}min")
                        else:
                            st.write("No se pudo calcular el tiempo de grabación.")

                        st.write("Sistemas satelitales únicos:", df['SYSTEM'].unique().tolist())
                        st.write("Recuento por sistema:")
                        st.write(df['SYSTEM'].value_counts())

                        epochs = df["epoch"].unique()
                        print("Epochs:", epochs)
                        st.write(f"Número total de épocas: {len(epochs)}")

                        time_diffs = np.diff(epochs).astype('timedelta64[ms]') / np.timedelta64(1, 's')
                        threshold = 0.2
                        missing_mask = time_diffs > threshold
                        if np.any(missing_mask):
                            indices_saltos = np.where(missing_mask)[0]
                            st.subheader(f"Se detectaron **{len(indices_saltos)}** saltos > {threshold}s:")
                            for idx in indices_saltos:
                                t1 = epochs[idx]
                                t2 = epochs[idx+1]
                                delta = time_diffs[idx]
                                st.write(f"  Entre **{t1}** y **{t2}**: {delta:.2f} s")
                        else:
                            st.subheader("**Sin saltos mayores de 0.2 s.**")

                        st.subheader("Resultados MP1 / MP2")
                        st.dataframe(df_mp.head(10))

                    return df_gps

                except subprocess.CalledProcessError as e:
                    st.error("Error en la conversión:")
                    st.text(e.stderr)
                    return None
                

        return None


# Función para mostrar el gráfico GNSS (usando df_gps)
def render_gps_graph(df_gps):
    st.title("Seguimiento Satelital - Visualización de Datos")
#    try:
#        df_gps['Epoch'] = pd.to_datetime(df_gps['Epoch'])
#    except Exception as e:
#        st.error(f"Error al convertir 'Epoch' a datetime: {e}")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(
        data=df_gps,
        x='Epoch',
        y='SV',
        hue='FREQ',
        palette={1: 'yellow', 2: 'green'},
        s=10,
        alpha=0.5,
        ax=ax
    )
    ax.set_title("Seguimiento Satelital - GPS en Frecuencia L1C y L2X")
    ax.set_xlabel("Época")
    ax.set_ylabel("Satélite (SV)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)   


# Función para mostrar el gráfico GNSS (usando df_glonass)
def render_glonass_graph(df_glonass):
    st.title("Seguimiento Satelital - Visualización de Datos")
    try:
        df_glonass['Epoch'] = pd.to_datetime(df_glonass['epoch'])
    except Exception as e:
        st.error(f"Error al convertir 'Epoch' a datetime: {e}")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(
        data=df_glonass,
        x='Epoch',
        y='SV',
        hue='FREQ',
        palette={1: 'yellow', 2: 'green'},
        s=10,
        alpha=0.5,
        ax=ax
    )
    ax.set_title("Seguimiento Satelital - en Frecuencia L1C y L2X")
    ax.set_xlabel("Época")
    ax.set_ylabel("Satélite (SV)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig) 



# ======================================================================
# Lógica principal basada en el menú lateral
if selected == "OBS-GNSS_ANALYSER":
    df_gps = run_obs_analyses()
    if df_gps is not None:
        # Almacenar df_gps en session state para reutilizar en otra sección
        st.session_state['df_gps'] = df_gps 
        #st.session_state['df_glonass'] = df_glonass

elif selected == "GNSS_Graph":
    st.title("Gráficos GNSS")
    # Verificar si ya se procesaron datos de observaciones
    if 'df_gps' in st.session_state:
        df_gps = st.session_state['df_gps']        
        render_gps_graph(df_gps)
        
    else:
        st.error("No se encontraron datos de observación. Por favor, ejecuta la opción OBS-GNSS_ANALYSER primero.")