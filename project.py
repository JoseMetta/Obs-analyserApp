import streamlit as st
import os
import subprocess
import numpy as np
import gnsspy as gp
import pandas as pd
import matplotlib.pyplot as plt
#from pyunpack import Archive
#import patool
import sys

print("Python executable in use:", sys.executable)

def main():

    st.title("Conversión de UBX a RINEX Observación con convbin")
    st.markdown("""
    Esta aplicación utiliza **convbin.exe** para convertir un archivo UBX a formato RINEX de observación.      
    """)

    # Subir el archivo UBX
    uploaded_file = st.file_uploader("Sube tu archivo UBX", type=["ubx"])

    # Campo para que el usuario ingrese la ruta destino (dentro del servidor)
    destination_path = st.text_input("Ruta para guardar el archivo (sin caracteres especiales):", "C:/Proyectos/gnsspy/data")
    
    # Campos para especificar el archivo de navegación y el de salida
    nav_file = st.text_input("Archivo de navegación (nav):", "salida.nav")
    obs_file = st.text_input("Archivo de salida (obs):", "salida.obs")
    
    if uploaded_file is not None:
        # Usar la ruta ingresada por el usuario para guardar el archivo
        os.makedirs(destination_path, exist_ok=True)
        input_path = os.path.join(destination_path, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Archivo UBX guardado en: {input_path}")
        
        if st.button("Convertir a RINEX Observación"):
            # Directorio de trabajo para la conversión será el mismo que contiene el archivo UBX
            cwd = os.path.dirname(input_path)
            st.write("Directorio de trabajo:", cwd)

            # Construir el comando de convbin
            # Asegúrate de que convbin.exe esté en el PATH o en el mismo directorio
            command = [
                "convbin.exe",
                "-r", "ubx",                
                "-hm",
                "-ho",
                "-hr",
                "-ha",
                "-v",
                "3.02",
                "-od",
                "-oi",
                "-ot",
                "-ol",
                "-os",
                "-halfc",
                "-trace", "level", "2",
                "-n", nav_file,
                "-o", obs_file,
                input_path  # El archivo UBX de entrada
            ]
            
            st.write("Ejecutando el siguiente comando:")
            st.code(" ".join(command))
            
            try:
                # Ejecutar el comando en el directorio destino
                result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
                st.success("¡Conversión exitosa!")
                st.markdown("**Salida del comando:**")
                st.write("Directorio de trabajo:", cwd)
                #st.text(result.stdout)
                #st.text("Salida de error (stderr):")
                #st.text(result.stderr)

                # Listar archivos en el directorio de trabajo para depurar
                #st.write("Archivos en el directorio de trabajo:")
                #files_in_dir = os.listdir(cwd)
                #st.write(files_in_dir)
                
                # Construir las rutas completas de salida en el mismo directorio
                obs_file_path = os.path.join(cwd, obs_file)
                nav_file_path = os.path.join(cwd, nav_file)

                # ====================   Ofrecer la descarga del archivo de observación (prioritario) ====================
                if os.path.exists(obs_file_path):
                    with open(obs_file_path, "rb") as f:                        
                        st.download_button("Descargar archivo OBS", f, file_name=obs_file)
                else:
                    st.error("No se encontró el archivo de salida OBS.")
                
                # Ofrecer la descarga del archivo de navegación
                #if os.path.exists(nav_file_path):
                #    with open(nav_file_path, "rb") as f:
                #        st.download_button("Descargar archivo NAV", f, file_name=nav_file)
                #else:
                #    st.error("No se encontró el archivo de salida NAV.")

            except subprocess.CalledProcessError as e:
                st.error("Error en la conversión:")
                st.text(e.stderr) 

        # Usar la ruta completa al archivo de observación
        obs_file_path = os.path.join(destination_path, obs_file)
        station = gp.read_obsFile(obs_file_path)
        df = station.observation
        st.dataframe(df)
        #print(df['epoch'].unique())



        # Constantes
        CLIGHT = 299792458.0        # Velocidad de la luz (m/s)
        FREQ1 = 1575.42e6           # Frecuencia L1 GPS (Hz)
        FREQ2 = 1227.60e6           # Frecuencia L2 GPS (Hz)

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
                        extra = lines[i][0:len(lines[i])].rstrip('\n')
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
        
        # Convertir la lista de datos a un DataFrame de pandas
        obs_data = read_rinex_obs_fixed_patch(obs_file_path)
        #print(obs_data)

        mp_results = compute_mp_rtklib_like(obs_data)

        data = []
        # Recorrer los resultados de MP1 y MP2
        for sat in sorted(mp_results.keys()):
            for time, mp1, mp2 in mp_results[sat]:
                data.append({
                    "sat": sat,
                    "time": time,
                    "MP1": round(mp1, 4) if mp1 is not None else None,
                    "MP2": round(mp2, 4) if mp2 is not None else None
                })

        # Crear el DataFrame
        df_mp = pd.DataFrame(data)
        df_mp["time"] = pd.to_datetime(df_mp["time"], format="%Y %m %d %H %M %S.%f")

      

        # ========= Mostrar la forma del DataFrame y los nombres de columnas
        st.write(f"DataFrame shape: {df.shape}")
        #st.write("Column names:", df.columns.tolist())
    
        # Identificar columnas observables (columnas numéricas)
        observable_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        st.write("Observable columns:", observable_columns)
    
        # Analizar la columna 'epoch'
        #st.write("Epoch data type:", df['epoch'].dtype)
        #st.write("Epoch range:", df['epoch'].min(), "to", df['epoch'].max())
    
        # Calcular la diferencia entre la época máxima y mínima
        diferencia = df['epoch'].max() - df['epoch'].min()
        # Se verifica si 'diferencia' tiene el método total_seconds (para objetos datetime)
        if hasattr(diferencia, "total_seconds"):
            segundos_totales = diferencia.total_seconds()
            horas = int(segundos_totales // 3600)
            minutos = int((segundos_totales % 3600) // 60)
            st.write(f"The recording time aprox is: {horas} horas y {minutos} minutos.")
        else:
            st.write("No se pudo calcular el tiempo de grabación. Verifica el tipo de datos de 'epoch'.")
    
        # Explorar la columna 'SYSTEM'
        st.write("Unique satellite systems:", df['SYSTEM'].unique().tolist())
        st.write("System observation counts:")
        st.write(df['SYSTEM'].value_counts())


        epochs = df_mp["time"].unique() # Unique epochs: Arreglo que contiene todas las épocas
        print(f": Número total de épocas con gnsspy: {len(df['epoch'].unique())}")
        st.write(f": Número total de épocas de archivo rinex: {len(df_mp['time'].unique())}")

        time_diffs = np.diff(epochs).astype('timedelta64[ms]') / np.timedelta64(1, 's')
        print(time_diffs)
        print(len(time_diffs))



        threshold = 0.2  # Si un salto excede a su tasa de registro, consideramos pérdida de datos

        # Identificar dónde hay saltos grandes
        missing_mask = time_diffs > threshold
        if np.any(missing_mask):
            indices_saltos = np.where(missing_mask)[0]
            st.subheader(f"Se detectaron **{len(indices_saltos)}** saltos mayores de **{threshold}**s:")
            for idx in indices_saltos:
                t1 = epochs[idx]
                t2 = epochs[idx+1]
                delta = time_diffs[idx]
                st.write(f"  Salto entre **{t1}** y **{t2}**: {delta:.2f} segundos")
        else:
            st.subheader("**No se han detectado saltos mayores de 0.2 s en la dimensión temporal.")
           
        
        # ================== Convertir diccionario a data frame ================== #
        data = []
        # Recorrer los resultados de MP1 y MP2
        for sat in sorted(mp_results.keys()):
            for time, mp1, mp2 in mp_results[sat]:
                data.append({
                    "sat": sat,
                    "time": time,
                    "MP1": round(mp1, 4) if mp1 is not None else None,
                    "MP2": round(mp2, 4) if mp2 is not None else None
                })

        # Crear el DataFrame
        df_mp = pd.DataFrame(data)
        st.subheader("Resultados de MP1 y MP2")
        st.write(df_mp.head(10))


   
if __name__ == "__main__":
    main()






    
  




