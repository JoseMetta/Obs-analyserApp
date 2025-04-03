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

    #print("Python executable in use:", sys.executable)
    st.title("Conversión de UBX a RINEX Observación con convbin")
    st.markdown("""
    Esta aplicación utiliza **convbin.exe** para convertir un archivo UBX a formato RINEX de observación.  
    Es necesario que `convbin.exe` esté en el PATH o en el mismo directorio de ejecución.
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
                st.write("Archivos en el directorio de trabajo:")
                files_in_dir = os.listdir(cwd)
                st.write(files_in_dir)
                
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
        st.dataframe(df.head(10))

        # ==================
            # Mostrar la forma del DataFrame y los nombres de columnas
        st.write(f"DataFrame shape: {df.shape}")
        st.write("Column names:", df.columns.tolist())
    
        # Identificar columnas observables (columnas numéricas)
        observable_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        st.write("Observable columns:", observable_columns)
    
        # Analizar la columna 'epoch'
        st.write("Epoch data type:", df['epoch'].dtype)
        st.write("Epoch range:", df['epoch'].min(), "to", df['epoch'].max())
    
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


        epochs = df["epoch"].unique() # Unique epochs: Arreglo que contiene todas las épocas
        st.write(f": Número total de épocas: {len(df['epoch'].unique())}")

        time_diffs = np.diff(epochs).astype('timedelta64[ms]') / np.timedelta64(1, 's')
        print(time_diffs)
        print(len(time_diffs))



        threshold = 1  # Si un salto excede a su tasa de registro, consideramos pérdida de datos

        # Identificar dónde hay saltos grandes
        missing_mask = time_diffs > threshold
        if np.any(missing_mask):
            indices_saltos = np.where(missing_mask)[0]
            st.write(f"Se detectaron {len(indices_saltos)} saltos mayores de {threshold}s:")
            for idx in indices_saltos:
                t1 = epochs[idx]
                t2 = epochs[idx+1]
                delta = time_diffs[idx]
                st.write(f"  Salto entre {t1} y {t2}: {delta:.2f} segundos")
        else:
            st.write("No se han detectado saltos mayores de 1 s en la dimensión temporal.")
      

   
if __name__ == "__main__":
    main()






    
  




