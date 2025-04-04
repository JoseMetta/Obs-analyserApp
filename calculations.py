import math
import csv

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

#Ejecutar
if __name__ == "__main__":
    rinex_path = "Prueba5.obs"
    output_csv = "mp1_mp2_final_98realnofake.csv"
    obs_data = read_rinex_obs_fixed_patch(rinex_path)
    mp_results = compute_mp_rtklib_like(obs_data)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sat", "time", "MP1", "MP2"])
        for sat in sorted(mp_results.keys()):
            for time, mp1, mp2 in mp_results[sat]:
                writer.writerow([sat, time, f"{mp1:.4f}" if mp1 is not None else "", f"{mp2:.4f}" if mp2 is not None else ""])