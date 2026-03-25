import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_single_row(kvi_service, kvi_service_req, index_res, signs, kvi_values): 
    row = np.zeros(len(kvi_service)) 
    maximum = np.max(kvi_values, axis=0) 
    minimum = np.min(kvi_values, axis=0) 
    for index, attribute in enumerate(kvi_service): 
        for requested in kvi_service_req: 
            exposed_kvi = kvi_values[index_res][index]   
            max_val = maximum[index] 
            min_val = minimum[index] 
            if exposed_kvi == attribute: 
                row[index] = 1 
            else: 
                if signs[index] == 1:  # benefit 
                    if max_val == requested or max_val == min_val: 
                        row[index] = 1 
                    else: 
                        row[index] = 1 - (max_val - exposed_kvi) / (max_val - requested) 
                else:  # cost 
                    if min_val == requested or max_val == min_val: 
                        row[index] = 1 
                    else: 
                        row[index] = 1 - (exposed_kvi - min_val) / (requested - min_val) 
    return np.abs(row)

# function KVI AI risk
def compute_ai_risk(service, resource, rho_min=0.2, rho_max=1.0, beta=0.3):
    """
    Restituisce il nuovo KVI di AI risk per la coppia (service, resource):
        L_ij = rho_i * r_hat_ij

    Più il valore è alto, maggiore è il rischio AI associato all'allocazione.
    """

    # --- Parametri del servizio (paper: s_t, I_t) ---
    # Sensibilità privacy della richiesta
    privacy_sensitivity = np.clip(getattr(service, "privacy_sensitivity", service.impact), 0, 1)

    # Priorità / criticità dell'intento
    intent_priority = np.clip(getattr(service, "intent_priority", privacy_sensitivity), 0, 1)

    # Modulazione intent-driven: rho_i
    rho_i = rho_min + (rho_max - rho_min) * intent_priority

    # --- Parametri della risorsa (paper: a_hat_t, c_t(x)) ---
    # Rischio AI predetto sulla risorsa
    predicted_ai_risk = np.clip(getattr(resource, "predicted_ai_risk", resource.likelihood), 0, 1)

    # Rischio AI previsto nel look-ahead window
    #forecast_ai_risk = np.clip(getattr(resource, "forecast_ai_risk", predicted_ai_risk), 0, 1)

    # Costo privacy della risorsa
    # fallback: proxy basato su carico richiesto / capacità giornaliera
    privacy_cost = np.clip(getattr(resource,"privacy_cost",service.size / max(resource.fpc * resource.lambda_services_per_day, 1)),0,1)

    # --- Costruzione di r_hat_ij ---
    # componente rischio base: servizio critico su risorsa esposta
    #base_risk = privacy_sensitivity * predicted_ai_risk

    # componente previsionale (prediction-augmented)
    #forecast_component = beta * forecast_ai_risk

    # componente di costo/privacy budget
    #budget_component = privacy_sensitivity * privacy_cost

    # rischio nominale complessivo
    r_hat_ij = np.clip(privacy_sensitivity * predicted_ai_risk + privacy_sensitivity * privacy_cost,0,1)

    # --- Indicatore finale del nuovo KVI ---
    L_ij = rho_i * r_hat_ij

    return float(L_ij)

# function computation time in h
def compute_computation_time(service, resource):
    return service.size * 1000 / resource.fpc


# function KVI environmental sustainability
def compute_energy_sustainability(resource, computation_time, CI=475, PUE=1.67):
    return  (computation_time / 3600) * resource.lambda_services_per_day * (
            resource.availability * resource.P_c * resource.u_c + resource.availability * resource.P_m) * PUE * CI

# function KVI trustworthiness
def compute_trustworthiness(service, resource):
    cyber_risk = resource.likelihood * service.impact
    cyber_confidence = 1 - cyber_risk
    return 900 + 4100 / (1 + np.exp(- 0.6 * (cyber_confidence - 0.5)))

# function KVI inclusiveness
def compute_failure_probability(computation_time, resource):
    exponent = - 24 / resource.lambda_failure
    failure_probability = (1 - np.exp(exponent))  # p_rn
    F_rn_0 = (1 - failure_probability) ** resource.availability
    print("F_rn_0", F_rn_0)
    return F_rn_0 * computation_time * resource.lambda_services_per_day

# function to compute indicators for each (service, resource) couple, normalization and weighted sum to get V(X)
def compute_normalized_kvi(services, resources, CI, signs, feature_range=(0.1, 1.0)):
    normalized_kvi = {}
    weighted_sum_kvi = {}
    energy_sustainability_values = {}
    trustworthiness_values = {}
    failure_probability_values = {}
 
    signs = np.array(signs, dtype=int) 
    low, high = feature_range
 
    for service in services:
        raw_kvi_matrix = []
        keys = []
 
        # calcolo KVI per tutte le risorse del servizio 
        for resource in resources:
            computation_time = compute_computation_time(service, resource)
 
            trustworthiness = float(compute_trustworthiness(service, resource))
            energy_sustainability = float(
                resource.carbon_offset
                - compute_energy_sustainability(resource, computation_time, CI)
            )
            failure_probability = float(
                compute_failure_probability(computation_time, resource)
            )
            ai_risk = float(compute_ai_risk(service, resource))
 
            raw_kvi_matrix.append([
                trustworthiness,
                failure_probability,
                energy_sustainability,
                ai_risk
            ])
            keys.append((resource.id, service.id))
 
            # 3 dizionari pesati  
            trustworthiness_values[(resource.id, service.id)] = (
                trustworthiness * service.weights_kvi[0]
            )
            failure_probability_values[(resource.id, service.id)] = (
                failure_probability * service.weights_kvi[1]
            )
            energy_sustainability_values[(resource.id, service.id)] = (
                energy_sustainability * service.weights_kvi[2]
            )
 
        raw_kvi_matrix = np.asarray(raw_kvi_matrix, dtype=float)
 
        # Normalizzazione in (0.1, 1.0) per evitare zeri
        scaler = MinMaxScaler(feature_range=feature_range, clip=True)
        norm_kvi_matrix = scaler.fit_transform(raw_kvi_matrix)
 
        # assegna max se col è costante
        constant_cols = np.ptp(raw_kvi_matrix, axis=0) == 0
        norm_kvi_matrix[:, constant_cols] = high
 
        # considerazione differente tra kvi costi/benefici sulla base degli 1 e -1
        cost_cols = signs == -1
        norm_kvi_matrix[:, cost_cols] = low + high - norm_kvi_matrix[:, cost_cols]
 
        # clip che forse può essere tolto, lasciamolo al momento come check aggiuntivo
        norm_kvi_matrix = np.clip(norm_kvi_matrix, low, high)
 
        for i, key in enumerate(keys):
            normalized_kvi[key] = norm_kvi_matrix[i]
            weighted_sum_kvi[key] = float(
                np.dot(service.weights_kvi, norm_kvi_matrix[i])
            )
            #l'ho commentata solo perché l'ho messa come check dei valori normalizzati
            #print("key:", key, "norm_kvi:", norm_kvi_matrix[i])
 
    return (
        normalized_kvi,
        weighted_sum_kvi,
        energy_sustainability_values,
        trustworthiness_values,
        failure_probability_values
    )

def normalize_single_row_kpi(kpi_service, kpi_service_req, resources, index_res, signs, kpi_values):
    row = np.zeros(len(kpi_service))  # row for the 3 kpis of the i-th service offered by the index_res-th resource
    maximum = np.max(kpi_values, axis=0)
    minimum = np.min(kpi_values, axis=0)

    for index, attribute in enumerate(kpi_service):
        for requested in kpi_service_req:
            exposed_kpi = resources[index_res].kpi_resource[
                index]  # vector offered by the resource
            # index_res-th
            max_val = maximum[index]  # value
            min_val = minimum[index]

            if exposed_kpi == attribute:
                row[index] = 1  # if the value == the requested one

            else:
                if signs[index] == 1:  # benefit, the higher the better
                    if max_val == requested:  # no zero division
                        row[index] = 1
                    elif max_val == min_val:  # if all values are =, assign 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (max_val - exposed_kpi) / (max_val - requested)

                else:  # Cost, the lower the better
                    if min_val == requested:  # no zero division
                        row[index] = 1
                    elif max_val == min_val:  # if all values are =, assign 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (exposed_kpi - min_val) / (requested - min_val)

    return np.abs(row)


def compute_normalized_kpi(services, resources, signs):
    # function to compute indicators for each (service, resource) couple, normalization and weighted sum to get Q(X)

    normalized_kpi = {}
    weighted_sum_kpi = {}

    for j, service in enumerate(services):
        kpi_values = []  # list of future lists, length = 3

        # Indicators' computation for all resources
        for n, resource in enumerate(resources):
            kpi_values.append(resource.kpi_resource)
        # Normalization
        for n, resource in enumerate(resources):
            norm_kpi = normalize_single_row_kpi(service.kpi_service, service.kpi_service_req, resources, n, signs,
                                                     kpi_values)
            normalized_kpi[(resource.id, service.id)] = norm_kpi

            # Weighted sum
            q_x = np.dot(service.weights_kpi, norm_kpi)
            weighted_sum_kpi[(resource.id, service.id)] = float(q_x)

    return normalized_kpi, weighted_sum_kpi

# function to compute Q and V req, per the optimization problem
def q_v_big_req(services, signs_kpi, signs_kvi):
    kpi_tot = np.array([service.kpi_service_req for service in services])
    kvi_tot = np.array([service.kvi_service_req for service in services])

    max_kpi_req = np.max(kpi_tot, axis=0)
    min_kpi_req = np.min(kpi_tot, axis=0)
    max_kvi_req = np.max(kvi_tot, axis=0)
    min_kvi_req = np.min(kvi_tot, 0)

    for service in services:
        temp_kpi = np.zeros(len(service.kpi_service_req))

        for index, requested in enumerate(service.kpi_service_req):
            if max_kpi_req[index] > min_kpi_req[index]:  # no by zero division
                if signs_kpi[index] == 1:  # Benefit, the higher the better
                    temp_kpi[index] = (requested - min_kpi_req[index]) / (max_kpi_req[index] - min_kpi_req[index])
                else:  # Cost, the lowest value we can get must be 1, the highest 0
                    temp_kpi[index] = (requested - max_kpi_req[index]) / (min_kpi_req[index] - max_kpi_req[index])
            else:
                temp_kpi[index] = 1  # if all values are equal, assign 1


