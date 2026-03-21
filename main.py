import os
import time

import numpy as np

from benchmark import *
from initialization import *
from optimization import *


class Service:

    def __init__(self, id, demand, min_kpi, min_kvi, impact, kpi_service_req, kvi_service_req, kpi_service, kvi_service,
                 weights_kpi, weights_kvi, size,privacy_sensitivity=0.5, intent_priority=0.5):
        self.id = id
        self.demand = demand
        self.min_kpi = 0  # valore minimo globale tollerabile kpi
        self.min_kvi = 0
        self.impact = impact  # valore minimo globale tollerabile kvi
        self.kpi_service_req = np.array(kpi_service_req)  # requested minimum
        self.kvi_service_req = np.array(kvi_service_req)  # requested minimum
        self.kpi_service = np.array(kpi_service)  # 4 KPI, valore desiderato
        self.kvi_service = np.array(kvi_service)  # 3 KVI, valore desiderato
        self.weights_kpi = np.array(weights_kpi)  # per calcolo kpi globale
        self.weights_kvi = np.array(weights_kvi)  # per calcolo kvi globale
        self.size = size
        self.privacy_sensitivity=privacy_sensitivity
        self.intent_priority=intent_priority
        # self.risk_appetite = risk_appetite

    # property            # first decorate the getter method
    def get_id(self):
        return self.id

    def get_demand(self):
        return self.demand

    def get_min_kpi(self):
        return self.min_kpi

    def get_min_kvi(self):
        return self.min_kvi

    def get_impact(self):
        return self.impact

    def get_kpi_service_req(self):
        return self.kpi_service_req

    def get_kvi_service_req(self):
        return self.kvi_service_req

    def get_kpi_service(self):
        return self.kpi_service

    def get_kvi_service(self):
        return self.kvi_service

    def get_weights_kpi(self):
        return self.weights_kpi

    def get_weights_kvi(self):
        return self.weights_kvi

    def get_size(self):
        return self.size
    
    def get_privacy_sensitivity(self):
        return self.privacy_sensitivity

    def get_intent_priority(self):
        return self.intent_priority


    def set_id(self, value):
        self.id = value

    def set_demand(self, value):
        self.demand = value

    def set_min_kpi(self, value):
        self.min_kpi = value

    def set_min_kvi(self, value):
        self.min_kvi = value

    def set_impact(self, value):
        self.impact = value

    def set_kpi_service_req(self, value):
        self.kpi_service_req = value

    def set_kvi_service_req(self, value):
        self.kvi_service_req = value

    def set_kpi_service(self, value):
        self.kpi_service = value

    def set_kvi_service(self, value):
        self.kvi_service = value

    def set_weights_kpi(self, value):
        self.weights_kpi = value

    def set_weights_kvi(self, value):
        self.weights_kvi = value

    def set_size(self, value):
        self.size = value

    def set_privacy_sensitivity(self, value):
        self.privacy_sensitivity = value

    def set_intent_priority(self, value):
        self.intent_priority = value

    # def set_risk_appetite(self, value):
    #     self.risk_appetite = value



class Resource:
    def __init__(self, id, availability, kpi_resource, kvi_resource, carbon_offset, P_c, u_c, P_m, fcp, N0,
                 lambda_failure, lambda_services_per_day, likelihood,predicted_ai_risk=0.5, forecast_ai_risk=0.5, privacy_cost=0.5):
        self.id = id
        self.availability = availability
        self.kpi_resource = np.array(kpi_resource)
        self.kvi_resource = np.array(kvi_resource)
        self.carbon_offset = carbon_offset
        self.P_c = P_c
        self.u_c = u_c
        self.P_m = P_m
        self.fpc = fcp
        self.N0 = N0
        self.lambda_failure = lambda_failure
        self.lambda_services_per_day = lambda_services_per_day
        self.likelihood = likelihood
        self.predicted_ai_risk = predicted_ai_risk
        self.forecast_ai_risk = forecast_ai_risk
        self.privacy_cost = privacy_cost

    def get_availability(self):
        return self.availability

    def get_kpi_resource(self):
        return self.kpi_resource

    def get_kvi_resource(self):
        return self.kvi_resource

    def get_carbon_offset(self):
        return self.carbon_offset

    def get_P_c(self):
        return self.P_c

    def get_u_c(self):
        return self.u_c

    def get_P_m(self):
        return self.P_m

    def get_fpc(self):
        return self.fpc

    def get_N0(self):
        return self.N0

    def get_lambda_failure(self):
        return self.lambda_failure

    def get_lambda_services_per_day(self):
        return self.lambda_services_per_day

    def get_likelihood(self):
        return self.likelihood
    
    def get_predicted_ai_risk(self):
        return self.predicted_ai_risk

    def get_forecast_ai_risk(self):
        return self.forecast_ai_risk

    def get_privacy_cost(self):
        return self.privacy_cost

    def set_availability(self, value):
        self.availability = value

    def set_kpi_resource(self, value):
        self.kpi_resource = value

    def set_kvi_resource(self, value):
        self.kvi_resource = value

    def set_carbon_offset(self, value):
        self.carbon_offset = value

    def set_P_c(self, value):
        self.P_c = value

    def set_u_c(self, value):
        self.u_c = value

    def set_P_m(self, value):
        self.P_m = value

    def set_fpc(self, value):
        self.fpc = value

    def set_N0(self, value):
        self.N0 = value

    def set_lambda_failure(self, value):
        self.lambda_failure = value

    def set_lambda_services_per_day(self, value):
        self.lambda_services_per_day = value

    def set_likelihood(self, value):
        self.likelihood = value

    def set_predicted_ai_risk(self, value):
        self.predicted_ai_risk = value

    def set_forecast_ai_risk(self, value):
        self.forecast_ai_risk = value

    def set_privacy_cost(self, value):
        self.privacy_cost = value

# Inizialization of service categories and resources information

if __name__ == '__main__':

    num_services_list = [90, 100, 110, 120]
    num_services_type = 8
    delta = 0.1
    num_resources = [80]
    weights_kpi = [0.2, 0.5, 0.3]  # Deadline, datarate, PLR, respectively
    weights_kvi = [0.7, 0.1, 0.1, 0.1]  # Trustworthiness, Inclusiveness, and Environmental Sustainability, respectively

    deadlines = [0.002, 0.5, 1, 10, 15]
    deadlines_req = [0.02, 0.6, 1.2, 50]
    plrs = [20.0, 20.0, 30.0, 40.0]
    plrs_req = [35.0, 45.0, 45.0, 50.0]
    data_rates = [70.0, 100.0, 100.0, 250.0]
    data_rates_req = [45.0, 60.0, 80.0, 95.0]
    sizes = [600e6, 1e9, 1e9, 1.2e9]  # Mb
    demand_values = [2, 4, 4, 5]
    impact_values = [0.25, 0.5, 0.75, 1]
    privacy_sensitivity_values = [0.3, 0.5, 0.8, 0.95]
    intent_priority_values = [0.2, 0.4, 0.7, 0.9]


    services = []
    for i in range(num_services_type):
        chosen_index = i % len(demand_values)

        service = Service(i, 0, 0, 0,0, 0, 0, 0,
                          0, weights_kpi, weights_kvi, 0)

        deadline = deadlines[chosen_index]
        plr = plrs[chosen_index]
        plr_req = plrs_req[chosen_index]
        data_rate = data_rates[chosen_index]
        impact = impact_values[chosen_index]


        if deadline > 9:
            deadline_req = deadlines_req[len(deadlines_req) - 1]
        else:
            deadline_req = deadlines_req[chosen_index]

        if data_rate == 70:
            data_rate_req = data_rates_req[0]
        else:
            data_rate_req = data_rates_req[chosen_index]

        service.set_kpi_service([deadline, data_rate, plr])
        service.set_kpi_service_req([deadline_req, data_rate_req, plr_req])
        service.set_demand(demand_values[chosen_index])
        service.set_impact(impact)
        service.set_size(sizes[chosen_index])
        
        service.set_privacy_sensitivity(privacy_sensitivity_values[chosen_index])
        service.set_intent_priority(intent_priority_values[chosen_index])

        services.append(service)

        print(f"Service id: {services[i].id}, {services[i].demand}, {services[i].min_kpi}, {services[i].impact}, "
              f"{services[i].kpi_service}, {services[i].kpi_service_req}, {services[i].kvi_service}, {services[i].kvi_service_req}, "
              f"{services[i].size}, {services[i].privacy_sensitivity}, {services[i].intent_priority}")


    for num_services in num_services_list:
        for num_resource in num_resources:
            results_dir = f"{num_services}_{num_resource}_{delta}"
            # Add your local path
            path_locale = r""
            full_path = os.path.join(path_locale, results_dir)
            os.makedirs(full_path, exist_ok=True)

            # Probabilities assigned to services, subject to change
            probabilities = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
            service_requests = []

            # Generation of requests based on the distribution
            for i in range(num_services):
                chosen_index = i % len(probabilities)
                service_requests.append(probabilities[chosen_index])
            print("Service request distribution:", service_requests)

            start = time.time()

            availability_values = [10, 20, 50, 50]
            carbon_offset_values = [(1.5*1e6) / 365, (2*1e6) / 365, (2*1e6) / 365, (2.5*1e6) / 365]  # x grams * 10^6 (ton) /365 gg as avg, with x = [1.5, 2, 2.5] --> [(1.5*1e6) / 365, (2*1e6) / 365, (2*1e6) / 365, (2.5*1e6) / 365]
            P_c_values = [0.01, 0.02, 0.02, 0.04]
            u_c_values = [0.1, 0.5, 0.8, 1]
            P_m_values = [0.1, 0.15, 0.15, 0.2]
            fcp_values = [40e9, 100e9, 100e9, 150e9]
            N0 = 10e-10
            lambda_failure_values = [8760, 8760, 8760, 45000, 45000]
            lambda_services_per_day_values = [150, 200, 200, 250]
            likelihood_values = [0.25, 0.5, 0.75, 1]

            # Indicators offered by the resources

            deadlines_off = [0.001, 0.4, 0.8, 20]
            data_rates_off = [85.0, 110.0, 110.0, 250.0]
            plr_off = [10.0, 20.0, 20.0, 40.0]

            resources = []

            for i in range(num_resource):
                chosen_index = i % len(availability_values)
                resource = Resource(i, 0, 0, [0, 0, 0], 0, 0, 0, 0, 0, N0, 0, 0, 0)

                availability_value = availability_values[chosen_index]

                carbon_offset_value = carbon_offset_values[chosen_index]

                if carbon_offset_value > (2*1e6) / 365:
                    P_c_value = 0.01
                    u_c_value = 0.1
                    P_m_value = 0.1
                    fcp_value = 40e9
                    lambda_failure_value = 8760
                    lambda_services_per_day_value = 250
                    likelihood_value = 0.25
                    deadline_off = 0.8
                    data_rate_off = 85.0
                    plr_off_res = 20.0
                elif carbon_offset_value == (2*1e6) / 365:
                    P_c_value = 0.02
                    u_c_value = 0.5
                    P_m_value = 0.2
                    fcp_value = 100e9
                    lambda_failure_value = 45000
                    lambda_services_per_day_value = 200
                    likelihood_value = 0.5
                    deadline_off = 0.4
                    data_rate_off = 110.0
                    plr_off_res = 20.0
                else:
                    P_c_value = 0.04
                    u_c_value = 0.8
                    P_m_value = 0.2
                    fcp_value = 150e9
                    lambda_failure_value = 45000
                    lambda_services_per_day_value = 150
                    likelihood_value = likelihood_values[chosen_index]
                    deadline_off = 0.01
                    data_rate_off = 250.0
                    plr_off_res = 10.0
                #Altro modo di set
               # if resource.id < 20:
               #     resource.set_predicted_ai_risk(0.2)
               #     resource.set_forecast_ai_risk(0.25)
               #     resource.set_privacy_cost(0.1)
               # else:
               #     resource.set_predicted_ai_risk(0.7)
               #     resource.set_forecast_ai_risk(0.75)
               #     resource.set_privacy_cost(0.3)

                resource.set_availability(availability_value)
                resource.set_kpi_resource([deadline_off, data_rate_off, plr_off_res])
                resource.set_carbon_offset(carbon_offset_value)
                resource.set_P_c(P_c_value)
                resource.set_u_c(u_c_value)
                resource.set_P_m(P_m_value)
                resource.set_fpc(fcp_value)
                resource.set_lambda_failure(lambda_failure_value)
                resource.set_lambda_services_per_day(lambda_services_per_day_value)
                resource.set_likelihood(likelihood_value)

                resource.set_predicted_ai_risk(np.random.uniform(0.1, 0.9))
                resource.set_forecast_ai_risk(np.random.uniform(0.1, 0.9))
                resource.set_privacy_cost(np.random.uniform(0.1, 0.5))

                resources.append(resource)


            for resource in resources:
                print(resource.id, resource.availability, resource.kpi_resource, resource.fpc,
                      resource.P_m, resource.P_c, resource.lambda_services_per_day, resource.likelihood)

            # Computation of Q_MIN and computation time

            q_v_big_req(services, [-1, 1, -1], [1, -1, -1])

            for service in services:
                for resource in resources:
                    computation_time = compute_computation_time(service, resource)
                    print(computation_time)

            # TIS:  Trustworthiness, Inclusiveness, and Environmental Sustainability
            normalized_kvi, weighted_sum_kvi, energy_sustainability_values, trustworthiness_values, failure_probability_values = compute_normalized_kvi(services, resources, CI=475, signs=[1, -1, -1, -1])  #

            normalized_kpi, weighted_sum_kpi = compute_normalized_kpi(services, resources, signs=[-1, 1, -1])  # latenza, data rate e plr



            ############## EPSILON-CONSTRAINT METHOD: COMPUTATION OF IDEAL AND NADIR POINTS AND EXACT-METHOD IMPLEMENTATION ############

            V_I = optimize_kvi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                               results_dir)

            Q_I = optimize_kpi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi,
                               weighted_sum_kvi, results_dir)

            V_N = v_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi,
                          weighted_sum_kpi, weighted_sum_kvi, Q_I, results_dir)

            Q_N = q_nadir(service_requests, services, resources, normalized_kpi,
                          normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I, results_dir)


            pareto_solutions_exact = epsilon_constraint_exact(service_requests, services, resources, normalized_kpi, normalized_kvi,
            weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=delta, results_dir=results_dir)

            pareto_filename = os.path.join(results_dir, "pareto_solutions.csv")
            save_pareto_solutions(pareto_solutions_exact, filename=pareto_filename)

            ############ BENCHMARK APPROACHES: GREEDY ASSIGNMENT KPI, KVI AND RANDOM ASSIGNMENT ############

            assignment, total_kpi, total_kvi = greedy_assignment_kpi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi)

            save_assignment_results(service_requests, assignment, services, resources,
                                    weighted_sum_kpi, weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi,
                                    total_kvi,
                                    results_dir=results_dir, filename="greedy_kpi_results.csv")

            assignment, total_kpi, total_kvi = greedy_assignment_kvi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi)

            save_assignment_results(service_requests, assignment, services, resources,
                                    weighted_sum_kpi, weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi,
                                    total_kvi,
                                    results_dir=results_dir, filename="greedy_kvi_results.csv")


            assignment, total_kpi, total_kvi = random_assignment(service_requests, services, resources, weighted_sum_kpi,
                                                                 weighted_sum_kvi)

            save_assignment_results(service_requests, assignment, services, resources, weighted_sum_kpi,
                                    weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi, total_kvi,
                                    results_dir=results_dir,
                                    filename="random_results.csv")


            # Execution times
            end_time = time.time()
            time_elapsed = end_time - start
            with open(os.path.join(results_dir, "execution_time.txt"), "w") as file:
                file.write(f"Services: {num_services}, Time: {time_elapsed:.6f} sec\n")

            print(f"Completed for {num_services} services. Time: {time_elapsed:.6f} sec")
