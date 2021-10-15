#!/usr/bin/env python3

import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict

algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns"]
# algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns", "priorityws"]
# algorithms = ["cusparse", "aowns", "ws", "mpi", "xowns", 'priorityws']
# algorithms = ["cusparse", "xowns", "axowns", 'priorityws']
# algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns", "axowns", 'priorityws']
# algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns", "axowns"]
# algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns", "axowns"]
algorithms = ["cusparse", "aowns", "ws", "mpi", "xowns", "axowns"]

algorithm_relabel = {'cusparse': 'S-C RDMA', 'aowns' : 'S-A RDMA', 'ws' : 'R WS S-A RDMA', 'mpi': 'SUMMA MPI', 'comblas': 'CombBLAS GPU', 'xowns': 'LA WS S-C RDMA', 'priorityws': 'Priority', 'axowns': 'LA WS S-A RDMA'}

def rec_dd():
    return defaultdict(rec_dd)

# rec_dd = lambda: defaultdict(rec_dd)
formatted_data = rec_dd()

# datasets = ['Nm7', 'Nm8', 'amazon_large_original_general', 'isolates']
# datasets = ['Nm7', 'Nm8', 'amazon_large_original_general', 'amazon_large_randomized_general', 'isolates', 'com-Orkut_general', 'reddit_general']
# REMOVE Nm7
# datasets = ['Nm7', 'Nm8', 'amazon_large_original_general', 'isolates']
datasets = ['amazon_large_original_general', 'isolates', 'friendster_original']
# datasets = ['Nm7', 'Nm8', 'amazon_large_original_general']

dataset_replacement = {'amazon_large_original_general': 'Amazon Large, Original', 'isolates': 'Isolates, Subgraph2', 'friendster': 'Friendster'}

color_map = {'comblas': 'C0', 'mpi': 'C1', 'cusparse': 'C2', 'aowns': 'C3', 'ws': 'C4', 'xowns': 'C5', 'axowns': 'C6'}

legend_order = {'comblas': 0, 'mpi': 1, 'cusparse': 2, 'cowns': 2, 'aowns': 3, 'ws': 4, 'xowns': 5, 'axowns': 6}

for algorithm in algorithms:
    directories = glob.glob(algorithm + '_*/')
    print(algorithm)

    for directory in directories:
        dataset = directory[len(algorithm)+1:-1]
        if algorithm == 'comblas' and dataset == 'friendster':
            print('CB Looking for CombBLAS data!')
        os.chdir(directory)
        print('  %s' % (directory,))
        print('DATASET: \"%s\"' % (dataset,))
        if dataset not in datasets:
            os.chdir('..')
            continue

        data_files = glob.glob('*.o*')

        for file in data_files:
            print('    ..%s' % (file,))

            for line in open(file, 'r'):
              m = re.match('jsrun .*-n (\d+) -a (\d+) .+? ./.+? .+? (\d+)', line)
              if m:
                  nnodes = int(m.group(1))
                  gpus_per_node = int(m.group(2))
                  k = int(m.group(3))
                  total_gpus = nnodes*gpus_per_node

              m = re.match('SpMM took (.+?) s', line)
              if m:
                  total_time = float(m.group(1))
                  # formatted_data[algorithm][total_gpus].append((k, total_time, {}))
                  if 'total_time' not in formatted_data[dataset][k][algorithm][total_gpus]:
                      formatted_data[dataset][k][algorithm][total_gpus]["total_time"] = []
                  formatted_data[dataset][k][algorithm][total_gpus]["total_time"].append(total_time)
                  print('recording %s %s %s %s %s = %s' % (dataset, k, algorithm, total_gpus, 'total_time', total_time))
                  if dataset == 'friendster' and algorithm == 'friendster':
                      print('CB recording total_time %s' % (total_time))

              component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                   'duration_accumulate', 'duration_barrier']

              for component in component_timings:
                  m = re.match('%s (.+?) \((.+?) -> (.+?)\)' % (component,), line)
                  if m:
                      time = float(m.group(1))
                      min_time = float(m.group(2))
                      max_time = float(m.group(3))
                      # formatted_data[algorithm][total_gpus][-1][2][component] = time
                      if component not in formatted_data[dataset][k][algorithm][total_gpus][component]:
                          formatted_data[dataset][k][algorithm][total_gpus][component] = []
                      formatted_data[dataset][k][algorithm][total_gpus][component].append(time)

        os.chdir('..')

ks = set()
for dataset in formatted_data.keys():
    for k in formatted_data[dataset].keys():
        ks.add(k)

# REMOVE everything but 128, 512
ks = [128, 512]

def graph_spmm_times(formatted_data, fname='out.pdf'):
    # figure, axis = plt.subplots(len(ks), len(formatted_data.keys()), figsize=(4.267*len(formatted_data.keys()), 3.2*len(ks)), constrained_layout=True)
    dims = (len(ks), len(formatted_data.keys()))
    # figsize=(4.267*dims[0], 3.2*dims[1])
    figsize=(4.267*dims[0], 3.2*dims[1])
    figure, axis = plt.subplots(len(ks), len(formatted_data.keys()), figsize=(4.267*len(formatted_data.keys()), 3.2*len(ks)), constrained_layout=True)
    # figure, axis = plt.subplots(dims[1], dims[0], figsize=(figsize[1], figsize[0]), constrained_layout=True)

    bounds = {}
    for dataset in sorted(formatted_data.keys()):
        bounds[dataset] = []
        for k in ks:
            for algorithm in formatted_data[dataset][k].keys():
                domain = sorted(formatted_data[dataset][k][algorithm].keys())
                if domain[0] <= 6:
                    domain = domain[1:]
                print('%s, %s' % (dataset, algorithm))
                min_time = min([np.median(formatted_data[dataset][k][algorithm][x]['total_time']) for x in domain])
                max_time = max([np.median(formatted_data[dataset][k][algorithm][x]['total_time']) for x in domain])
                if len(bounds[dataset]) == 0:
                    bounds[dataset] = [min_time, max_time]
                else:
                    bounds[dataset][0] = min(bounds[dataset][0], min_time)
                    bounds[dataset][1] = max(bounds[dataset][1], max_time)
                print('%s, (%s -> %s)' % (dataset, bounds[dataset][0], bounds[dataset][1]))

    print('iterating through...')
    x_ = 0
    for dataset in sorted(formatted_data.keys()):
        print(dataset)
        y_ = 0
        for k in ks:
            x = x_
            y = y_
            min_value = 1e10
            for algorithm in sorted(formatted_data[dataset][k].keys(), key=lambda x: legend_order[x]):
                # for p in sorted(formatted_data[dataset][k][algorithm].keys()):
                #     print('%s => %s' % (formatted_data[dataset][k][algorithm][p]['total_time'], np.median(formatted_data[dataset][k][algorithm][p]['total_time'])))
                ngpus = sorted(formatted_data[dataset][k][algorithm].keys())
                if ngpus[0] <= 6:
                    ngpus = ngpus[1:]
                times = [np.median(formatted_data[dataset][k][algorithm][x]["total_time"]) for x in ngpus]
                print('%s, %s, %s: %s' % (dataset, algorithm, k, times))
                print('%s vs %s' % (len(ngpus), len(times)))

                if y == 0:
                    dataset_title = dataset
                    if dataset_title in dataset_replacement:
                        dataset_title = dataset_replacement[dataset_title]
                    axis[y, x].set_title(dataset_title)

                min_value = min(min_value, times[0] * (ngpus[0]/6))
                print('min_value %s for (%s, %s)' % (times[0] / (ngpus[0] / 6), times[0], ngpus[0] / 6))
                axis[y, x].loglog(ngpus, times, label=algorithm_relabel[algorithm], marker='s', color=color_map[algorithm], markerfacecolor='w')
                # axis[y, x].set_ybound(0.03, 3.5)
                domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
                if domain[0] <= 6:
                    domain = domain[1:]
                axis[y, x].minorticks_off()
                axis[y, x].set_xticks(domain)
                axis[y, x].set_xticklabels([str(x) for x in domain])

                if y == len(axis)-1:
                    axis[y, x].set_xlabel("Number of GPUs")
                    pass

                yticks = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 6.4*2]
                # yticks = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
                axis[y, x].set_yticks(yticks)
                # print('%s setting ylim(%s, %s)' % (dataset, bounds[dataset][0]*0.9, bounds[dataset][1]*1.1))
                axis[y, x].set_ylim(bottom=bounds[dataset][0]*0.9, top=bounds[dataset][1]*1.1)
                axis[y, x].set_yticklabels([str(x) for x in yticks])
                if x == 0:
                    axis[y, x].set_ylabel("N = %s\n\nRuntime(s)" % (k,))

            domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
            if domain[0] <= 6:
                domain = domain[1:]
            times = [min_value]*len(domain) / (domain/6)
            mul=1.0
            axis[y, x].loglog(domain, times*mul, label='Perfect Scaling', linestyle='--', color='grey')
            print('perfect: %s,%s' % (domain[0], (times*mul)[0]))
            axis[y, x].minorticks_off()
            axis[y, x].set_xticks(domain)
            axis[y, x].set_xticklabels([str(x) for x in domain])

            if y == len(axis)-1:
                axis[y, x].set_xlabel("Number of GPUs")
                pass

            yticks = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 6.4*2]
            # yticks = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
            axis[y, x].set_yticks(yticks)
            # print('%s setting ylim(%s, %s)' % (dataset, bounds[dataset][0]*0.9, bounds[dataset][1]*1.1))
            axis[y, x].set_ylim(bottom=bounds[dataset][0]*0.9, top=bounds[dataset][1]*1.1)
            axis[y, x].set_yticklabels([str(x) for x in yticks])

            y_ += 1
        x_ += 1


    # axis[0, 0].legend(loc='best')
    axis[1, 1].legend(loc='best')
    # plt.tight_layout()

    plt.savefig(fname)

component_relabel = {'duration_issue': 'Issue', 'duration_sync': 'Sync', 'duration_compute': 'Compute', 'duration_accumulate': 'Accumulate', 'duration_barrier': 'Barrier', 'mem_alloc': 'Mem Alloc'}

def graph_spmm_components(formatted_data, fname='out.pdf'):
    # figure, axis = plt.subplots(len(algorithms), len(formatted_data.keys()), sharex=True, figsize=(25.6, 12.8))
    # figure, axis = plt.subplots(len(algorithms), len(datasets), figsize=(4.267*len(formatted_data.keys()), 3.2*len(ks)))
    figure, axis = plt.subplots(len(algorithms), len(datasets), figsize=(4.267*len(datasets), 3.2*len(algorithms)))

    bounds = {}
    for dataset in sorted(formatted_data.keys()):
        # REMOVE, CHANGE k
        k = 256
        # k = 256
        bounds[dataset] = []
        for algorithm in formatted_data[dataset][k].keys():
            domain = sorted(formatted_data[dataset][k][algorithm].keys())
            print('%s, %s' % (dataset, algorithm))
            min_time = min([np.median(formatted_data[dataset][k][algorithm][x]['total_time']) for x in domain])
            max_time = max([np.median(formatted_data[dataset][k][algorithm][x]['total_time']) for x in domain])
            if len(bounds[dataset]) == 0:
                bounds[dataset] = [min_time, max_time]
            else:
                bounds[dataset][0] = min(bounds[dataset][0], min_time)
                bounds[dataset][1] = max(bounds[dataset][1], max_time)
            print('%s, (%s -> %s)' % (dataset, bounds[dataset][0], bounds[dataset][1]))

    print('DATA PRINTOUT')
    for dataset in sorted(formatted_data.keys()):
        print('dataset: %s' % (dataset,))
        for k in [128, 512]:
            print('k: %s' % (k,))
            for algorithm in formatted_data[dataset][k].keys():
                print('algorithm: %s' % (algorithm,))
                for p in [12,192]:
                    print('p: %s' % (p,))
                    for component in formatted_data[dataset][k][algorithm][p]:
                        print('%s: %s' % (component, np.median(formatted_data[dataset][k][algorithm][p][component])))


    print('iterating through...')
    x = 0
    for dataset in sorted(formatted_data.keys()):
        print(dataset)
        y = 0
        k = 256
        for algorithm in formatted_data[dataset][k].keys():
            # algorithm = 'cusparse'
            # "total_gpus"
            domain = sorted(formatted_data[dataset][k][algorithm].keys())
            times = [np.median(formatted_data[dataset][k][algorithm][x]["total_time"]) for x in domain]

            component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                 'duration_accumulate', 'duration_barrier']

            if y == 0:
                dataset_title = dataset
                if dataset_title in dataset_replacement:
                    dataset_title = dataset_replacement[dataset_title]
                axis[y, x].set_title(dataset_title)

            durations = {}
            width = 0.35
            bottom = np.array([0]*len(domain), dtype=np.float64)
            widths = width*np.array(domain)
            for component in component_timings:
                graph = False
                if graph and 'isolates' in dataset and 'duration_barrier' in component:
                    for k_ in ks:
                        print('Graphing isolates...')
                        print('isolates dataset, %s algorithm, k = %s' % (algorithm, k_))
                        for p in domain:
                            total_time = np.median(formatted_data[dataset][k_][algorithm][p]['total_time'])
                            duration_barrier = np.median(formatted_data[dataset][k_][algorithm][p]['duration_barrier'])
                            print('%s %s (%s total, %s barrier) ' % (p, total_time - duration_barrier, total_time, duration_barrier))
                        '''
                        for x in domain:
                            print(x)
                        '''
                print('Looking at the median of %s,%s,%s,%s,%s: %s' % (dataset, k, algorithm, domain, component, formatted_data[dataset][k][algorithm][x][component]))
                times = [np.median(formatted_data[dataset][k][algorithm][x][component]) for x in domain]
                print('%s: %s' % (component, times,))
                # axis[y, x].loglog(domain, times, label=component)
                component_label = component
                if component_label in component_relabel:
                    component_label = component_relabel[component_label]
                axis[y, x].bar(domain, times, width=widths, bottom=bottom, label=component_label)
                bottom += times
            times = np.array([np.median(formatted_data[dataset][k][algorithm][x]['total_time']) for x in domain]) - bottom
            axis[y, x].bar(domain, times, width=widths, bottom=bottom, label='mem_alloc')
            axis[y, x].set_xscale('log')
            print('Setting ylim of %s to %s' % (dataset, bounds[dataset][1]))
            axis[y, x].set_ylim(top=bounds[dataset][1])

            domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
            axis[y, x].minorticks_off()
            axis[y, x].set_xticks(domain)
            axis[y, x].set_xticklabels([str(x) for x in domain])

            if y == len(axis)-1:
                axis[y, x].set_xlabel("Number of GPUs")
                pass

            if x == 0:
                axis[y, x].set_ylabel("%s\n\nRuntime (s)" % (algorithm_relabel[algorithm],))

            y_ += 1
        x_ += 1

    axis[0, 0].minorticks_off()
    axis[0, 0].set_xticks(6*np.array([1, 2, 4, 8, 16, 32, 64]))
    axis[0, 0].set_xticklabels(6*np.array([1, 2, 4, 8, 16, 32, 64]))
    axis[0, 0].legend(loc='best')
    plt.tight_layout()

    plt.savefig(fname)

def print_data(formatted_data):
    for algorithm in algorithms:
        print('%s' % (algorithm,))
        for dataset in sorted(formatted_data.keys()):
            print('%s' % (dataset,))
            print('np,k,tot,comm,comp,barr')
            for num_p in [12, 24, 48, 96, 192, 384]:
                first = 0
                for k in sorted(formatted_data[dataset].keys()):
                    # algorithm = 'cusparse'
                    # "total_gpus"

                    component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                         'duration_accumulate', 'duration_barrier', 'total_time']
                    times = {}
                    for component in component_timings:
                        if num_p in formatted_data[dataset][k][algorithm]:
                            times[component] = np.median(formatted_data[dataset][k][algorithm][num_p][component])
                        else:
                            times[component] = ''
                    # print(times)
                    num_gpus = num_p
                    if first:
                        num_gpus = ''
                    print('%s,%s,%s,%s,%s,%s' % (num_gpus, k, times['total_time'], times['duration_issue'] + times['duration_sync'] + times['duration_accumulate'], times['duration_compute'], times['duration_barrier']))
                    first += 1

def graph_some_components(formatted_data, fname='out.png'):
    k = 256
    my_datasets = ['amazon_large_original_general', 'isolates']
    figure,axis = plt.subplots(1, len(my_datasets), figsize=(4.267*len(my_datasets), 3.2))

    x = 0
    for dataset in ['amazon_large_original_general', 'isolates']:
        print('Plotting %s' % (dataset,))
        algo_labels = []
        gpu_labels = []
        communication = []
        computation = []
        accumulate = []
        load_imbalance = []
        for algorithm in algorithms:
            algo_labels.append(algorithm)
            for num_p in [24, 16, 192, 256]:
                gpu_labels.append(num_p)
                if num_p not in formatted_data[dataset][k][algorithm]:
                    communication.append(0)
                    computation.append(0)
                    accumulate.append(0)
                    load_imbalance.append(0)
                else:
                    comm = np.median(formatted_data[dataset][k][algorithm][num_p]['duration_sync']) + np.median(formatted_data[dataset][k][algorithm][num_p]['duration_issue'])
                    communication.append(comm)
                    computation.append(np.median(formatted_data[dataset][k][algorithm][num_p]['duration_compute']))
                    accumulate.append(np.median(formatted_data[dataset][k][algorithm][num_p]['duration_accumulate']))
                    load_imbalance.append(np.median(formatted_data[dataset][k][algorithm][num_p]['duration_barrier']))
                    print('%s,%s,%s,%s: comm %s comp %s acc %s barrier %s' % (dataset,k,algorithm,num_p,communication[-1],computation[-1],accumulate[-1],load_imbalance[-1]))
        print(communication)

        domain = np.array(range(1, len(communication)+1))
        width = 0.35
        bottom = np.array([0]*len(communication), dtype=np.float64)
        widths = width*np.array(len(domain))

        axis[x].bar(domain, communication, width=widths, bottom=bottom)
        bottom += communication
        axis[x].bar(domain, computation, width=widths, bottom=bottom)

        x += 1
    plt.savefig(fname)


# graph_spmm_components(formatted_data, 'out.png')
graph_spmm_times(formatted_data, 'out.png')
# print_data(formatted_data)

# graph_some_components(formatted_data, 'out.png')

import pickle
pickle.dump(formatted_data, open('summit_spmm.pickle', 'wb'))
