#!/usr/bin/env python3

import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict

algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns"]
algorithms = ["cusparse", "aowns", "ws", "mpi", "comblas", "xowns", "priorityws"]
# algorithms = ["cusparse", "aowns", "ws", "mpi", "xowns"]

algorithm_relabel = {'cusparse': 'S-C RDMA', 'aowns' : 'S-A RDMA', 'ws' : 'WS S-A RDMA', 'mpi': 'SUMMA MPI', 'comblas': 'CombBLAS GPU', 'xowns': 'X-Owns WS', 'priorityws': 'Priority'}

rec_dd = lambda: defaultdict(rec_dd)
formatted_data = rec_dd()

# datasets = ['Nm7', 'Nm8', 'amazon_large_original_general', 'isolates']
# datasets = ['Nm7', 'Nm8', 'amazon_large_original_general', 'amazon_large_randomized_general', 'isolates', 'com-Orkut_general', 'reddit_general']
# REMOVE Nm7
# datasets = ['Nm8', 'amazon_large_original_general', 'isolates']
datasets = ['Nm7', 'Nm8', 'amazon_large_original_general']

dataset_replacement = {'amazon_large_original_general': 'Amazon Large, Original', 'isolates': 'Isolates, Subgraph2'}

for algorithm in algorithms:
    directories = glob.glob(algorithm + '_*/')
    print(algorithm)

    for directory in directories:
        dataset = directory[len(algorithm)+1:-1]
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
                  formatted_data[dataset][k][algorithm][total_gpus]["total_time"] = total_time
                  print('recording %s %s %s %s %s = %s' % (dataset, k, algorithm, total_gpus, 'total_time', total_time))

              component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                   'duration_accumulate', 'duration_barrier']

              for component in component_timings:
                  m = re.match('%s (.+?) \((.+?) -> (.+?)\)' % (component,), line)
                  if m:
                      time = float(m.group(1))
                      min_time = float(m.group(2))
                      max_time = float(m.group(3))
                      # formatted_data[algorithm][total_gpus][-1][2][component] = time
                      formatted_data[dataset][k][algorithm][total_gpus][component] = time

        os.chdir('..')

ks = set()
for dataset in formatted_data.keys():
    for k in formatted_data[dataset].keys():
        ks.add(k)

# REMOVE everything but 128, 512
ks = [128, 512]

def graph_spmm_times(formatted_data, fname='out.pdf'):
    figure, axis = plt.subplots(len(ks), len(formatted_data.keys()), figsize=(4.267*len(formatted_data.keys()), 3.2*len(ks)))

    bounds = {}
    for dataset in sorted(formatted_data.keys()):
        bounds[dataset] = []
        for k in ks:
            for algorithm in formatted_data[dataset][k].keys():
                domain = sorted(formatted_data[dataset][k][algorithm].keys())
                print('%s, %s' % (dataset, algorithm))
                min_time = min([formatted_data[dataset][k][algorithm][x]['total_time'] for x in domain])
                max_time = max([formatted_data[dataset][k][algorithm][x]['total_time'] for x in domain])
                if len(bounds[dataset]) == 0:
                    bounds[dataset] = [min_time, max_time]
                else:
                    bounds[dataset][0] = min(bounds[dataset][0], min_time)
                    bounds[dataset][1] = max(bounds[dataset][1], max_time)
                print('%s, (%s -> %s)' % (dataset, bounds[dataset][0], bounds[dataset][1]))

    print('iterating through...')
    x = 0
    for dataset in sorted(formatted_data.keys()):
        print(dataset)
        y = 0
        for k in ks:
            for algorithm in formatted_data[dataset][k].keys():
                times = [formatted_data[dataset][k][algorithm][x]["total_time"] for x in sorted(formatted_data[dataset][k][algorithm].keys())]
                ngpus = sorted(formatted_data[dataset][k][algorithm].keys())
                print('%s, %s, %s: %s' % (dataset, algorithm, k, times))
                print('%s vs %s' % (len(ngpus), len(times)))

                if y == 0:
                    dataset_title = dataset
                    if dataset_title in dataset_replacement:
                        dataset_title = dataset_replacement[dataset_title]
                    axis[y, x].set_title(dataset_title)

                axis[y, x].loglog(ngpus, times, label=algorithm_relabel[algorithm], marker='s', markerfacecolor='w')
                # axis[y, x].set_ybound(0.03, 3.5)
                domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
                axis[y, x].minorticks_off()
                axis[y, x].set_xticks(domain)
                axis[y, x].set_xticklabels([str(x) for x in domain])

                if y == len(axis)-1:
                    axis[y, x].set_xlabel("Number of GPUs")
                    pass

                yticks = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 6.4*2]
                # yticks = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
                axis[y, x].set_yticks(yticks)
                print('%s setting ylim(%s, %s)' % (dataset, bounds[dataset][0]*0.9, bounds[dataset][1]*1.1))
                axis[y, x].set_ylim(bottom=bounds[dataset][0]*0.9, top=bounds[dataset][1]*1.1)
                axis[y, x].set_yticklabels([str(x) for x in yticks])
                if x == 0:
                    axis[y, x].set_ylabel("N = %s\n\nRuntime(s)" % (k,))
            y += 1
        x += 1

    axis[0, 0].legend(loc='best')
    plt.tight_layout()

    plt.savefig(fname)

component_relabel = {'duration_issue': 'Issue', 'duration_sync': 'Sync', 'duration_compute': 'Compute', 'duration_accumulate': 'Accumulate', 'duration_barrier': 'Barrier', 'mem_alloc': 'Mem Alloc'}

def graph_spmm_components(formatted_data, fname='out.pdf'):
    # figure, axis = plt.subplots(len(algorithms), len(formatted_data.keys()), sharex=True, figsize=(25.6, 12.8))
    # figure, axis = plt.subplots(len(algorithms), len(datasets), figsize=(4.267*len(formatted_data.keys()), 3.2*len(ks)))
    figure, axis = plt.subplots(len(algorithms), len(datasets), figsize=(4.267*len(ks), 3.2*len(formatted_data.keys())))

    bounds = {}
    for dataset in sorted(formatted_data.keys()):
        # REMOVE, CHANGE k
        k = 256
        # k = 256
        bounds[dataset] = []
        for algorithm in formatted_data[dataset][k].keys():
            domain = sorted(formatted_data[dataset][k][algorithm].keys())
            print('%s, %s' % (dataset, algorithm))
            min_time = min([formatted_data[dataset][k][algorithm][x]['total_time'] for x in domain])
            max_time = max([formatted_data[dataset][k][algorithm][x]['total_time'] for x in domain])
            if len(bounds[dataset]) == 0:
                bounds[dataset] = [min_time, max_time]
            else:
                bounds[dataset][0] = min(bounds[dataset][0], min_time)
                bounds[dataset][1] = max(bounds[dataset][1], max_time)
            print('%s, (%s -> %s)' % (dataset, bounds[dataset][0], bounds[dataset][1]))

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
            times = [formatted_data[dataset][k][algorithm][x]["total_time"] for x in domain]

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
                            total_time = formatted_data[dataset][k_][algorithm][p]['total_time']
                            duration_barrier = formatted_data[dataset][k_][algorithm][p]['duration_barrier']
                            print('%s %s (%s total, %s barrier) ' % (p, total_time - duration_barrier, total_time, duration_barrier))
                        '''
                        for x in domain:
                            print(x)
                        '''
                times = [formatted_data[dataset][k][algorithm][x][component] for x in domain]
                print('%s: %s' % (component, times,))
                # axis[y, x].loglog(domain, times, label=component)
                component_label = component
                if component_label in component_relabel:
                    component_label = component_relabel[component_label]
                axis[y, x].bar(domain, times, width=widths, bottom=bottom, label=component_label)
                bottom += times
            times = np.array([formatted_data[dataset][k][algorithm][x]['total_time'] for x in domain]) - bottom
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

            y += 1
        x += 1

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
            for np in [12, 24, 48, 96, 192, 384]:
                first = 0
                for k in sorted(formatted_data[dataset].keys()):
                    # algorithm = 'cusparse'
                    # "total_gpus"

                    component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                         'duration_accumulate', 'duration_barrier', 'total_time']
                    times = {}
                    for component in component_timings:
                        if np in formatted_data[dataset][k][algorithm]:
                            times[component] = formatted_data[dataset][k][algorithm][np][component]
                        else:
                            times[component] = ''
                    # print(times)
                    num_gpus = np
                    if first:
                        num_gpus = ''
                    print('%s,%s,%s,%s,%s,%s' % (num_gpus, k, times['total_time'], times['duration_issue'] + times['duration_sync'] + times['duration_accumulate'], times['duration_compute'], times['duration_barrier']))
                    first += 1

# graph_spmm_components(formatted_data, 'out.png')
graph_spmm_times(formatted_data, 'out.png')
print_data(formatted_data)
