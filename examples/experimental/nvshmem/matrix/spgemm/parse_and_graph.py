#!/usr/bin/env python3

import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# algorithms = ["cowns", "aowns", "aownsws", "mpi", "petsc"]
algorithms = ["cowns", "aowns", "aownsws", "xowns", "axowns", "mpi", "petsc"]
algorithms = ["cowns", "aowns", "aownsws", "xowns", "axowns", "mpi"]
# algorithms = ["cowns", "aowns", "aownsws", "xowns", "axowns", "mpi"]

algorithm_relabel = {'cusparse': 'S-C RDMA', 'cowns': 'S-C RDMA', 'aowns' : 'S-A RDMA', 'ws' : 'R WS S-A RDMA', 'aownsws': 'R WS S-A RDMA', 'mpi': 'SUMMA MPI', 'petsc': 'PETSc GPU', 'comblas': 'CombBLAS GPU', 'xowns': 'LA WS S-C RDMA', 'priorityws': 'Priority', 'axowns': 'LA WS S-A RDMA'}

rec_dd = lambda: defaultdict(rec_dd)
formatted_data = rec_dd()

# datasets = ['ldoor_sorted', 'mouse_gene_sorted', 'nlpkkt160_sorted', 'subgraph4']
# REMOVE ldoor
datasets = ['mouse_gene_sorted', 'nlpkkt160_sorted', 'subgraph4']
dataset_replacement = {'amazon_large_original_general': 'Amazon Large, Original', 'isolates': 'Isolates, Subgraph2', 'subgraph4': 'Isolates, Subgraph 4', 'ldoor_sorted': 'ldoor', 'mouse_gene_sorted': 'Mouse Gene', 'nlpkkt160_sorted': 'nlpkkt160'}

color_map = {'comblas': 'C0', 'petsc': 'C0', 'mpi': 'C1', 'cusparse': 'C2', 'cowns': 'C2', 'aowns': 'C3', 'ws': 'C4', 'aownsws': 'C4', 'xowns': 'C5', 'axowns': 'C6'}

legend_order = {'comblas': 0, 'petsc': 0, 'mpi': 1, 'cusparse': 2, 'cowns': 2, 'aowns': 3, 'ws': 4, 'aownsws': 4, 'xowns': 5, 'axowns': 6}

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
              m = re.match('jsrun .*-n (\d+) -a (\d+) .+? ./.+? .+?', line)
              if m:
                  nnodes = int(m.group(1))
                  gpus_per_node = int(m.group(2))
                  total_gpus = nnodes*gpus_per_node

              m = re.match('Matrix multiply finished in (.+?) s', line)
              if m:
                  total_time = float(m.group(1))
                  # formatted_data[algorithm][total_gpus].append((k, total_time, {}))
                  formatted_data[dataset][algorithm][total_gpus]["total_time"] = total_time
                  print('recording %s %s %s %s = %s' % (dataset, algorithm, total_gpus, component, total_time))

              component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                   'duration_accumulate', 'duration_barrier']

              for component in component_timings:
                  m = re.match('%s (.+?) \((.+?) -> (.+?)\)' % (component,), line)
                  if m:
                      time = float(m.group(1))
                      min_time = float(m.group(2))
                      max_time = float(m.group(3))
                      # formatted_data[algorithm][total_gpus][-1][2][component] = time
                      formatted_data[dataset][algorithm][total_gpus][component] = time

        os.chdir('..')

ks = set()
for dataset in formatted_data.keys():
    for k in formatted_data[dataset].keys():
        ks.add(k)
print('ks: %s' % (ks,))

def graph_spmm_times(formatted_data, fname = 'out.pdf'):
    ngraphs = len(formatted_data.keys())
    # figure, axis = plt.subplots(1, len(formatted_data.keys()), sharex=True, figsize=(4.267*ngraphs, 3.2))
    figure, axis = plt.subplots(len(formatted_data.keys()), 1, sharex=True, figsize=(4.267, 3.2*ngraphs))

    print('iterating through...')
    y = 0
    for dataset in sorted(formatted_data.keys()):
        print(dataset)
        for algorithm in sorted(formatted_data[dataset].keys(), key=lambda x: legend_order[x]):
            times = [formatted_data[dataset][algorithm][x]["total_time"] for x in sorted(formatted_data[dataset][algorithm].keys())]
            ngpus = sorted(formatted_data[dataset][algorithm].keys())
            print('%s, %s: %s' % (dataset, algorithm, times))
            print('%s vs %s' % (len(ngpus), len(times)))


            dataset_title = dataset
            if dataset_title in dataset_replacement:
                dataset_title = dataset_replacement[dataset_title]
            if len(axis.shape) == 2:
                axis[y, x].set_title(dataset_title)
            else:
                axis[y].set_title(dataset_title)

            if len(axis.shape) == 2:
              axis[y, x].loglog(ngpus, times, label=algorithm_relabel[algorithm], marker='s', color=color_map[algorithm], markerfacecolor='w')
              # axis[y, x].set_ybound(0.03, 3.5)
              domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
              axis[y, x].minorticks_off()
              axis[y, x].set_xticks(domain)
              axis[y, x].set_xticklabels([str(x) for x in domain])
            else:
              axis[y].loglog(ngpus, times, label=algorithm_relabel[algorithm], marker='s', color=color_map[algorithm], markerfacecolor='w')
              # axis[y, x].set_ybound(0.03, 3.5)
              domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
              axis[y].minorticks_off()
              axis[y].set_xticks(domain)
              axis[y].set_xticklabels([str(x) for x in domain])

            yticks = [0.025, 0.1, 0.4, 1.6, 6.4, 6.4*4]
            if len(axis.shape) == 2:
                axis[y, x].set_yticks(yticks)
                axis[y, x].set_yticklabels([str(x) for x in yticks])
            else:
                axis[y].set_yticks(yticks)
                axis[y].set_yticklabels([str(x) for x in yticks])
                axis[y].set_xlabel('Number of GPUs')
            if y == 0:
                axis[y].set_ylabel('Runtime (s)')
        y += 1

    if len(axis.shape) == 2:
        axis[0, 0].legend(loc='best')
    else:
        axis[0].legend(loc='best')
    plt.tight_layout()

    plt.savefig(fname)

component_relabel = {'duration_issue': 'Issue', 'duration_sync': 'Sync', 'duration_compute': 'Compute', 'duration_accumulate': 'Accumulate', 'duration_barrier': 'Barrier', 'mem_alloc': 'Mem Alloc'}

def graph_spmm_components(formatted_data, fname = 'out.pdf'):
    ngraphs = len(formatted_data.keys())
    figure, axis = plt.subplots(len(ks), len(formatted_data.keys()), sharex=False, figsize=(4.267*ngraphs, len(ks)*3.2))

    bounds = {}
    for dataset in sorted(formatted_data.keys()):
        bounds[dataset] = []
        for algorithm in formatted_data[dataset].keys():
            domain = sorted(formatted_data[dataset][algorithm].keys())
            min_time = min([formatted_data[dataset][algorithm][x]['total_time'] for x in domain])
            max_time = max([formatted_data[dataset][algorithm][x]['total_time'] for x in domain])
            if len(bounds[dataset]) == 0:
                bounds[dataset] = [min_time, max_time]
            else:
                bounds[dataset][0] = min(bounds[dataset][0], min_time)
                bounds[dataset][1] = max(bounds[dataset][1], max_time)
            print('%s, (%s -> %s)' % (dataset, bounds[dataset][0], bounds[dataset][1]))

    print('iterating through...')
    y = 0
    for dataset in sorted(formatted_data.keys()):
        print(dataset)
        x = 0
        for algorithm in formatted_data[dataset].keys():
            # algorithm = 'cusparse'
            # "total_gpus"
            domain = sorted(formatted_data[dataset][algorithm].keys())
            times = [formatted_data[dataset][algorithm][x]["total_time"] for x in domain]

            component_timings = ['duration_issue', 'duration_sync', 'duration_compute',
                                 'duration_accumulate', 'duration_barrier']


            print('Working with axis %s' % (axis.shape,))
            dataset_title = dataset
            if dataset_title in dataset_replacement:
                dataset_title = dataset_replacement[dataset_title]
            if x == 0:
                if len(axis.shape) == 2:
                    axis[x, y].set_title(dataset_title)
                else:
                    axis[x].set_title(dataset_title)

            durations = {}
            width = 0.35
            bottom = np.array([0]*len(domain), dtype=np.float64)
            widths = width*np.array(domain)
            for component in component_timings:
                times = [formatted_data[dataset][algorithm][x][component] for x in domain]
                print('%s: %s' % (component, times,))
                # axis[y, x].loglog(domain, times, label=component)

                component_label = component
                if component_label in component_relabel:
                    component_label = component_relabel[component_label]

                if len(axis.shape) == 2:
                    axis[x, y].bar(domain, times, width=widths, bottom=bottom, label=component_label)
                else:
                    axis[y].bar(domain, times, width=widths, bottom=bottom, label=component_label)
                bottom += times
            times = np.array([formatted_data[dataset][algorithm][x]['total_time'] for x in domain]) - bottom
            if len(axis.shape) == 2:
                axis[x, y].bar(domain, times, width=widths, bottom=bottom, label='mem_alloc')
                axis[x, y].set_xscale('log')
                # plt.axis((x1,x2,25,250))
                x1,x2,y1,y2 = axis[x, y].axis()
                # axis[x, y].axis((x1, x2, y1, bounds[dataset][1] + 0.1*bounds[dataset][1]))
                axis[x, y].set_ylim(top=bounds[dataset][1])
                print('SETTING %s limit at around %s' % (dataset, bounds[dataset][1]))
            else:
                axis[y].bar(domain, times, width=widths, bottom=bottom, label='mem_alloc')
                axis[y].set_xscale('log')

            if y == 0:
                if len(axis.shape) == 2:
                    axis[x, y].set_ylabel("%s\n\nRuntime (s)" % (algorithm_relabel[algorithm],))
                else:
                    axis[y].set_ylabel("%s\n\nRuntime (s)" % (algorithm_relabel[algorithm],))

            if len(axis.shape) == 2:
                if x == axis.shape[1]-1:
                    axis[x, y].set_xlabel('Number of GPUs')
            else:
                axis[y].set_xlabel('Number of GPUs')

            if len(axis.shape) == 2:
                print('Setting %s, %s' % (x, y))
                axis[x, y].minorticks_off()
                axis[x, y].set_xticks(6*np.array([1, 2, 4, 8, 16, 32, 64]))
                axis[x, y].set_xticklabels(6*np.array([1, 2, 4, 8, 16, 32, 64]))
                # axis[x, y].legend(loc='best')

            x += 1
        y += 1

    if len(axis.shape) == 2:
        axis[0, 0].minorticks_off()
        axis[0, 0].set_xticks(6*np.array([1, 2, 4, 8, 16, 32, 64]))
        axis[0, 0].set_xticklabels(6*np.array([1, 2, 4, 8, 16, 32, 64]))
        axis[0, 0].legend(loc='best')
    else:
        axis[0].minorticks_off()
        axis[0].set_xticks(6*np.array([1, 2, 4, 8, 16, 32, 64]))
        axis[0].set_xticklabels(6*np.array([1, 2, 4, 8, 16, 32, 64]))
        axis[0].legend(loc='best')
    plt.tight_layout()

    plt.savefig(fname)

def graph_some_components(formatted_data, fname='out.png'):
    my_datasets = ['mouse_gene_sorted']
    figure,axis = plt.subplots(1, len(my_datasets), figsize=(4.267*len(my_datasets), 3.2))
    # figure,axis = plt.subplots(len(my_datasets), 1, figsize=(3.2, 4.267*len(my_datasets)))

    x = 0
    for dataset in my_datasets:
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
                if num_p not in formatted_data[dataset][algorithm]:
                    communication.append(0)
                    computation.append(0)
                    accumulate.append(0)
                    load_imbalance.append(0)
                else:
                    comm = np.median(formatted_data[dataset][algorithm][num_p]['duration_sync']) + np.median(formatted_data[dataset][algorithm][num_p]['duration_issue'])
                    communication.append(comm)
                    computation.append(np.median(formatted_data[dataset][algorithm][num_p]['duration_compute']))
                    accumulate.append(np.median(formatted_data[dataset][algorithm][num_p]['duration_accumulate']))
                    load_imbalance.append(np.median(formatted_data[dataset][algorithm][num_p]['duration_barrier']))
                    print('%s,%s,%s: comp %.2lf comm %.2lf acc %.2lf barrier %.2lf' % (dataset,algorithm,num_p,computation[-1],communication[-1],accumulate[-1],load_imbalance[-1]))
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


print(formatted_data)

# graph_spmm_components(formatted_data, 'out.png')
graph_spmm_times(formatted_data, 'out.png')
# graph_some_components(formatted_data, 'out.png')

'''
domain = np.array([1, 2, 4, 8, 16, 32, 64])*6
for p in domain:
    if p in formatted_data['subgraph4']['cowns']:
        total_time = formatted_data['subgraph4']['cowns'][p]['total_time']
        print(formatted_data['subgraph4']['cowns'][p].keys())
        duration_barrier = formatted_data['subgraph4']['cowns'][p]['duration_barrier']
        print('%s: %s (%s - %s)' % (p, total_time - duration_barrier, total_time, duration_barrier))
'''
