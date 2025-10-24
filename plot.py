import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import random
import copy

import time as t
from collections import defaultdict

from utils import *

if __name__ == "__main__":
    pass

    # # Plot of classical attacks
    # deg_dict = read_json(path("attack_degree.json"))["content"]["paris"]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["paris"]
    # plt.figure()
    # plt.plot(bet_dict["static"]["cost"], bet_dict["static"]["LCC metric"], label=f'static-BCA')
    # plt.plot(deg_dict["static"]["cost"], deg_dict["static"]["LCC metric"], label=f'static-DA')
    # plt.plot(bet_dict["dynamic"]["cost"], bet_dict["dynamic"]["LCC metric"], label=f'dynamic-BCA')
    # plt.plot(deg_dict["dynamic"]["cost"], deg_dict["dynamic"]["LCC metric"], label=f'dynamic-DA')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_classicals_LCC.png"), dpi = 300)
    # plt.close()
    # plt.figure()
    # plt.plot(bet_dict["static"]["cost"][:151], bet_dict["static"]["efficiency"], label=f'static-BCA')
    # plt.plot(deg_dict["static"]["cost"][:151], deg_dict["static"]["efficiency"], label=f'static-DA')
    # plt.plot(bet_dict["dynamic"]["cost"][:151], bet_dict["dynamic"]["efficiency"], label=f'dynamic-BCA')
    # plt.plot(deg_dict["dynamic"]["cost"][:151], deg_dict["dynamic"]["efficiency"], label=f'dynamic-DA')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_classicals_efficiency.png"), dpi = 300)
    # plt.close()

    # # Plot of CAs
    # city = "paris"
    # k = 2
    # imb = 0.1
    # max_displayed_cost = 200
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # plt.plot(ca_dict["static"]["CF"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["static"]["CF"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CF-CA')
    # plt.plot(ca_dict["static"]["random"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["static"]["random"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'random-CA')
    # plt.plot(ca_dict["static"]["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["static"]["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'BC-CA')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, max_displayed_cost)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_bestcut1000_k{k}_imb{imb}_efficiency.png"), dpi = 300)
    # plt.close()

    # # Plot of iterated CAs
    # city = "paris"
    # k = 2
    # max_displayed_cost = 500
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"][city]["dynamic"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for imb in [0.03, 0.1]:
    #     if imb == 0.03:
    #         plt.plot(ca_dict["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["dynamic"]["random3"][f"k={k}, imbalance={imb}"]["efficiency"], label=fr'random3-CA: $\epsilon={imb}$')
    #     plt.plot(ca_dict["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["dynamic"]["BC2"][f"k={k}, imbalance={imb}"]["efficiency"], label=fr'BC2-CA: $\epsilon={imb}$')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.xlim(-10, max_displayed_cost)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ca_it_efficiency.png"), dpi = 300)
    # plt.close()

    # # Plot of CFAs
    # city = "paris"
    # cfa_dict = read_json(path("attack_cfa.json"))["content"][city]["static"]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["dynamic"]
    # # Along imbalance
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for imb in [0.03, 0.1, 0.16, 0.22, 0.3]:
    #     plt.plot(cfa_dict[f"k=2, imbalance={imb}"]["cost"][:151], cfa_dict[f"k=2, imbalance={imb}"]["efficiency"], label=fr'CFA: $\epsilon={imb}$')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(imb)_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for imb in [0.03, 0.1, 0.16, 0.22, 0.3]:
    #     plt.plot(cfa_dict[f"k=2, imbalance={imb}"]["cost"], cfa_dict[f"k=2, imbalance={imb}"]["LCC metric"], label=fr'CFA: $\epsilon={imb}$')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(imb)_LCC.png"), dpi = 300)
    # # Along k
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for k in [2, 3, 4, 5, 6]:
    #     plt.plot(cfa_dict[f"k={k}, imbalance=0.03"]["cost"][:151], cfa_dict[f"k={k}, imbalance=0.03"]["efficiency"], label=fr'CFA: $k={k}$')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(k)_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for k in [2, 3, 4, 5, 6]:
    #     plt.plot(cfa_dict[f"k={k}, imbalance=0.03"]["cost"], cfa_dict[f"k={k}, imbalance=0.03"]["LCC metric"], label=fr'CFA: $k={k}$')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(k)_LCC.png"), dpi = 300)
    # plt.close()
    # # Along n
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # for n in [10, 100, 1000, 10000]:
    #     plt.plot(cfa_dict[f"n={n}"]["cost"][:151], cfa_dict[f"n={n}"]["efficiency"], label=fr'CFA: $n={n}$')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(n)_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for n in [10, 100, 1000, 10000]:
    #     plt.plot(cfa_dict[f"n={n}"]["cost"], cfa_dict[f"n={n}"]["LCC metric"], label=fr'CFA: $n={n}$')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa(n)_LCC.png"), dpi = 300)
    # plt.close()

    # # Plot of dynamic vs static CFA
    # city = "paris"
    # k = 2
    # imb = 0.03
    # max_displayed_costs = [350, 550] # eff, LCC
    # cfa_dict = read_json(path("attack_cfa.json"))["content"][city]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["dynamic"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["static"]
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # plt.plot(ca_dict["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'BC-CA')
    # plt.plot(ca_dict["CF"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["CF"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'CF-CA')
    # for v in ["static", "dynamic"]:
    #     plt.plot(cfa_dict[v][f"k={k}, imbalance={imb}"]["cost"][:151], cfa_dict[v][f"k={k}, imbalance={imb}"]["efficiency"], label=f'{v}-CFA')
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.xlim(-10, max_displayed_costs[0])
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa_dyn_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # for v in ["static", "dynamic"]:
    #     plt.plot(cfa_dict[v][f"k={k}, imbalance={imb}"]["cost"], cfa_dict[v][f"k={k}, imbalance={imb}"]["LCC metric"], label=f'{v}-CFA')
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.xlim(-10, max_displayed_costs[1])
    # plt.tight_layout()
    # plt.savefig(path(f"attack_cfa_dyn_LCC.png"), dpi = 300)
    # plt.close()

    # # Plot of CCFAs
    # city = "paris"
    # k = 2
    # imb = 0.1
    # l = 25000
    # max_displayed_costs = [350, 1500] # eff, LCC
    # cfa_dict = read_json(path("attack_cfa.json"))["content"][city]["static"]
    # bet_dict = read_json(path("attack_betweenness.json"))["content"]["dynamic"]
    # ca_dict = read_json(path("attack_ca.json"))["content"][city]["static"]
    # ccfa_dict = read_json(path("attack_ccfa.json"))["content"] # Old format without the city key
    # cluster_list = read_file(path(f"clusters_birch_md{l}_clean{imb}")) # 'C' is only needed for imb = 0.03
    # plt.figure()
    # plt.plot(bet_dict["cost"][:151], bet_dict["efficiency"], label=f'BCA')
    # plt.plot(ca_dict["BC"][f"k={k}, imbalance={imb}"]["cost"], ca_dict["BC"][f"k={k}, imbalance={imb}"]["efficiency"], label=f'BC-CA')
    # plt.plot(cfa_dict[f"k={k}, imbalance={imb}"]["cost"][:151], cfa_dict[f"k={k}, imbalance={imb}"]["efficiency"], label=f'CFA')
    # for i in range(5):
    #     plt.plot(ccfa_dict[f"{imb}"][f"{l}"][f"{i}"]["cost"][:151], ccfa_dict[f"{imb}"][f"{l}"][f"{i}"]["efficiency"], label=rf"CCFA: $f_{i}={len(cluster_list[i])/1000}$", alpha=0.7)
    # plt.xlabel('cost')
    # plt.ylabel('efficiency')
    # plt.legend()
    # plt.xlim(-10, max_displayed_costs[0])
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ccfa_imb{imb}_l{l}_efficiency.png"), dpi = 300)
    # plt.figure()
    # plt.plot(bet_dict["cost"], bet_dict["LCC metric"], label=f'BCA')
    # plt.plot(cfa_dict[f"k={k}, imbalance={imb}"]["cost"], cfa_dict[f"k={k}, imbalance={imb}"]["LCC metric"], label=f'CFA')
    # for i in range(5):
    #     plt.plot(ccfa_dict[f"{imb}"][f"{l}"][f"{i}"]["cost"], ccfa_dict[f"{imb}"][f"{l}"][f"{i}"]["LCC metric"], label=rf"CCFA: $f_{i}={len(cluster_list[i])/1000}$", alpha=0.7)
    # plt.xlabel('cost')
    # plt.ylabel('LCC metric')
    # plt.legend()
    # plt.xlim(-10, min(max_displayed_costs[1], cfa_dict[f"k={k}, imbalance={imb}"]["cost"][-1]))
    # plt.tight_layout()
    # plt.savefig(path(f"attack_ccfa_imb{imb}_l{l}_LCC.png"), dpi = 300)
    # plt.close()
