# src/metrics/__init__.py
# (intentional: expose small convenience surface)
from .inequality import gini, welfare, status_share
from .network_stats import income_homophily, degree_assortativity, clustering_coefficients
from .logging import compute_step_metrics, write_panel, write_step_summary