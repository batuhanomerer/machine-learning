"""Run both notebooks in fresh kernels and rebuild the six combined result files."""
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
import pandas as pd
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
DATASETS = ["breast_cancer", "wine"]
ROW_COUNTS = {
    "classifier_runs": 132, "classifier_summary": 44, "classifier_test": 2,
    "kmeans_runs": 12, "kmeans_summary": 4, "kmeans_test": 4,
}


def execute_project(evidence_dir=None):
    with TemporaryDirectory(prefix="midterm_") as directory:
        exports = Path(directory)
        kernel_env = {**os.environ, "MIDTERM_EXPORT_DIR": directory, "MPLBACKEND": "Agg",
                      "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        clients = {}
        with ExitStack() as kernels:
            for dataset in DATASETS:
                notebook = nbformat.read(ROOT / f"{dataset}.ipynb", as_version=4)
                manager = KernelManager(kernel_name="python3")
                # Use the Python environment that launched this script.
                manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
                client = NotebookClient(notebook, km=manager, timeout=180)
                client.reset_execution_trackers()
                kernels.enter_context(client.setup_kernel(cwd=str(ROOT), env=kernel_env, cleanup_kc=True))
                clients[dataset] = client
            # Finish validation and freeze both selections before executing either final test cell.
            for phase in ("validation", "test"):
                for dataset, client in clients.items():
                    print(f"{dataset}: {phase}", flush=True)
                    for index, cell in enumerate(client.nb.cells):
                        if cell.cell_type == "code" and cell.metadata.phase == phase:
                            client.execute_cell(cell, index)

        for name, expected in ROW_COUNTS.items():
            frames = [pd.read_csv(exports / f"{dataset}_{name}.csv", keep_default_na=False)
                      for dataset in DATASETS]
            combined = pd.concat(frames, ignore_index=True)
            assert len(combined) == expected
            assert combined.groupby("dataset").size().eq(expected // 2).all()
            combined.to_csv(ROOT / f"{name}.csv", index=False)
        for dataset, client in clients.items():
            nbformat.validate(client.nb)
            nbformat.write(client.nb, ROOT / f"{dataset}.ipynb")
        if evidence_dir is not None:
            for dataset in DATASETS:
                shutil.copy2(exports / f"{dataset}_evidence.json", evidence_dir)
        print("132 classifier fits, 12 K-Means fits, six result CSVs.")


if __name__ == "__main__":
    execute_project()
