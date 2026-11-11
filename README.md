# Machine Learning Midterm

The two notebooks compare Decision Tree, KNN, Gaussian Naive Bayes and K-Means on Breast Cancer Wisconsin and Wine Classification. The classification models were selected on validation macro-F1 and evaluated on fixed test rows using the fitted seed-42 instances.

## Run the project

The submitted outputs were generated with Python 3.12.14 and the exact package versions in `requirements.txt`. Extract the ZIP and open a terminal in the extracted project folder. With Python 3.12 installed, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_project.py
```

On macOS or Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` for the last two commands.

The script starts two fresh kernels. It runs both validation sections before either notebook reaches the final test evaluations, then refreshes both notebooks, the six combined CSVs and the PNGs. The Word report describes the submitted run. To work interactively, open either notebook in VS Code with the Python and Jupyter extensions, select the same virtual environment as the kernel and use **Run All**. To use Google Colab, upload either .ipynb file and choose Runtime > Run all. Each notebook includes its own setup and loads its dataset from scikit-learn.

## Files

| File | Contents |
| --- | --- |
| `breast_cancer.ipynb`, `wine.ipynb` | Executed notebooks for Sections 1–10 |
| `report.docx` | Report, figures, validation summaries and conclusions |
| `classifier_runs.csv` | 132 classifier fits |
| `classifier_summary.csv` | 44 configurations with three-seed means and sample SDs |
| `classifier_test.csv` | Two final seed-42 evaluations |
| `kmeans_runs.csv` | 12 clustering fits |
| `kmeans_summary.csv` | Four representation summaries with valid-seed counts |
| `kmeans_test.csv` | Four final seed-42 clustering evaluations |
| `figures/` | 21 PNGs used in the notebooks and report |
| `helpers.py`, `run_project.py` | Shared notebook functions and the paired runner |
| `AI_DISCLOSURE.md` | AI and tool-use disclosure |

## Sources and submission links

- [Breast Cancer Wisconsin loader and dataset source](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Wine Classification loader and dataset source](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_wine.html)
- [GitHub repository](https://github.com/batuhanomerer/machine-learning)
- [Walkthrough video](https://youtu.be/7TwU40EB4kI)
