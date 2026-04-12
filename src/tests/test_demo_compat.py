from pathlib import Path
import re

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from causallift import CausalLift, generate_data

ROOT = Path(__file__).resolve().parents[2]


def read_text(filename: str) -> str:
    return (ROOT / filename).read_text()


def read_lines(filename: str) -> list[str]:
    return [line.strip() for line in (ROOT / filename).read_text().splitlines() if line.strip()]


def read_setup_version() -> str:
    setup_py = (ROOT / "setup.py").read_text()
    match = re.search(r'version="([^"]+)"', setup_py)
    assert match is not None
    return match.group(1)


def test_runtime_requirements_allow_flexible_versions_except_pipelinex():
    requirements = read_lines("requirements.txt")
    assert set(requirements) == {
        "pandas",
        "scikit-learn",
        "numpy",
        "easydict",
        "kedro",
        "kedro-datasets",
        "pipelinex>=0.8.0",
    }

    assert "numpy" in requirements
    assert "easydict" in requirements


def test_setup_py_supports_python_310_to_313_only():
    setup_py = (ROOT / "setup.py").read_text()

    assert "Programming Language :: Python :: 3.10" in setup_py
    assert "Programming Language :: Python :: 3.11" in setup_py
    assert "Programming Language :: Python :: 3.12" in setup_py
    assert "Programming Language :: Python :: 3.13" in setup_py
    assert "Programming Language :: Python :: 3.9" not in setup_py
    assert "Programming Language :: Python :: 3.8" not in setup_py
    assert "Programming Language :: Python :: 3.7" not in setup_py
    assert "Programming Language :: Python :: 3.14" not in setup_py


def test_setup_py_declares_supported_python_requires():
    setup_py = (ROOT / "setup.py").read_text()

    assert 'python_requires=">=3.10,<3.14"' in setup_py


def test_github_actions_ci_targets_supported_python_matrix():
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)

    assert set(workflow["on"].keys()) == {"push", "pull_request"}
    assert workflow["jobs"]["test"]["strategy"]["fail-fast"] == "false"

    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert matrix == ["3.10", "3.11", "3.12", "3.13"]

    steps = workflow["jobs"]["test"]["steps"]
    install_step = next(step for step in steps if step.get("name") == "Install package and test dependencies")
    assert install_step["name"] == "Install package and test dependencies"
    assert install_step["run"].splitlines() == [
        "pip install -r requirements.txt",
        "pip install -r requirements_dev.txt",
        "pip install -e .",
    ]

    assert any(step.get("run") == "pytest -ra" for step in steps)


def test_readme_documents_supported_python_and_modern_stack():
    readme = read_text("README.md")

    assert "Python 3.10" in readme
    assert "Python 3.11" in readme
    assert "Python 3.12" in readme
    assert "Python 3.13" in readme
    assert "Python 3.9" not in readme
    assert "Python 3.8" not in readme

    assert "- `pandas`" in readme
    assert "- `scikit-learn`" in readme
    assert "- `kedro`" in readme
    assert "- `kedro-datasets`" in readme
    assert "- `pipelinex>=0.8.0`" in readme

    assert "pip install -r requirements.txt" in readme
    assert "requirements_dev.txt" in readme


def test_notebook_install_instructions_match_supported_stack():
    import json

    notebook = json.loads(read_text("notebooks/demo/CausalLift_demo.ipynb"))
    expected_version = read_setup_version()
    metadata = notebook["metadata"]
    language_version = metadata["language_info"]["version"]
    major_minor = tuple(int(part) for part in language_version.split(".")[:2])

    assert metadata["kernelspec"]["name"] == "python3"
    assert metadata["kernelspec"]["display_name"] == "Python 3"
    assert metadata["language_info"]["name"] == "python"
    assert major_minor >= (3, 10)
    assert major_minor != (3, 6)

    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

    install_cells = [source for source in sources if "pip install" in source]
    assert len(install_cells) == 1
    assert any("pandas==3.0.2" in source for source in install_cells)
    assert any("scikit-learn==1.8.0" in source for source in install_cells)
    assert any("kedro==1.3.1" in source for source in install_cells)
    assert any("kedro-datasets==9.3.0" in source for source in install_cells)
    assert any("pipelinex==0.8.0" in source for source in install_cells)
    assert any("causallift" in source for source in install_cells)
    assert all("fallback" not in source.lower() for source in sources)
    assert all("pip install -e ." not in source for source in sources)
    assert all("git clone" not in source for source in sources)
    assert all("setup.py develop" not in source for source in sources)
    assert expected_version in read_text("notebooks/demo/CausalLift_demo.ipynb")


def test_sphinx_docs_do_not_contradict_supported_stack():
    index = read_text("docs/source/index.rst")
    content00 = read_text("docs/source/contents/00.md")
    content07 = read_text("docs/source/contents/07.md")
    content12 = read_text("docs/source/contents/12.md")

    for text in (index, content00, content07, content12):
        assert "Python 3.5" not in text
        assert "Python 3.6" not in text
        assert "Python 3.7" not in text
        assert "kedro==0.17.7" not in text
        assert "scikit-learn==0.21.3" not in text

    assert "Python 3.10" in content12
    assert "Python 3.13" in content12
    assert "pip install causallift" in content07
    assert "pip install -r requirements.txt" in content07
    assert "pip install -r requirements_dev.txt" in content07
    assert "pip install -e ." in content07
    assert "- pandas" in content07
    assert "- scikit-learn" in content07
    assert "- kedro" in content07
    assert "- kedro-datasets" in content07
    assert "- pipelinex>=0.8.0" in content07


def test_package_imports_on_supported_stack():
    import causallift

    assert causallift.__version__ == read_setup_version()


def test_notebook_smoke_flow_executes_on_supported_stack():
    df = generate_data(
        N=200,
        n_features=3,
        beta=[0, -2, 3, -5],
        error_std=0.1,
        tau=[1, -5, -5, 10],  # type: ignore[arg-type]
        tau_std=0.1,  # type: ignore[arg-type]
        discrete_outcome=True,
        seed=0,
        feature_effect=0,
        propensity_coef=[0, -1, 1, -1],
        index_name="index",
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["Treatment"]
    )

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=0, runner=None)  # type: ignore[arg-type]
    train_out, test_out = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()

    assert isinstance(train_out, pd.DataFrame)
    assert isinstance(test_out, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)
