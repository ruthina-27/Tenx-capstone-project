# End-to-End Insurance Risk Analytics & Predictive Modeling

This repository contains the code, data, and documentation for a comprehensive insurance analytics capstone project. The focus is on analyzing 18 months of historical car insurance claim data for a fictional insurer to optimize marketing strategies, identify low-risk customer segments, and reduce churn through premium adjustments. The goal is to deliver actionable insights using exploratory data analysis (EDA), data version control (DVC), A/B hypothesis testing, and predictive modeling.

## Project Objectives
The primary objective is to enhance customer retention and profitability by analyzing historical insurance data. Key tasks include:

- **Task 1:** Set up Git, CI/CD, and perform EDA to uncover risk and profitability patterns.
- **Task 2:** Implement DVC for reproducible data management and versioning.
- **Task 3:** Conduct A/B hypothesis testing to identify risk drivers and validate business assumptions.
- **Task 4:** Build predictive models for claim severity and premium optimization, including feature importance analysis.

## Initial Setup
1. Clone the repository and navigate to the project folder.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Restore data files using DVC (`dvc pull`).
5. Run analysis scripts and notebooks for EDA, hypothesis testing, and modeling.

## Key Features & Workflow
- Modular code layout for easy navigation and reproducibility.
- DVC for robust data versioning and auditability.
- Automated CI/CD pipeline for code quality and reproducibility.
- Jupyter notebooks and scripts for EDA, statistical testing, and predictive modeling.
- Visualizations and reports for business insights and recommendations.

## Business Impact
- Identifies actionable risk drivers and customer segments for targeted marketing and premium adjustment.
- Supports regulatory compliance and auditability with DVC and CI/CD.
- Enables dynamic, data-driven pricing strategies to reduce churn and improve profitability.

## Contact
For questions or collaboration, please contact the repository owner via GitHub.

## Structure
- **data/**: Raw and processed datasets (tracked with DVC for reproducibility)
- **notebooks/**: Jupyter notebooks for EDA, hypothesis testing, and modeling
- **src/**: Source code for analysis, modeling, and utilities
- **visualizations/**: Key plots and figures generated during analysis
- **.github/workflows/**: CI/CD pipeline configuration

## Key Tasks
1. **Git & GitHub Setup**: Version control, CI/CD, and project documentation
2. **EDA & Statistics**: Data summarization, quality assessment, outlier detection, and creative visualizations
3. **DVC Data Pipeline**: Reproducible and auditable data management using Data Version Control
4. **Hypothesis Testing**: Statistical validation of risk drivers (province, zip code, gender, margin)
5. **Predictive Modeling**: Building and evaluating models for claim severity, premium optimization, and risk segmentation

## How to Run
1. Clone the repository:
	```
	git clone https://github.com/ruthina-27/Tenx-capstone-project.git
	cd Tenx-capstone-project
	```
2. Create and activate a virtual environment:
	```
	python -m venv venv
	.\venv\Scripts\activate  # On Windows
	```
3. Install dependencies:
	```
	pip install -r requirements.txt
	```
4. Restore data files (if using DVC):
	```
	dvc pull
	```
5. Open and run notebooks in the `notebooks/` folder for analysis and modeling.

## CI/CD
GitHub Actions automatically runs linting and tests on every push to ensure code quality and reproducibility.

## Business Impact
- Provides actionable insights for risk segmentation and premium adjustment
- Enables reproducible, auditable analytics for regulatory compliance
- Supports dynamic, data-driven pricing strategies
