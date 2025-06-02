import datetime
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

# --- Matplotlib Style ---
plt.style.use("seaborn-v0_8-whitegrid")

# --- Configuration ---
FILE_PATH = "1_data_preparation/cleaned_survey_data.csv"
OUTPUT_BASE_DIR = "2_descriptive_analysis"
MARKDOWN_FILE = os.path.join(OUTPUT_BASE_DIR, "2_descriptive_analysis_report.md")

# LLM Categorization Lists
POSITION_CATEGORIES = [
    "Management (Director, VP, Head, Manager)",
    "Engineer/Specialist",
    "Analyst",
    "C-Level (CEO, CFO, CIO, CTO)",
    "Consultant",
    "Researcher/Academic",
    "Operational Staff",
    "Technical Lead/Architect",
    "Data Scientist/ML Engineer",
    "Student/Intern",
    "Project Manager",
    "Quality Assurance/Control",
    "Production Lead/Supervisor",
]
INDUSTRY_SECTOR_CATEGORIES = [
    "Manufacturing (General)",
    "Technology/Software",
    "Automotive",
    "Aerospace",
    "Energy/Utilities",
    "Pharmaceuticals/Healthcare",
    "Consulting",
    "Logistics/Supply Chain",
    "Chemicals",
    "Food & Beverage",
    "Construction",
    "Telecommunications",
    "Financial Services",
]
COUNTRY_CATEGORIES = [
    "USA",
    "Germany",
    "India",
    "UK",
    "Canada",
    "France",
    "China",
    "Brazil",
    "Japan",
    "Australia",
    "Netherlands",
    "Switzerland",
    "Italy",
    "Spain",
    "Sweden",
]
DA_TOOL_CATEGORIES = [  # Generic for maint, qual, prod tools categorization
    "Spreadsheet",
    "ERP System",
    "BI Tool",
    "CMMS (Computerized Maintenance Management System)",
    "Statistical Software/Programming",
    "ML/AI Platform",
    "Custom/In-house System",
    "Database System",
    "SCADA/Historian",
    "Cloud Platform",
]
# These _STANDARD_OPTION lists are used for multi-select analysis to define known choices
AI_GENAI_TOOL_STANDARD_OPTIONS = [
    "ChatGPT (OpenAI)",
    "Copilot (Microsoft)",
    "Gemini (Google)",
    "SAP AI",
    "In-House Developed ML Solution",
    "In-House Developed AI Solution",
]
CHALLENGES_DATA_UTILIZATION_STANDARD_OPTIONS = [
    "Data quality issues",
    "Lack of skilled personnel or expertise in data analytics",
    "Integration difficulties",
    "High costs",
    "Difficulty in extracting insights from data",
    "Resistance to change, challenging culture change management",
    "Insufficient support from top management",
]
BARRIERS_AIML_GENERAL_STANDARD_OPTIONS = [
    "Unauthorized access to sensitive operational data",
    "Lack of transparency and Explainability",
    "Potential for biased or inaccurate predictions affecting operations",
    "Integration security issues with existing IT systems",
    "Difficulties in monitoring and auditing AI model behavior over time",
]
BARRIERS_GENAI_CONCERNS_STANDARD_OPTIONS = [
    "Intellectual Property (IP) Protection",
    "Regulatory Compliance",
    "Model Bias and Ethical Concerns",
    "Third-Party AI Service Risks",
    "Lack of Transparency and Explainability",
]
AI_TRANSPARENCY_METHODS_STANDARD_OPTIONS = [
    "Clear documentation of AI models and data sources",
    "Explainable AI (XAI) tools and techniques",
    "Transparent data governance policies",
    "Third-party validation and certification",
    "Employee training and awareness programs",
]


# Likert Scale Labels
LIKERT_SCALE_LABELS_AGREEMENT = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly Agree",
}
LIKERT_SCALE_LABELS_IMPORTANCE = {
    1: "Very Unimportant",
    2: "Unimportant",
    3: "Neutral",
    4: "Important",
    5: "Very Important",
}

# Ordered Categories for Plots
EXPERIENCE_ORDER = [
    "Less than 1 year",
    "1-3 years",
    "4-6 years",
    "7-10 years",
    "More than 10 years",
    "Not Specified",
]
COMPANY_SIZE_ORDER = [
    "1-50",
    "51-250",
    "251-1000",
    "1001-5000",
    "More than 5000",
    "Not Specified",
]
REVENUE_ORDER = [
    "<$1 million",
    "$1 million - $10 million",
    "$10 million - $50 million",
    "$50 million - $100 million",
    "$100 million - $500 million",
    ">$500 million",
    "Not Specified",
]


# --- Helper Functions ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


markdown_content = []


def add_to_markdown(text, level=None):
    if level:
        markdown_content.append(f"\n{'#' * level} {text}\n")
    else:
        markdown_content.append(text + "\n")


def save_plot_and_add_to_markdown_dual(
    fig_count, fig_pct, title_slug_base, subfolder, caption_base
):
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, subfolder))

    img_path_count_rel = os.path.join(
        OUTPUT_BASE_DIR, subfolder, f"{title_slug_base}_count.png"
    )
    img_path_pct_rel = os.path.join(
        OUTPUT_BASE_DIR, subfolder, f"{title_slug_base}_percent.png"
    )

    img_path_count_md = os.path.join(subfolder, f"{title_slug_base}_count.png")
    img_path_pct_md = os.path.join(subfolder, f"{title_slug_base}_percent.png")

    fig_count.savefig(img_path_count_rel, bbox_inches="tight", dpi=150)
    plt.close(fig_count)
    fig_pct.savefig(img_path_pct_rel, bbox_inches="tight", dpi=150)
    plt.close(fig_pct)

    add_to_markdown(
        f"| ![{caption_base} (Counts)]({img_path_count_md}) | ![{caption_base} (Percentages)]({img_path_pct_md}) |"
        + "\n|-----------------|-----------------|\n"
        + f"| *{caption_base} (Counts)* | *{caption_base} (Percentages)* |"
    )


def save_plot_and_add_to_markdown_single(fig, title_slug, subfolder, caption):
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, subfolder))
    img_path_relative = os.path.join(OUTPUT_BASE_DIR, subfolder, f"{title_slug}.png")
    img_path_markdown = os.path.join(subfolder, f"{title_slug}.png")

    if isinstance(fig, sns.FacetGrid):
        fig.savefig(img_path_relative, bbox_inches="tight", dpi=150)
    else:
        fig.figure.savefig(img_path_relative, bbox_inches="tight", dpi=150)
    plt.close(fig.figure if hasattr(fig, "figure") else fig)

    add_to_markdown(f"![{caption}]({img_path_markdown})")
    add_to_markdown(f"*{caption}*")


def parse_list_string(s):
    # This function remains useful as CSVs store lists as strings.
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    s_str = str(s).strip()
    if not s_str:
        return []

    if s_str.startswith("[") and s_str.endswith("]"):
        items_str = s_str[1:-1].strip()
        if not items_str:
            return []

        try:
            parsed_list = [
                item.strip().strip("'\"")
                for item in re.findall(
                    r"(?:[^,'\"\[\]]+|'[^']*'|\"[^\"]*\")+", items_str
                )
            ]
            return [item for item in parsed_list if item]
        except Exception:
            return [
                item.strip().strip("'\"")
                for item in items_str.split(",")
                if item.strip().strip("'\"")
            ]

    return [item.strip() for item in s_str.split(";") if item.strip()]


# --- Core Analysis Functions ---


def analyze_text_single(
    df: pd.DataFrame,
    column_name: str,
    title_desc: str,
    subfolder,
    ordered_categories=None,
    na_fill_value="Not Specified",
):
    add_to_markdown(f"Distribution of {title_desc} (`{column_name}`)", level=3)
    if column_name not in df.columns:
        add_to_markdown(f"*Column `{column_name}` not found in the dataset.*")
        return

    series = df[column_name].fillna(
        na_fill_value if isinstance(na_fill_value, str) else str(na_fill_value)
    )
    counts = series.value_counts(dropna=False)
    percentages = series.value_counts(normalize=True, dropna=False).mul(100)

    summary_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})

    effective_order = []
    if ordered_categories:
        temp_order = list(ordered_categories)
        series_list = series.to_list()
        for cat in temp_order + series_list:
            if (
                cat not in effective_order
                and cat in series_list
                and summary_df.loc[cat, "Count"] > 0
            ):
                effective_order.append(cat)

    else:
        effective_order = counts.index.tolist()

    summary_df = summary_df.reindex(effective_order).fillna(0)
    add_to_markdown(
        summary_df.to_markdown(headers=[title_desc, "Count", "Percentage (%)"])
    )

    plot_order = effective_order

    # Plot for Counts
    fig_c, ax_c = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x=summary_df.index,
        y=summary_df["Count"],
        ax=ax_c,
        order=plot_order,
    )
    ax_c.set_title(f"{title_desc} (Counts)")
    ax_c.set_ylabel("Number of Respondents")
    ax_c.set_xlabel(title_desc)
    plt.xticks(rotation=45, ha="right")

    # Plot for Percentages
    fig_p, ax_p = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x=summary_df.index,
        y=summary_df["Percentage (%)"],
        ax=ax_p,
        order=plot_order,
    )
    ax_p.set_title(f"{title_desc} (Percentages)")
    ax_p.set_ylabel("Percentage of Respondents (%)")
    ax_p.set_xlabel(title_desc)
    plt.xticks(rotation=45, ha="right")

    caption = f"Fig: Distribution of {title_desc}"
    save_plot_and_add_to_markdown_dual(fig_c, fig_p, column_name, subfolder, caption)


def analyze_likert_scale(
    df: pd.DataFrame, column_name: str, title_desc: str, subfolder, likert_labels
):
    add_to_markdown(f"Analysis of: {title_desc} (`{column_name}`)", level=3)
    if column_name not in df.columns:
        add_to_markdown(f"*Column `{column_name}` not found in the dataset.*")
        return

    series = pd.to_numeric(df[column_name], errors="raise")

    desc_stats = series.agg(
        [
            "mean",
            "median",
            "std",
            lambda x: x.mode()[0]
            if not x.mode().empty and pd.notna(x.mode()[0])
            else np.nan,
        ]
    ).rename({"<lambda_0>": "mode", "std": "std_dev"})
    add_to_markdown("**Descriptive Statistics:**")
    add_to_markdown(desc_stats.to_frame().T.to_markdown(index=False))

    counts = (
        series.value_counts().reindex(likert_labels.keys(), fill_value=0).sort_index()
    )
    # Calculate percentages based on non-missing responses for this specific question
    non_missing_count = series.dropna().shape[0]
    percentages = (
        (counts / non_missing_count * 100).fillna(0)
        if non_missing_count > 0
        else pd.Series([0.0] * len(counts), index=counts.index)
    )

    summary_df = pd.DataFrame(
        {
            "Response": [likert_labels.get(k, k) for k in counts.index],
            "Count": counts.values,
            "Percentage (%)": percentages.values,
        }
    )
    add_to_markdown(summary_df.to_markdown(index=False))

    # Plot for Counts
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=[likert_labels.get(k, k) for k in counts.index],
        y=counts.values,
        ax=ax_c,
        order=[likert_labels.get(k, k) for k in likert_labels.keys()],
    )
    ax_c.set_title(f"{title_desc} (Counts)")
    ax_c.set_ylabel("Number of Respondents")
    ax_c.set_xlabel("Response")
    plt.xticks(rotation=45, ha="right")

    # Plot for Percentages
    fig_p, ax_p = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=[likert_labels.get(k, k) for k in percentages.index],
        y=percentages.values,
        ax=ax_p,
        order=[likert_labels.get(k, k) for k in likert_labels.keys()],
    )
    ax_p.set_title(f"{title_desc} (Percentages)")
    ax_p.set_ylabel("Percentage of Respondents (%)")
    ax_p.set_xlabel("Response")
    plt.xticks(rotation=45, ha="right")

    caption = f"Fig: Distribution for {title_desc}"
    save_plot_and_add_to_markdown_dual(fig_c, fig_p, column_name, subfolder, caption)


def analyze_text_multi(
    df: pd.DataFrame,
    column_name: str,
    title_desc: str,
    subfolder,
    ordered_categories=None,
    na_fill_value="Not Specified",
):
    add_to_markdown(f"Distribution of {title_desc} (`{column_name}`)", level=3)
    if column_name not in df.columns:
        add_to_markdown(f"*Column `{column_name}` not found in the dataset.*")
        return

    num_respondents = df[column_name].shape[0]
    parsed_list_col = (
        df[column_name]
        .fillna(na_fill_value if isinstance(na_fill_value, str) else str(na_fill_value))
        .apply(parse_list_string)
    )
    exploded_items = parsed_list_col.explode().str.strip()
    exploded_items = exploded_items[exploded_items != ""]
    series = pd.Series(exploded_items.tolist())
    counts = series.value_counts(dropna=False)
    percentages = counts / num_respondents * 100

    summary_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})

    effective_order = []
    if ordered_categories:
        temp_order = list(ordered_categories)
        series_list = series.to_list()
        for cat in temp_order + series_list:
            if (
                cat not in effective_order
                and cat in series_list
                and summary_df.loc[cat, "Count"] > 0
            ):
                effective_order.append(cat)

    else:
        effective_order = counts.index.tolist()

    summary_df = summary_df.reindex(effective_order).fillna(0)
    add_to_markdown(
        summary_df.to_markdown(headers=[title_desc, "Count", "Percentage (%)"])
    )

    plot_order = effective_order

    # Plot for Counts
    fig_c, ax_c = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x=summary_df.index,
        y=summary_df["Count"],
        ax=ax_c,
        order=plot_order,
    )
    ax_c.set_title(f"{title_desc} (Counts)")
    ax_c.set_ylabel("Number of Respondents")
    ax_c.set_xlabel(title_desc)
    plt.xticks(rotation=45, ha="right")

    # Plot for Percentages
    fig_p, ax_p = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x=summary_df.index,
        y=summary_df["Percentage (%)"],
        ax=ax_p,
        order=plot_order,
    )
    ax_p.set_title(f"{title_desc} (Percentages)")
    ax_p.set_ylabel("Percentage of Respondents (%)")
    ax_p.set_xlabel(title_desc)
    plt.xticks(rotation=45, ha="right")

    caption = f"Fig: Distribution of {title_desc}"
    save_plot_and_add_to_markdown_dual(fig_c, fig_p, column_name, subfolder, caption)


def analyze_composite_score(
    df, columns_to_average, composite_name, title_desc, subfolder, fig_prefix
):
    add_to_markdown(f"Composite Score: {title_desc} (`{composite_name}`)", level=3)

    present_cols = [col for col in columns_to_average if col in df.columns]
    if len(present_cols) < 2:  # Cronbach's alpha needs at least 2 items
        add_to_markdown(
            f"*Not enough items ({len(present_cols)} found from '{', '.join(columns_to_average)}') to calculate composite score or Cronbach's Alpha.*"
        )
        return

    for col in present_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    data_for_alpha = df[present_cols].dropna()
    if data_for_alpha.shape[0] < 2 or data_for_alpha.shape[1] < 2:
        add_to_markdown(
            f"*Not enough complete cases ({data_for_alpha.shape[0]}) or items ({data_for_alpha.shape[1]}) "
            f"for Cronbach's Alpha calculation for {title_desc}. Skipping Alpha calculation.*"
        )
    else:
        try:
            alpha_stats = pg.cronbach_alpha(data=data_for_alpha)
            add_to_markdown(
                f"Cronbach's Alpha for {title_desc} ({len(present_cols)} items): {alpha_stats[0]:.3f} (CI: {alpha_stats[1][0]:.3f}-{alpha_stats[1][1]:.3f})"
            )
            if alpha_stats[0] < 0.7:
                add_to_markdown(
                    f"  - *Note: Cronbach's Alpha ({alpha_stats[0]:.3f}) is somewhat low. Interpret composite score with caution.*"
                )
        except Exception as e:
            add_to_markdown(
                f"*Error calculating Cronbach's Alpha for {title_desc}: {e}. Skipping Alpha.*"
            )

    df[composite_name] = df[present_cols].mean(axis=1)

    desc_stats_composite = (
        df[composite_name]
        .agg(
            [
                "mean",
                "median",
                "std",
                lambda x: x.mode()[0]
                if not x.mode().empty and pd.notna(x.mode()[0])
                else np.nan,
            ]
        )
        .rename({"<lambda_0>": "mode", "std": "std_dev"})
    )
    add_to_markdown(f"**Descriptive Statistics for {title_desc} Composite Score:**")
    add_to_markdown(desc_stats_composite.to_frame().T.to_markdown(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_edges = np.arange(1.0, 5.0, 0.25)
    sns.histplot(df[composite_name].dropna(), kde=True, ax=ax, bins=bin_edges)
    ax.set_title(f"Distribution of {title_desc} Composite Score")
    ax.set_xlabel(f"{title_desc} Composite Score (Average)")
    ax.set_ylabel("Frequency")
    save_plot_and_add_to_markdown_single(
        ax,
        fig_prefix + "_distribution",
        subfolder,
        f"Fig: Distribution of {title_desc} Composite Score",
    )


def handle_company_name(df, column_name):
    add_to_markdown(f"Analysis of Company Names (`{column_name}`)", level=3)
    if column_name not in df.columns:
        add_to_markdown(f"*Column `{column_name}` not found.*")
        return

    valid_names = df[column_name][(df[column_name].notna())]
    num_unique_companies = valid_names.nunique()
    num_responses_with_company_name = len(valid_names)

    add_to_markdown(
        f"- Number of responses providing a company name: {num_responses_with_company_name}"
    )
    add_to_markdown(
        f"- Number of unique company names provided: {num_unique_companies}"
    )
    add_to_markdown(
        "- *Note: Individual company names are not listed to maintain anonymity. This data can be used for external validation or aggregation if needed (e.g., by industry if company data is matched externally).*"
    )


def compare_likert_across_departments(df, question_cols_map, title, subfolder):
    add_to_markdown(f"Comparison: {title}", level=3)
    plot_prefix = re.sub(r"\W+", "_", title.lower())

    data_to_compare = []
    for dept, col_name in question_cols_map.items():
        if col_name in df.columns:
            series = pd.to_numeric(df[col_name], errors="coerce").dropna()
            for val in series:
                data_to_compare.append({"Department": dept, "Score": val})
        else:
            add_to_markdown(f"*Column `{col_name}` for {dept} not found. Skipping.*")

    if not data_to_compare:
        add_to_markdown(
            f"*No data found for comparison for '{title}'. Ensure columns exist and have numeric data.*"
        )
        return

    compare_df = pd.DataFrame(data_to_compare)

    means = compare_df.groupby("Department")["Score"].mean().reset_index()
    add_to_markdown("**Mean Scores by Department:**")
    add_to_markdown(means.to_markdown(index=False))

    # Grouped Bar Plot of Means
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Department",
        y="Score",
        data=compare_df,
        ax=ax_bar,
        estimator=np.mean,
        errorbar="sd",
    )
    ax_bar.set_title(f"Mean {title} by Department")
    ax_bar.set_ylabel("Mean Score (1-5 Scale)")
    save_plot_and_add_to_markdown_single(
        ax_bar,
        plot_prefix + "_mean_comparison",
        subfolder,
        f"Fig: Mean {title} by Department",
    )

    # Box Plot of Distributions
    fig_box, ax_box = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Department", y="Score", data=compare_df, ax=ax_box)
    ax_box.set_title(f"Distribution of {title} by Department")
    ax_box.set_ylabel("Score (1-5 Scale)")
    save_plot_and_add_to_markdown_single(
        ax_box,
        plot_prefix + "_dist_comparison",
        subfolder,
        f"Fig: Distribution of {title} by Department",
    )


def create_combined_cross_departmental_likert_plots(
    df, dept_configs, comparable_likert_metrics, subfolder
):
    add_to_markdown("Combined Cross-Departmental Likert Metric Comparisons", level=3)

    all_metrics_data_for_plot = []
    for metric_title, q_key_suffix in comparable_likert_metrics.items():
        for dept_name, config in dept_configs.items():
            col_name = config["q_map"].get(q_key_suffix)
            if col_name and col_name in df.columns:
                series = pd.to_numeric(df[col_name], errors="coerce").dropna()
                for val in series:
                    all_metrics_data_for_plot.append(
                        {"Department": dept_name, "Score": val, "Metric": metric_title}
                    )

    if not all_metrics_data_for_plot:
        add_to_markdown("*No data found for combined Likert metric comparisons.*")
        return

    all_metrics_df = pd.DataFrame(all_metrics_data_for_plot)

    metric_order = list(comparable_likert_metrics.keys())  # Preserve order for y-axis

    # Plot 1: Mean Scores for Metrics, Grouped by Department
    add_to_markdown("View 1: Mean Scores for Metrics, Grouped by Department", level=4)
    mean_scores_by_dept_df = (
        all_metrics_df.groupby(["Metric", "Department"])["Score"].mean().reset_index()
    )

    mean_scores_by_dept_df["Metric"] = pd.Categorical(
        mean_scores_by_dept_df["Metric"], categories=metric_order, ordered=True
    )
    mean_scores_by_dept_df = mean_scores_by_dept_df.sort_values("Metric")

    fig_agg_dept, ax_agg_dept = plt.subplots(
        figsize=(12, max(8, len(metric_order) * 0.8))
    )
    sns.barplot(
        data=mean_scores_by_dept_df,
        y="Metric",
        x="Score",
        hue="Department",
        ax=ax_agg_dept,
    )

    ax_agg_dept.set_title("Mean Scores for Metrics, Grouped by Department")
    ax_agg_dept.set_xlabel("Mean Score (1-5 Scale)")
    ax_agg_dept.set_ylabel("Metric")
    ax_agg_dept.legend(title="Department", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    save_plot_and_add_to_markdown_single(
        ax_agg_dept,
        "combined_likert_means_by_dept",
        subfolder,
        "Fig: Mean Scores for Key Metrics, Grouped by Department",
    )

    # Plot 2: Overall Mean Score for Each Metric (Averaged Across Departments)
    add_to_markdown(
        "View 2: Overall Mean Score for Each Metric (Averaged Across All Departments)",
        level=4,
    )
    overall_mean_scores_df = (
        all_metrics_df.groupby("Metric")["Score"]
        .mean()
        .reindex(metric_order)
        .reset_index()
    )

    add_to_markdown(
        overall_mean_scores_df.to_markdown(
            headers=["Metric", "Overall Mean Score"], index=False
        )
    )

    fig_overall_mean, ax_overall_mean = plt.subplots(
        figsize=(10, max(6, len(metric_order) * 0.5))
    )
    sns.barplot(
        data=overall_mean_scores_df,
        y="Metric",
        x="Score",
        ax=ax_overall_mean,
        order=metric_order,
    )
    ax_overall_mean.set_title(
        "Overall Mean Score for Comparable Likert Metrics (All Departments)"
    )
    ax_overall_mean.set_xlabel("Overall Mean Score (1-5 Scale)")
    ax_overall_mean.set_ylabel("Metric")
    plt.tight_layout()

    save_plot_and_add_to_markdown_single(
        ax_overall_mean,
        "combined_likert_overall_means",
        subfolder,
        "Fig: Overall Mean Score for Comparable Likert Metrics (All Departments)",
    )


def compare_multiselect_across_departments(
    df, dept_configs_for_comparison, map_key, title, subfolder, standard_options
):
    add_to_markdown(f"Comparison: {title}", level=3)
    plot_prefix = re.sub(r"\W+", "_", title.lower())
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, subfolder))

    all_dept_data = []

    for dept_name, config in dept_configs_for_comparison.items():
        list_col = config["q_map"].get(map_key)

        if not list_col:
            add_to_markdown(
                f"*Skipping {dept_name} for '{title}' as list/other columns not fully defined.*"
            )
            continue

        current_dept_items = []
        respondent_indices_dept = set()

        if list_col in df.columns:
            parsed_list_col = df[list_col].apply(parse_list_string)
            exploded = parsed_list_col.explode().dropna().str.strip()
            current_dept_items.extend(exploded[exploded != ""].tolist())
            respondent_indices_dept.update(
                df[df[list_col].apply(lambda x: bool(parse_list_string(x)))].index
            )

        item_counts_dept = pd.Series(Counter(current_dept_items))
        num_respondents_dept = max(1, len(respondent_indices_dept))

        for item, count_val in item_counts_dept.items():
            all_dept_data.append(
                {
                    "Department": dept_name,
                    "Item": item,
                    "Count": count_val,
                    "Percentage": (count_val / num_respondents_dept) * 100,
                }
            )

    if not all_dept_data:
        add_to_markdown(f"*No data found for {title} comparison.*")
        return

    compare_df = pd.DataFrame(all_dept_data)

    # Sort by overall count of items to make plots more readable
    item_total_counts = (
        compare_df.groupby("Item")["Count"].sum().sort_values(ascending=False)
    )
    compare_df["Item"] = pd.Categorical(
        compare_df["Item"], categories=item_total_counts.index, ordered=True
    )
    compare_df = compare_df.sort_values("Item")

    pivot_df_count = compare_df.pivot_table(
        index="Item", columns="Department", values="Count", fill_value=0, observed=False
    )
    pivot_df_count = pivot_df_count.reindex(item_total_counts.index)  # Keep sort order
    add_to_markdown("**Count of Selections by Department:**")
    add_to_markdown(pivot_df_count.to_markdown())

    pivot_df_pct = compare_df.pivot_table(
        index="Item",
        columns="Department",
        values="Percentage",
        fill_value=0,
        observed=False,
    )
    pivot_df_pct = pivot_df_pct.reindex(item_total_counts.index)  # Keep sort order
    add_to_markdown("**Percentage of Respondents Selecting Item by Department:**")
    add_to_markdown(pivot_df_pct.to_markdown())

    unique_items = compare_df["Item"].unique()  # Already category type, order preserved
    num_items = len(unique_items)

    if num_items > 0:
        # Plot for Counts
        fig_c, ax_c = plt.subplots(figsize=(14, max(8, num_items * 0.5)))
        sns.barplot(
            x="Count",
            y="Item",
            hue="Department",
            data=compare_df,
            ax=ax_c,
            order=unique_items,
        )
        ax_c.set_title(f"{title} by Department (Counts)")
        ax_c.set_xlabel("Number of Selections")
        ax_c.set_ylabel("Item")
        ax_c.legend(title="Department", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for legend

        # Plot for Percentages
        fig_p, ax_p = plt.subplots(figsize=(14, max(8, num_items * 0.5)))
        sns.barplot(
            x="Percentage",
            y="Item",
            hue="Department",
            data=compare_df,
            ax=ax_p,
            order=unique_items,
        )
        ax_p.set_title(f"{title} by Department (Percentages)")
        ax_p.set_xlabel("Percentage of Respondents Selecting Item (%)")
        ax_p.set_ylabel("Item")
        ax_p.legend(title="Department", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for legend

        caption_base = f"Fig: {title} by Department"
        save_plot_and_add_to_markdown_dual(
            fig_c,
            fig_p,
            plot_prefix + "_item_comparison_by_dept",
            subfolder,
            caption_base,
        )

    else:
        add_to_markdown("*No items to plot for comparison.*")


# --- Main Analysis Script ---
def descriptive_analysis(df: pd.DataFrame):
    global markdown_content
    markdown_content = []

    # --- Initialize Report Collection ---
    script_start_time = datetime.datetime.now()
    add_to_markdown("Phase 2: Descriptive Analysis of Survey Data", level=1)
    add_to_markdown(
        f"Execution Start Time: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_to_markdown("---")

    # --- Section 1: Respondent Demographics & Company Profile ---
    add_to_markdown("Respondent Demographics & Company Profile", level=2)
    demo_subfolder = "1_demographics"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, demo_subfolder))

    # q2_position (Text)
    analyze_text_single(
        df, "q2_position", "Respondent Position", demo_subfolder, POSITION_CATEGORIES
    )
    # q3_department (MC Single)
    analyze_text_single(df, "q3_department", "Department", demo_subfolder)
    # q4_experience_role (MC Single)
    analyze_text_single(
        df,
        "q4_experience_role",
        "Years of Experience in Current Role",
        demo_subfolder,
        ordered_categories=EXPERIENCE_ORDER,
    )
    # q5_experience_company (MC Single)
    analyze_text_single(
        df,
        "q5_experience_company",
        "Years of Experience with Current Company",
        demo_subfolder,
        ordered_categories=EXPERIENCE_ORDER,
    )
    # q6_company_name (Text - Special Handling)
    handle_company_name(df, "q6_company_name")
    # q7_industry_sector (Text)
    analyze_text_single(
        df,
        "q7_industry_sector",
        "Industry Sector",
        demo_subfolder,
        INDUSTRY_SECTOR_CATEGORIES,
    )
    # q8_company_size (MC Single)
    analyze_text_single(
        df,
        "q8_company_size",
        "Company Size (Number of Employees)",
        demo_subfolder,
        ordered_categories=COMPANY_SIZE_ORDER,
    )
    # q9_company_hq_country (Text)
    analyze_text_single(
        df,
        "q9_company_hq_country",
        "Company HQ Country",
        demo_subfolder,
        COUNTRY_CATEGORIES,
    )
    # q10_company_revenue (MC Single)
    analyze_text_single(
        df,
        "q10_company_revenue",
        "Company Approximate Annual Revenue",
        demo_subfolder,
        ordered_categories=REVENUE_ORDER,
    )

    # --- Section 2: Operational Excellence (OpEx) Perceptions ---
    add_to_markdown("Operational Excellence (OpEx) Perceptions", level=2)
    opex_subfolder = "2_opex_perceptions"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, opex_subfolder))

    opex_likert_questions = {
        "q11_opex_ce_respect_internal": "CE: Respect for Internal Individuals",
        "q12_opex_ce_respect_external": "CE: Respect for External Individuals",
        "q13_opex_ce_employee_health": "CE: Concern for Employee Health",
        "q14_opex_ce_employee_safety": "CE: Concern for Employee Safety",
        "q15_opex_ce_social_community_rights": "CE: Care for Social Community Rights",
        "q16_opex_ce_training_investment": "CE: Training as Investment",
        "q17_opex_ce_handson_investment": "CE: Hands-on Exposure as Investment",
        "q18_opex_cpi_customer_demand_immediate": "CPI: Immediate Response to Customer Demand",
        "q19_opex_cpi_products_flexible": "CPI: Flexible Products/Services",
        "q20_opex_cpi_monitor_deterioration": "CPI: Monitoring Product/Service Deterioration",
        "q21_opex_cpi_value_addition": "CPI: Ensuring Value Addition at Each Step",
        "q22_opex_cpi_direct_observations": "CPI: Using Direct Observations for Process Evaluation",
        "q23_opex_cpi_employee_empowerment_defects": "CPI: Employee Empowerment for Defect Elimination",
        "q24_opex_cpi_manager_empowerment_stop_process": "CPI: Manager Empowerment to Stop Defective Process",
        "q25_opex_cpi_counter_measures_defects": "CPI: Counter Measures for Defect Reoccurrence",
        "q26_opex_cpi_improvement_daily_work": "CPI: Improvement as Integral Part of Daily Work",
        "q27_opex_cpi_tools_reliable_data": "CPI: Effective Tools for Reliable Data Gathering",
        "q28_opex_ea_planning_respect_inputs": "EA: Planning Process Respects Inputs",
        "q29_opex_ea_planning_involves_inputs": "EA: Planning Process Involves Inputs from Different Levels",
        "q30_opex_ea_work_description_defined": "EA: Defined Work Description for All Levels",
        "q31_opex_ea_work_description_standardized": "EA: Standardized Work Description for All Levels",
        "q32_opex_ea_defines_metrics_data": "EA: Defines Relevant Metrics/Data for Users",
        "q33_opex_ea_adjust_personal_values": "EA: Encourages Adjusting Personal Values for Organizational Objectives",
    }
    for col, title in opex_likert_questions.items():
        analyze_likert_scale(
            df, col, title, opex_subfolder, LIKERT_SCALE_LABELS_AGREEMENT
        )

    # OpEx Composite Scores
    ce_cols = [col for col in opex_likert_questions.keys() if "_opex_ce_" in col]
    cpi_cols = [col for col in opex_likert_questions.keys() if "_opex_cpi_" in col]
    ea_cols = [col for col in opex_likert_questions.keys() if "_opex_ea_" in col]

    analyze_composite_score(
        df,
        ce_cols,
        "opex_ce_composite",
        "Cultural Enablers (CE)",
        opex_subfolder,
        "opex_ce_composite",
    )
    analyze_composite_score(
        df,
        cpi_cols,
        "opex_cpi_composite",
        "Continuous Process Improvement (CPI)",
        opex_subfolder,
        "opex_cpi_composite",
    )
    analyze_composite_score(
        df,
        ea_cols,
        "opex_ea_composite",
        "Enterprise Alignment (EA)",
        opex_subfolder,
        "opex_ea_composite",
    )

    # --- Section 3: Data & AI/ML Practices (Maintenance, Quality, Production) ---
    add_to_markdown("Data & AI/ML Practices by Department", level=2)

    dept_configs = {
        "Maintenance": {
            "prefix": "maint",
            "q_map": {
                "dg_collection_freq": "q34_maint_dg_collection_freq",
                "da_analysis_freq": "q35_maint_da_analysis_freq",
                "da_tools_text": "q36_maint_da_tools_text",
                "ai_usage": "q37_maint_ai_usage",
                "ai_tools_mc": "q38_maint_ai_tools_mc",
                "dm_insights_drive_decisions": "q39_maint_dm_insights_drive_decisions",
                "tl_staff_trained": "q40_maint_tl_staff_trained",
                "tl_data_driven_culture": "q41_maint_tl_data_driven_culture",
                "out_measurable_improvements": "q42_maint_out_measurable_improvements",
                "out_insights_ci": "q43_maint_out_insights_ci",
                "out_data_minimizes_cost": "q44_maint_out_data_minimizes_cost",
                "out_challenges_utilizing_data": "q45_maint_out_challenges_utilizing_data",
                "out_challenges_list_mc": "q46_maint_out_challenges_list_mc",
            },
            "da_tool_llm_categories": DA_TOOL_CATEGORIES,
            "folder": "3_data_practices_maintenance",
        },
        "Quality": {
            "prefix": "qual",
            "q_map": {
                "dg_collection_freq": "q47_qual_dg_collection_freq",
                "da_analysis_freq": "q48_qual_da_analysis_freq",
                "da_tools_text": "q49_qual_da_tools_text",
                "ai_usage": "q50_qual_ai_usage",
                "ai_tools_mc": "q51_qual_ai_tools_mc",
                "dm_insights_drive_decisions": "q52_qual_dm_insights_drive_decisions",
                "tl_staff_trained": "q53_qual_tl_staff_trained",
                "tl_data_driven_culture": "q54_qual_tl_data_driven_culture",
                "out_measurable_improvements": "q55_qual_out_measurable_improvements",
                "out_insights_ci": "q56_qual_out_insights_ci",
                "out_data_minimizes_cost": "q57_qual_out_data_minimizes_cost",
                "out_challenges_utilizing_data": "q58_qual_out_challenges_utilizing_data",
                "out_challenges_list_mc": "q59_qual_out_challenges_list_mc",
            },
            "da_tool_llm_categories": DA_TOOL_CATEGORIES,
            "folder": "4_data_practices_quality",
        },
        "Production": {
            "prefix": "prod",
            "q_map": {
                "dg_collection_freq": "q60_prod_dg_collection_freq",
                "da_analysis_freq": "q61_prod_da_analysis_freq",
                "da_tools_text": "q62_prod_da_tools_text",
                "ai_usage": "q63_prod_ai_usage",
                "ai_tools_mc": "q64_prod_ai_tools_mc",
                "dm_insights_drive_decisions": "q65_prod_dm_insights_drive_decisions",
                "tl_staff_trained": "q66_prod_tl_staff_trained",
                "tl_data_driven_culture": "q67_prod_tl_data_driven_culture",
                "out_measurable_improvements": "q68_prod_out_measurable_improvements",
                "out_insights_ci": "q69_prod_out_insights_ci",
                "out_data_minimizes_cost": "q70_prod_out_data_minimizes_cost",
                "out_challenges_utilizing_data": "q71_prod_out_challenges_utilizing_data",
                "out_challenges_list_mc": "q72_prod_out_challenges_list_mc",
            },
            "da_tool_llm_categories": DA_TOOL_CATEGORIES,
            "folder": "5_data_practices_production",
        },
    }

    for dept_name, config in dept_configs.items():
        add_to_markdown(f"{dept_name} Operations", level=3)
        subfolder = config["folder"]
        ensure_dir(os.path.join(OUTPUT_BASE_DIR, subfolder))
        qm = config["q_map"]

        # Likert Scale Questions
        likert_q_details = {
            qm.get("dg_collection_freq"): f"{dept_name}: Regular Data Collection",
            qm.get("da_analysis_freq"): f"{dept_name}: Regular Data Analysis",
            qm.get("ai_usage"): f"{dept_name}: AI & GenAI Usage",
            qm.get(
                "dm_insights_drive_decisions"
            ): f"{dept_name}: Data Insights Drive Decisions",
            qm.get("tl_staff_trained"): f"{dept_name}: Staff Trained for Data Use",
            qm.get(
                "tl_data_driven_culture"
            ): f"{dept_name}: Data-Driven Culture in Teams",
            qm.get(
                "out_measurable_improvements"
            ): f"{dept_name}: Measurable Improvements from Data Use",
            qm.get(
                "out_insights_ci"
            ): f"{dept_name}: Data Insights for Continuous Improvement",
            qm.get("out_data_minimizes_cost"): f"{dept_name}: Data Minimizes Cost",
            qm.get(
                "out_challenges_utilizing_data"
            ): f"{dept_name}: Faces Challenges Utilizing Data",
        }
        for col, title in likert_q_details.items():
            if col:
                analyze_likert_scale(
                    df, col, title, subfolder, LIKERT_SCALE_LABELS_AGREEMENT
                )

        # Text Question (Tools Used - LLM Categorized)
        if qm.get("da_tools_text"):
            analyze_text_multi(
                df,
                qm["da_tools_text"],
                f"{dept_name} Data Analysis Tools",
                subfolder,
                config["da_tool_llm_categories"],
            )

        # Multi-Select (AI Tools Used)
        if qm.get("ai_tools_mc"):
            analyze_text_multi(
                df,
                qm["ai_tools_mc"],
                f"{dept_name} AI/GenAI Tools Used (Other)",
                subfolder,
                AI_GENAI_TOOL_STANDARD_OPTIONS,
            )

        # Multi-Select (Challenges List)
        if qm.get("out_challenges_list_mc"):
            analyze_text_multi(
                df,
                qm["out_challenges_list_mc"],
                f"{dept_name} Challenges in Utilizing Data (Other)",
                subfolder,
                CHALLENGES_DATA_UTILIZATION_STANDARD_OPTIONS,
            )

    # --- Section 4: Barriers to AI/ML and Transparency ---
    add_to_markdown("Barriers to AI/ML and Transparency", level=2)
    barriers_subfolder = "6_barriers_and_transparency"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, barriers_subfolder))

    barrier_likert_questions = {
        "q73_barriers_ps_ai_limit": "Privacy/Security Concerns Limit AI/ML Use",
        "q74_barriers_ps_genai_limit": "Privacy/Security Concerns Limit GenAI Use",
        "q77_barriers_ai_transparency_efforts": "Efforts for AI Transparency for Business Purposes",
        "q78_barriers_ai_transparency_importance_maint": "Importance of AI Transparency for Maintenance",
        "q79_barriers_ai_transparency_importance_qual": "Importance of AI Transparency for Quality",
        "q80_barriers_ai_transparency_importance_prod": "Importance of AI Transparency for Production",
    }
    for col, title in barrier_likert_questions.items():
        scale_labels = (
            LIKERT_SCALE_LABELS_IMPORTANCE
            if "importance" in col
            else LIKERT_SCALE_LABELS_AGREEMENT
        )
        analyze_likert_scale(df, col, title, barriers_subfolder, scale_labels)

    # Multi-Select Questions for Barriers & Transparency
    analyze_text_multi(
        df,
        "q75_barriers_aiml_general_mc",
        "General Barriers to AI/ML Usage",
        barriers_subfolder,
        BARRIERS_AIML_GENERAL_STANDARD_OPTIONS,
    )

    analyze_text_multi(
        df,
        "q76_barriers_ps_genai_concerns_mc",
        "Specific Privacy/Security Concerns for GenAI",
        barriers_subfolder,
        BARRIERS_GENAI_CONCERNS_STANDARD_OPTIONS,
    )

    analyze_text_multi(
        df,
        "q81_barriers_ai_transparency_methods_mc",
        "Methods to Improve AI Transparency",
        barriers_subfolder,
        AI_TRANSPARENCY_METHODS_STANDARD_OPTIONS,
    )

    # --- Section 5: Cross-Departmental Comparisons ---
    add_to_markdown("Cross-Departmental Comparisons", level=2)
    comparison_subfolder = "7_cross_departmental_comparisons"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, comparison_subfolder))

    comparable_likert_metrics = {
        "Regular Data Collection Frequency": "dg_collection_freq",
        "Regular Data Analysis Frequency": "da_analysis_freq",
        "AI & GenAI Usage": "ai_usage",
        "Data Insights Drive Decisions": "dm_insights_drive_decisions",
        "Staff Trained for Data Use": "tl_staff_trained",
        "Data-Driven Culture in Teams": "tl_data_driven_culture",
        "Measurable Improvements from Data Use": "out_measurable_improvements",
        "Faces Challenges Utilizing Data": "out_challenges_utilizing_data",
    }

    for metric_title, q_key_suffix in comparable_likert_metrics.items():
        cols_map = {
            dept: config["q_map"].get(q_key_suffix)
            for dept, config in dept_configs.items()
        }
        valid_cols_map = {k: v for k, v in cols_map.items() if v and v in df.columns}
        if len(valid_cols_map) > 1:
            compare_likert_across_departments(
                df, valid_cols_map, metric_title, comparison_subfolder
            )
        else:
            add_to_markdown(
                f"*Skipping comparison for '{metric_title}' due to missing columns for sufficient departments.*"
            )

    # Combined cross-departmental Likert plots
    create_combined_cross_departmental_likert_plots(
        df, dept_configs, comparable_likert_metrics, comparison_subfolder
    )

    # Multi-select comparisons across departments
    compare_multiselect_across_departments(
        df,
        dept_configs,
        "ai_tools_mc",
        "AI/GenAI Tools Used",
        comparison_subfolder,
        AI_GENAI_TOOL_STANDARD_OPTIONS,
    )
    compare_multiselect_across_departments(
        df,
        dept_configs,
        "out_challenges_list_mc",
        "Challenges in Utilizing Data",
        comparison_subfolder,
        CHALLENGES_DATA_UTILIZATION_STANDARD_OPTIONS,
    )

    # --- Finalize Markdown ---
    add_to_markdown("\nEnd of Phase 2 Descriptive Analysis Report.")

    script_end_time = datetime.datetime.now()
    add_to_markdown("---")
    add_to_markdown(
        f"Execution End Time: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_to_markdown(f"Total Execution Time: {script_end_time - script_start_time}")

    with open(MARKDOWN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))
    print(f"Markdown report generated: {MARKDOWN_FILE}")
    print(f"Visualizations and report saved in: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    ensure_dir(OUTPUT_BASE_DIR)

    try:
        df_survey = pd.read_csv(FILE_PATH)
        print(f"Successfully loaded data from {FILE_PATH}. Shape: {df_survey.shape}")
    except FileNotFoundError:
        print(
            f"Error: The file {FILE_PATH} was not found. Please check the path from Script 1's output."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data from {FILE_PATH}: {e}")
        sys.exit(1)

    if df_survey.empty:
        print(f"Loaded DataFrame from {FILE_PATH} is empty. Please check the CSV file.")
        sys.exit(1)

    descriptive_analysis(df_survey.copy())
