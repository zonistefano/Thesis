import datetime
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Matplotlib Style ---
plt.style.use("seaborn-v0_8-whitegrid")

# --- Configuration ---
OUTPUT_BASE_DIR = "3_inferential_analysis"
MARKDOWN_FILE = os.path.join(
    OUTPUT_BASE_DIR, "3_inferential_analysis_report.md"
)

# Department configurations
DEPT_CONFIGS = {
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
            "out_challenges_utilizing_data": "q45_maint_out_challenges_utilizing_data",
            "out_challenges_list_mc": "q46_maint_out_challenges_list_mc",
        },
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
            "out_challenges_utilizing_data": "q58_qual_out_challenges_utilizing_data",
            "out_challenges_list_mc": "q59_qual_out_challenges_list_mc",
        },
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
            "out_challenges_utilizing_data": "q71_prod_out_challenges_utilizing_data",
            "out_challenges_list_mc": "q72_prod_out_challenges_list_mc",
        },
    },
}
# OpEx Composite Score Columns (expected from descriptive_analysis script or to be calculated)
OPEX_COMPOSITE_COLS = [
    "opex_ce_composite",
    "opex_cpi_composite",
    "opex_ea_composite",
    "opex_overall_composite",
]

# --- Helper Functions ---
markdown_content = []


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def add_to_markdown(text, level=None):
    if level:
        markdown_content.append(f"\n{'#' * level} {text}\n")
    else:
        markdown_content.append(text + "\n")


def save_plot_and_add_to_markdown_single(fig, title_slug, subfolder, caption):
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, subfolder))
    img_path_relative = os.path.join(OUTPUT_BASE_DIR, subfolder, f"{title_slug}.png")
    img_path_markdown = os.path.join(subfolder, f"{title_slug}.png")

    figure_to_save = None
    figure_to_close = None

    if isinstance(fig, sns.FacetGrid):
        figure_to_save = fig
        figure_to_close = fig.fig
    elif isinstance(fig, plt.Figure):
        figure_to_save = fig
        figure_to_close = fig
    elif hasattr(fig, "figure") and isinstance(fig.figure, plt.Figure):
        figure_to_save = fig.figure
        figure_to_close = fig.figure
    else:
        error_message = f"Plot '{title_slug}' (type: {type(fig)}) cannot be saved. It is not a `matplotlib.figure.Figure`, `seaborn.FacetGrid`, or an `Axes` object with a valid `.figure` attribute."
        add_to_markdown(f"- *Error saving plot: {error_message}*")
        if isinstance(fig, plt.Axes):
            plt.close(fig.get_figure())
        return

    try:
        figure_to_save.savefig(img_path_relative, bbox_inches="tight", dpi=150)
    except Exception as e:
        add_to_markdown(f"- *Error during savefig for '{title_slug}': {str(e)}*")
        if figure_to_close:
            plt.close(figure_to_close)
        return

    if figure_to_close:
        plt.close(figure_to_close)

    add_to_markdown(f"![{caption}]({img_path_markdown})")
    add_to_markdown(f"*{caption}*")


def parse_list_string(s):
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


def normality_check(data_dict, alpha=0.05):
    results = {}
    add_to_markdown("##### Normality Check (Shapiro-Wilk Test):")
    normality_table_data = []
    for name, data_series in data_dict.items():
        data_series_cleaned = data_series.dropna()
        if len(data_series_cleaned) < 3:
            stat, p_value, is_normal = np.nan, np.nan, "Cannot assess (N < 3)"
            add_to_markdown(
                f"- {name}: N={len(data_series_cleaned)}. Cannot assess normality (sample size < 3)."
            )
        else:
            stat, p_value = stats.shapiro(data_series_cleaned)
            is_normal = "Yes" if p_value > alpha else "No"
            add_to_markdown(
                f"- {name}: W = {stat:.3f}, p = {p_value:.3f}. Normal? **{is_normal}** (alpha={alpha})"
            )
        results[name] = {"W": stat, "p_value": p_value, "is_normal": is_normal == "Yes"}
        normality_table_data.append(
            [
                name,
                len(data_series_cleaned),
                f"{stat:.3f}" if pd.notna(stat) else "N/A",
                f"{p_value:.3f}" if pd.notna(p_value) else "N/A",
                is_normal,
            ]
        )

    if normality_table_data:
        normality_df = pd.DataFrame(
            normality_table_data,
            columns=[
                "Group",
                "N (non-NaN)",
                "Shapiro-Wilk W",
                "p-value",
                f"Normal (alpha={alpha})?",
            ],
        )
        add_to_markdown(normality_df.to_markdown(index=False))
    return results


def homogeneity_variance_check(samples_list, group_names, alpha=0.05):
    add_to_markdown("##### Homogeneity of Variances Check (Levene's Test):")
    cleaned_samples = [s.dropna() for s in samples_list]
    # Levene's test requires at least 2 samples per group
    valid_samples = [s for s in cleaned_samples if len(s) >= 2]

    if len(valid_samples) < 2:  # Need at least two groups with sufficient samples
        add_to_markdown(
            f"- Cannot perform Levene's test: Not enough groups with sufficient data (found {len(valid_samples)} valid groups)."
        )
        return None, None, False

    stat, p_value = stats.levene(*valid_samples)
    homogeneous = p_value > alpha
    add_to_markdown(
        f"- Levene's Test: W = {stat:.3f}, p = {p_value:.3f}. Variances homogeneous? **{'Yes' if homogeneous else 'No'}** (alpha={alpha})"
    )
    return stat, p_value, homogeneous


# --- Core Analysis Functions for Steps 5, 6, 7 ---


def compare_likert_anova_kruskal(
    df, group_cols_map, metric_name, subfolder, alpha=0.05
):
    add_to_markdown(
        f"Statistical Comparison: {metric_name} Across Departments", level=4
    )

    groups_data = {}
    group_names_ordered = []
    for dept_name, col_name in group_cols_map.items():
        group_names_ordered.append(dept_name)
        if col_name and col_name in df.columns:
            groups_data[dept_name] = pd.to_numeric(
                df[col_name], errors="coerce"
            ).dropna()
        else:
            add_to_markdown(
                f"- *Column `{col_name}` for department `{dept_name}` not found or invalid.*"
            )

    valid_group_names_for_plot_order = [
        name
        for name in group_names_ordered
        if name in groups_data and not groups_data[name].empty
    ]

    if len(groups_data) < 2:
        add_to_markdown(
            "- *Not enough departmental data (at least 2 departments required) to perform comparison.*"
        )
        return

    samples = [
        groups_data[name]
        for name in valid_group_names_for_plot_order
        if len(groups_data[name]) > 0
    ]
    current_group_names = [
        name for name in valid_group_names_for_plot_order if len(groups_data[name]) > 0
    ]

    if len(samples) < 2:
        add_to_markdown(
            "- *Not enough valid samples (at least 2 groups with data required) to perform comparison.*"
        )
        return

    # Visualization: Box plot of distributions
    plot_data_list = []
    for name, data_series in zip(current_group_names, samples):
        for value in data_series:
            plot_data_list.append({"Department": name, metric_name: value})

    if plot_data_list:
        plot_df = pd.DataFrame(plot_data_list)
        fig_box, ax_box = plt.subplots(
            figsize=(max(8, len(current_group_names) * 2), 6)
        )
        sns.boxplot(
            x="Department",
            y=metric_name,
            data=plot_df,
            ax=ax_box,
            order=valid_group_names_for_plot_order,
        )
        sns.stripplot(
            x="Department",
            y=metric_name,
            data=plot_df,
            ax=ax_box,
            order=valid_group_names_for_plot_order,
            color=".25",
            size=4,
            alpha=0.6,
        )

        ax_box.set_title(f"Distribution of {metric_name} Across Departments")
        ax_box.set_xlabel("Department")
        ax_box.set_ylabel(metric_name)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        save_plot_and_add_to_markdown_single(
            fig_box,  # Pass the Figure object
            f"{metric_name.replace(' ', '_').lower()}_comparison_boxplot",
            subfolder,
            f"Fig: Distribution of {metric_name} Across Departments",
        )
    else:
        add_to_markdown(
            f"- *No data available to plot distributions for {metric_name}.*"
        )

    normality_results = normality_check(
        {name: data for name, data in zip(current_group_names, samples)}
    )
    all_normal = all(
        res["is_normal"]
        for res in normality_results.values()
        if isinstance(res["is_normal"], bool)
    )

    homogeneity_stat, homogeneity_p, variances_homogeneous = homogeneity_variance_check(
        samples, current_group_names
    )

    use_anova = all_normal and variances_homogeneous and len(samples) >= 2

    if use_anova:
        add_to_markdown(f"Proceeding with ANOVA for {metric_name}.")
        f_stat, p_value = stats.f_oneway(*samples)
        add_to_markdown(
            f"**ANOVA Result:** F-statistic = {f_stat:.3f}, p-value = {p_value:.4f}"
        )
        if p_value < alpha:
            add_to_markdown(
                f"- The difference in mean {metric_name} across departments is statistically significant (p < {alpha})."
            )
            # Perform Tukey's HSD post-hoc test
            melt_data = []
            for i, name in enumerate(current_group_names):
                for val in samples[i]:
                    melt_data.append({"group": name, "value": val})
            melt_df = pd.DataFrame(melt_data)

            try:
                tukey_results = pairwise_tukeyhsd(
                    melt_df["value"], melt_df["group"], alpha=alpha
                )
                add_to_markdown("##### Tukey's HSD Post-hoc Test Results:")
                add_to_markdown(f"```\n{tukey_results}\n```")

                # Plot Tukey's HSD
                fig_tukey, ax_tukey = plt.subplots(figsize=(10, 6))
                tukey_results.plot_simultaneous(ax=ax_tukey)
                ax_tukey.set_title(
                    f"Tukey's HSD: Pairwise Comparison for {metric_name}"
                )
                save_plot_and_add_to_markdown_single(
                    fig_tukey,
                    f"{metric_name.replace(' ', '_').lower()}_tukey_hsd",
                    subfolder,
                    f"Fig: Tukey's HSD for {metric_name}",
                )

            except Exception as e:
                add_to_markdown(
                    f"- *Error performing Tukey's HSD: {e}. Consider Games-Howell if variances are unequal or sample sizes differ greatly.*"
                )
                if (
                    not variances_homogeneous
                ):  # Suggest Games-Howell if variances are unequal
                    try:
                        games_howell_results = pg.pairwise_gameshowell(
                            data=melt_df, dv="value", between="group"
                        )
                        add_to_markdown(
                            "##### Games-Howell Post-hoc Test Results (due to unequal variances):"
                        )
                        add_to_markdown(games_howell_results.to_markdown(index=False))
                    except Exception as e_gh:
                        add_to_markdown(f"- *Error performing Games-Howell: {e_gh}.*")

        else:
            add_to_markdown(
                f"- There is no statistically significant difference in mean {metric_name} across departments (p >= {alpha})."
            )
    else:
        add_to_markdown(
            f"Assumptions for ANOVA not met (Normality: {all_normal}, Homogeneity: {variances_homogeneous}). Proceeding with Kruskal-Wallis Test for {metric_name}."
        )
        h_stat, p_value = stats.kruskal(*samples)
        add_to_markdown(
            f"**Kruskal-Wallis Result:** H-statistic = {h_stat:.3f}, p-value = {p_value:.4f}"
        )
        if p_value < alpha:
            add_to_markdown(
                f"- The difference in {metric_name} distributions across departments is statistically significant (p < {alpha})."
            )
            # Perform Dunn's post-hoc test
            melt_data = []
            for i, name in enumerate(current_group_names):
                for val in samples[i]:
                    melt_data.append({"group": name, "value": val})
            melt_df = pd.DataFrame(melt_data)

            try:
                dunn_results = posthoc_dunn(
                    melt_df, val_col="value", group_col="group", p_adjust="bonferroni"
                )
                add_to_markdown(
                    "##### Dunn's Post-hoc Test Results (Bonferroni corrected):"
                )
                add_to_markdown(dunn_results.to_markdown())
            except Exception as e:
                add_to_markdown(f"- *Error performing Dunn's test: {e}.*")

        else:
            add_to_markdown(
                f"- There is no statistically significant difference in {metric_name} distributions across departments (p >= {alpha})."
            )
    add_to_markdown("---")


def compare_categorical_chi_square(
    df,
    dept_configs,
    item_column_key_suffix,
    item_description,
    subfolder,
    alpha=0.05,
    min_expected_freq=5,
):
    add_to_markdown(f"Chi-Square Test: {item_description} Across Departments", level=4)

    contingency_data = {}
    all_items = set()

    # Gather data for contingency table
    # This handles both single categorized text and exploded multi lists
    for dept_name, config in dept_configs.items():
        col_name = config["q_map"].get(item_column_key_suffix)  # For single category

        if (
            item_column_key_suffix.endswith("_mc")
            or "tools_text" in item_column_key_suffix
        ):  # Handle multi
            list_col = config["q_map"].get(item_column_key_suffix)

            dept_items = []
            if list_col and list_col in df.columns:
                series = (
                    df[list_col].apply(parse_list_string).explode().dropna().str.strip()
                )
                dept_items.extend(series[series != ""].tolist())

            if not dept_items:
                add_to_markdown(f"- *No data for {dept_name} in {item_description}.*")
                continue
            contingency_data[dept_name] = pd.Series(Counter(dept_items))
            all_items.update(contingency_data[dept_name].index)

        elif col_name and col_name in df.columns:  # Single category column
            series = df[col_name].dropna().str.strip()
            series_filtered = series[
                (series != "")
                & (~series.str.contains("Unclear/Comment", case=False, na=False))
                & (~series.str.contains("Not Specified", case=False, na=False))
            ]
            if series_filtered.empty:
                add_to_markdown(
                    f"- *No valid data for {dept_name} in {item_description} from column {col_name}.*"
                )
                continue
            contingency_data[dept_name] = series_filtered.value_counts()
            all_items.update(contingency_data[dept_name].index)
        else:
            add_to_markdown(
                f"- *Column for `{item_column_key_suffix}` not found for department `{dept_name}`.*"
            )

    if len(contingency_data) < 2 or not all_items:
        add_to_markdown(
            f"- *Not enough data across departments or no items found for {item_description} to perform Chi-Square test.*"
        )
        return

    contingency_table = pd.DataFrame(index=sorted(list(all_items)))
    for dept_name, counts_series in contingency_data.items():
        contingency_table[dept_name] = counts_series
    contingency_table = contingency_table.fillna(0).astype(int)

    # Filter out rows (items) with sum of 0 across all departments
    contingency_table = contingency_table[contingency_table.sum(axis=1) > 0]

    if not contingency_table.empty:
        item_totals = contingency_table.sum(axis=1)
        OBSERVED_FREQ_THRESHOLD_FOR_GROUPING = 5
        items_to_group = item_totals[
            item_totals < OBSERVED_FREQ_THRESHOLD_FOR_GROUPING
        ].index.tolist()
        other_category_label = "Other"

        if len(items_to_group) > 0:
            grouped_counts = contingency_table.loc[items_to_group].sum(axis=0)
            contingency_table_remaining = contingency_table.drop(index=items_to_group)

            if other_category_label in contingency_table_remaining.index:
                contingency_table_remaining.loc[other_category_label] += grouped_counts
            else:
                grouped_counts.name = other_category_label
                contingency_table_remaining = pd.concat(
                    [contingency_table_remaining, pd.DataFrame(grouped_counts).T]
                )

            contingency_table = contingency_table_remaining.fillna(0).astype(int)

            add_to_markdown(
                f"- *Note: Categories with a total observed frequency across all departments < {OBSERVED_FREQ_THRESHOLD_FOR_GROUPING} (namely: {', '.join(items_to_group)}) were combined into the '{other_category_label}' category for this Chi-Square analysis.*"
            )

    if (
        contingency_table.empty
        or contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):
        add_to_markdown(
            f"- *Contingency table for {item_description} (after potential grouping of small categories) has insufficient dimensions (Rows: {contingency_table.shape[0]}, Columns: {contingency_table.shape[1]}) to perform a Chi-Square test of independence. Requires at least 2 rows and 2 columns.*"
        )
        return

    add_to_markdown("##### Observed Frequencies (Contingency Table):")
    add_to_markdown(contingency_table.to_markdown())

    # Visualization: Stacked Bar Chart of Proportions
    if not contingency_table.empty:
        contingency_table_proportions = contingency_table.apply(
            lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=0
        )
        fig_bar, ax_bar = plt.subplots(
            figsize=(max(10, contingency_table.shape[1] * 1.5), 7)
        )
        contingency_table_proportions.T.plot(
            kind="bar", stacked=True, ax=ax_bar, colormap="Spectral", width=0.8
        )
        ax_bar.set_title(
            f"Proportional Distribution of {item_description} by Department"
        )
        ax_bar.set_xlabel("Department")
        ax_bar.set_ylabel("Percentage (%)")
        ax_bar.legend(
            title=item_description,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize="small",
        )
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_plot_and_add_to_markdown_single(
            fig_bar,
            f"{item_description.replace(' ', '_').replace('/', '_').lower()}_dist_by_dept_stackedbar",
            subfolder,
            f"Fig: Proportional Distribution of {item_description} by Department",
        )

    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        add_to_markdown("##### Expected Frequencies:")
        expected_df = pd.DataFrame(
            expected, index=contingency_table.index, columns=contingency_table.columns
        )
        add_to_markdown(expected_df.round(2).to_markdown())

        num_cells_low_expected = (expected < min_expected_freq).sum().sum()
        percent_cells_low_expected = (
            (num_cells_low_expected / expected.size) * 100 if expected.size > 0 else 0
        )

        assumption_met_msg = f"- {num_cells_low_expected} cells ({percent_cells_low_expected:.1f}%) have expected frequency < {min_expected_freq}."
        if percent_cells_low_expected > 20:  # Common rule of thumb
            assumption_met_msg += " **Chi-square test may be unreliable.** Consider Fisher's Exact Test for smaller tables or combining categories if appropriate."
        add_to_markdown(assumption_met_msg)

        add_to_markdown(
            f"**Chi-Square Test Result:** Chi2 = {chi2:.3f}, p-value = {p:.4f}, Degrees of Freedom = {dof}"
        )
        if p < alpha:
            add_to_markdown(
                f"- There is a statistically significant association between department and the distribution of {item_description} (p < {alpha})."
            )

            add_to_markdown("##### Standardized Residuals Analysis:")
            residuals_std = (
                (contingency_table - expected_df)
                / np.sqrt(expected_df.replace(0, np.nan))
            )  # Added replace(0, np.nan) to avoid division by zero in sqrt if expected is 0

            if residuals_std.isnull().all().all():
                add_to_markdown(
                    "- *Could not compute standardized residuals, possibly due to zero expected frequencies in critical cells.*"
                )
            else:
                fig_width = max(8, contingency_table.shape[1] * 0.8 + 2)
                fig_height = max(6, contingency_table.shape[0] * 0.5 + 2)

                fig_res, ax_res = plt.subplots(figsize=(fig_width, fig_height))
                sns.heatmap(
                    residuals_std,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    fmt=".2f",
                    linewidths=0.5,
                    ax=ax_res,
                    annot_kws={"size": 8},
                )
                ax_res.set_title(
                    f"Standardized Residuals for {item_description}", fontsize=12
                )
                ax_res.set_xlabel("Department", fontsize=10)
                ax_res.set_ylabel(item_description, fontsize=10)
                plt.xticks(rotation=45, ha="right", fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                plt.tight_layout()
                save_plot_and_add_to_markdown_single(
                    fig_res,
                    f"{item_description.replace(' ', '_').replace('/', '_').lower()}_residuals_heatmap",
                    subfolder,
                    f"Fig: Standardized Residuals - {item_description}",
                )
                add_to_markdown(
                    "- *Standardized residuals highlight cells where observed counts deviate most from expected counts. Values approximately > |2| or < -|2| are often considered noteworthy contributors to a significant chi-square result.*"
                )
        else:
            add_to_markdown(
                f"- There is no statistically significant association between department and the distribution of {item_description} (p >= {alpha})."
            )

    except ValueError as e:
        add_to_markdown(
            f"- *Error performing Chi-Square test for {item_description}: {e}. This might be due to insufficient data dimensions (e.g., less than 2x2 table after grouping) or other data issues.*"
        )
    add_to_markdown("---")


def compare_paired_likert_wilcoxon_ttest(
    df, col1_name, col2_name, desc1, desc2, subfolder, alpha=0.05
):
    add_to_markdown(f"Paired Comparison: {desc1} vs. {desc2}", level=4)

    if col1_name not in df.columns or col2_name not in df.columns:
        add_to_markdown(
            f"- *One or both columns (`{col1_name}`, `{col2_name}`) not found. Skipping comparison.*"
        )
        return

    data1 = pd.to_numeric(df[col1_name], errors="raise")
    data2 = pd.to_numeric(df[col2_name], errors="raise")

    paired_df = pd.DataFrame({"data1": data1, "data2": data2}).dropna()
    if len(paired_df) < 10:  # Arbitrary small N threshold
        add_to_markdown(
            f"- *Too few paired observations (N={len(paired_df)}) for robust comparison. Skipping.*"
        )
        return

    differences = paired_df["data1"] - paired_df["data2"]

    # Visualization: Box plot of differences
    fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=differences,
        ax=ax_diff,
        orient="v",
        width=0.3,
    )
    sns.stripplot(
        data=differences, ax=ax_diff, color=".25", size=4, alpha=0.5, jitter=0.05
    )
    ax_diff.axhline(0, color="red", linestyle="--")
    ax_diff.set_ylabel(f"Difference ({desc1} - {desc2})")
    ax_diff.set_title(f"Paired Differences: {desc1} vs. {desc2}")
    ax_diff.set_xticks([])
    plt.tight_layout()
    save_plot_and_add_to_markdown_single(
        fig_diff,
        f"paired_diff_{desc1.split(' ')[0].lower().replace('/', '_')}_{desc2.split(' ')[0].lower().replace('/', '_')}_boxplot",
        subfolder,
        f"Fig: Distribution of Paired Differences ({desc1} vs. {desc2})",
    )

    normality_results = normality_check({"Differences": differences})
    differences_normal = normality_results["Differences"]["is_normal"]

    if differences_normal:
        add_to_markdown(
            f"Differences appear normally distributed. Proceeding with Paired t-test for {desc1} vs {desc2}."
        )
        t_stat, p_value = stats.ttest_rel(paired_df["data1"], paired_df["data2"])
        add_to_markdown(
            f"**Paired t-test Result:** t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}"
        )
    else:
        add_to_markdown(
            f"Differences do not appear normally distributed. Proceeding with Wilcoxon Signed-Rank Test for {desc1} vs {desc2}."
        )
        try:
            w_stat, p_value = stats.wilcoxon(paired_df["data1"], paired_df["data2"])
            add_to_markdown(
                f"**Wilcoxon Signed-Rank Test Result:** W-statistic = {w_stat:.3f}, p-value = {p_value:.4f}"
            )
        except ValueError as e:
            add_to_markdown(
                f"- *Error in Wilcoxon test (possibly all differences are zero): {e}*"
            )
            add_to_markdown(
                f"- *Mean {desc1}: {paired_df['data1'].mean():.2f}, Mean {desc2}: {paired_df['data2'].mean():.2f}*"
            )
            return

    if p_value < alpha:
        add_to_markdown(
            f"- There is a statistically significant difference between {desc1} (Mean={paired_df['data1'].mean():.2f}) and {desc2} (Mean={paired_df['data2'].mean():.2f}) (p < {alpha})."
        )
    else:
        add_to_markdown(
            f"- There is no statistically significant difference between {desc1} (Mean={paired_df['data1'].mean():.2f}) and {desc2} (Mean={paired_df['data2'].mean():.2f}) (p >= {alpha})."
        )
    add_to_markdown("---")


def perform_spearman_correlation_matrix(
    df, row_items, column_items, matrix_title, subfolder, alpha=0.05
):
    add_to_markdown(f"Spearman Correlation Matrix: {matrix_title}", level=4)

    valid_row_items = [
        col
        for col in row_items
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    valid_column_items = [
        col
        for col in column_items
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not valid_row_items or not valid_column_items:
        add_to_markdown(
            "- *Not enough numeric columns found for rows or columns to compute a correlation matrix.*"
        )
        return

    all_involved_cols = sorted(list(set(valid_row_items + valid_column_items)))

    if not all_involved_cols:
        add_to_markdown("- *No valid columns selected for correlation analysis.*")
        return

    corr_df = df[all_involved_cols].copy()

    for col in all_involved_cols:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

    corr_df.dropna(subset=all_involved_cols, how="all", inplace=True)
    corr_df.dropna(axis=1, how="all", inplace=True)

    if corr_df.empty or len(corr_df.columns) == 0:
        add_to_markdown(
            "- *No data available after cleaning for correlation analysis.*"
        )
        return

    valid_row_items_final = [col for col in valid_row_items if col in corr_df.columns]
    valid_column_items_final = [col for col in column_items if col in corr_df.columns]

    if not valid_row_items_final or not valid_column_items_final:
        add_to_markdown(
            "- *Not enough valid columns remain after NaN handling for rows or columns to compute a correlation matrix.*"
        )
        return

    is_1x1_self_correlation = (
        len(corr_df.columns) == 1
        and len(valid_row_items_final) == 1
        and len(valid_column_items_final) == 1
        and valid_row_items_final[0] == corr_df.columns[0]
        and valid_column_items_final[0] == corr_df.columns[0]
    )

    spearman_corr = None
    p_values_matrix = None

    if is_1x1_self_correlation:
        col_name = corr_df.columns[0]
        spearman_corr = pd.DataFrame(data=[[1.0]], index=[col_name], columns=[col_name])
        p_values_matrix = pd.DataFrame(
            data=[[1.0]], index=[col_name], columns=[col_name]
        )
        add_to_markdown("##### Correlation Coefficients (rho):")
        add_to_markdown(spearman_corr.round(3).to_markdown())
        add_to_markdown("##### P-values for Correlation Coefficients:")
        add_to_markdown(p_values_matrix.round(4).to_markdown())
        add_to_markdown(f"##### Significant Correlations (alpha = {alpha}):")
        add_to_markdown("- *N/A for single variable self-correlation.*")

    elif len(corr_df.columns) < 2:
        add_to_markdown(
            "- *Fewer than two unique columns available for correlation after preprocessing. Cannot compute full pairwise p-values with Pingouin.*"
        )
        spearman_corr_full = corr_df.corr(method="spearman")
        spearman_corr = spearman_corr_full.loc[
            valid_row_items_final, valid_column_items_final
        ]
        p_values_matrix = pd.DataFrame(
            index=spearman_corr.index, columns=spearman_corr.columns
        )
        add_to_markdown("##### Correlation Coefficients (rho):")
        if not spearman_corr.empty:
            add_to_markdown(spearman_corr.round(3).to_markdown())
        else:
            add_to_markdown("- *Correlation matrix is empty.*")
        add_to_markdown("##### P-values for Correlation Coefficients:")
        add_to_markdown(
            "- *P-values could not be computed due to insufficient columns for pairwise analysis.*"
        )
        add_to_markdown(f"##### Significant Correlations (alpha = {alpha}):")
        add_to_markdown("- *Cannot determine significance without p-values.*")
    else:
        spearman_corr_full = corr_df.corr(method="spearman")
        p_values_matrix_full = None
        try:
            p_values_pg = pg.pairwise_corr(corr_df, method="spearman", padjust="none")
            p_values_matrix_temp = p_values_pg.pivot(
                index="X", columns="Y", values="p-unc"
            )
            p_values_matrix_full = p_values_matrix_temp.reindex(
                index=spearman_corr_full.index, columns=spearman_corr_full.columns
            )
            for col_name_diag in spearman_corr_full.index:
                if col_name_diag in p_values_matrix_full.columns:
                    p_values_matrix_full.loc[col_name_diag, col_name_diag] = 1.0
            for r_name_fill in spearman_corr_full.index:
                for c_name_fill in spearman_corr_full.columns:
                    if (
                        pd.isna(p_values_matrix_full.loc[r_name_fill, c_name_fill])
                        and c_name_fill in p_values_matrix_full.index
                        and r_name_fill in p_values_matrix_full.columns
                        and pd.notna(p_values_matrix_full.loc[c_name_fill, r_name_fill])
                    ):
                        p_values_matrix_full.loc[r_name_fill, c_name_fill] = (
                            p_values_matrix_full.loc[c_name_fill, r_name_fill]
                        )
        except Exception as e:
            add_to_markdown(
                f"- *Could not compute p-values for correlations using Pingouin: {e}. Reporting coefficients only.*"
            )

        spearman_corr = spearman_corr_full.loc[
            valid_row_items_final, valid_column_items_final
        ]
        if p_values_matrix_full is not None:
            p_values_matrix = p_values_matrix_full.loc[
                valid_row_items_final, valid_column_items_final
            ]
        else:
            p_values_matrix = pd.DataFrame(
                index=spearman_corr.index, columns=spearman_corr.columns
            )

        add_to_markdown("##### Correlation Coefficients (rho):")
        if not spearman_corr.empty:
            add_to_markdown(spearman_corr.round(3).to_markdown())
        else:
            add_to_markdown("- *Correlation matrix is empty.*")

        if p_values_matrix is not None and not p_values_matrix.empty:
            add_to_markdown("##### P-values for Correlation Coefficients:")
            add_to_markdown(p_values_matrix.round(4).to_markdown())
            add_to_markdown(f"##### Significant Correlations (alpha = {alpha}):")
            significant_corrs = []
            for r_name in spearman_corr.index:
                for c_name in spearman_corr.columns:
                    rho = spearman_corr.loc[r_name, c_name]
                    p_val = p_values_matrix.loc[r_name, c_name]
                    if (
                        pd.notna(p_val)
                        and p_val < alpha
                        and not (r_name == c_name and rho == 1.0)
                    ):  # Exclude perfect self-correlation
                        significant_corrs.append(
                            f"- **{r_name} & {c_name}:** rho = {rho:.3f}, p = {p_val:.4f}"
                        )
            if significant_corrs:
                for item in significant_corrs:
                    add_to_markdown(item)
            else:
                add_to_markdown(
                    f"- No statistically significant correlations found at alpha = {alpha}."
                )
        else:
            add_to_markdown(f"##### Significant Correlations (alpha = {alpha}):")
            add_to_markdown(
                "- *Cannot determine significance as p-values are not available.*"
            )

    if spearman_corr is None or spearman_corr.empty:
        add_to_markdown(
            "- *Spearman correlation matrix is empty, skipping heatmap visualization.*"
        )
        add_to_markdown("---")
        return None, None  # Should return consistent types

    temp_slug = matrix_title.lower()
    temp_slug = re.sub(r"\s+", "_", temp_slug)
    temp_slug = re.sub(r"[^\w\-_.]", "", temp_slug)
    temp_slug = re.sub(r"_+", "_", temp_slug)
    temp_slug = temp_slug.strip("_.-")
    final_file_slug = f"{temp_slug}_heatmap"

    fig_h, ax_h = plt.subplots(
        figsize=(
            max(8, len(valid_column_items_final) * 0.9 + 2),
            max(6, len(valid_row_items_final) * 0.8 + 2),
        )
    )
    sns.heatmap(
        spearman_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax_h,
        vmin=-1,
        vmax=1,
        annot_kws={
            "size": 8
            if max(len(valid_row_items_final), len(valid_column_items_final)) < 15
            else 6
        },
    )
    ax_h.set_title(f"Spearman Correlation Matrix: {matrix_title}", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    save_plot_and_add_to_markdown_single(
        fig_h,
        final_file_slug,
        subfolder,
        f"Fig: Spearman Corr Heatmap - {matrix_title}",
    )
    add_to_markdown("---")
    return spearman_corr, p_values_matrix


def compare_groups_mannwhitney_ttest(
    df,
    value_col_name,
    group_col_name,
    group1_name,
    group2_name,
    metric_desc,
    subfolder,
    alpha=0.05,
):
    add_to_markdown(
        f"Group Comparison: {metric_desc} between {group1_name} and {group2_name} (on `{group_col_name}`)",
        level=4,
    )

    if value_col_name not in df.columns or group_col_name not in df.columns:
        add_to_markdown(
            f"- *Required columns (`{value_col_name}`, `{group_col_name}`) not found.*"
        )
        return

    data_val = pd.to_numeric(df[value_col_name], errors="coerce")

    group1_data = data_val[df[group_col_name] == group1_name].dropna()
    group2_data = data_val[df[group_col_name] == group2_name].dropna()

    if len(group1_data) < 3 or len(group2_data) < 3:
        add_to_markdown(
            f"- *Insufficient data for one or both groups (N1={len(group1_data)}, N2={len(group2_data)}). Skipping comparison.*"
        )
        return

    # Visualization: Box plots for the two groups
    plot_data_mw = pd.DataFrame(
        {
            metric_desc: pd.concat([group1_data, group2_data]),
            group_col_name: [group1_name] * len(group1_data)
            + [group2_name] * len(group2_data),
        }
    )

    fig_mw, ax_mw = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x=group_col_name,
        y=metric_desc,
        data=plot_data_mw,
        ax=ax_mw,
        order=[group1_name, group2_name],
    )
    sns.stripplot(
        x=group_col_name,
        y=metric_desc,
        data=plot_data_mw,
        ax=ax_mw,
        order=[group1_name, group2_name],
        color=".3",
        jitter=0.1,
        alpha=0.7,
    )
    ax_mw.set_title(
        f"Distribution of {metric_desc}\nby {group_col_name.replace('_', ' ').title()}"
    )
    ax_mw.set_xlabel(group_col_name.replace("_", " ").title())
    ax_mw.set_ylabel(metric_desc)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot_and_add_to_markdown_single(
        fig_mw,
        f"{metric_desc.replace(' ', '_').lower()}_by_{group_col_name.replace(' ', '_').lower()}_boxplot",
        subfolder,
        f"Fig: {metric_desc} by {group_col_name.replace('_', ' ').title()}",
    )

    add_to_markdown(
        f"- Group Sizes: {group1_name} (N={len(group1_data)}), {group2_name} (N={len(group2_data)})"
    )
    add_to_markdown(
        f"- Mean {metric_desc} for {group1_name}: {group1_data.mean():.2f} (SD={group1_data.std():.2f})"
    )
    add_to_markdown(
        f"- Mean {metric_desc} for {group2_name}: {group2_data.mean():.2f} (SD={group2_data.std():.2f})"
    )

    normality_results = normality_check(
        {group1_name: group1_data, group2_name: group2_data}
    )
    both_normal = (
        normality_results[group1_name]["is_normal"]
        and normality_results[group2_name]["is_normal"]
    )

    homogeneity_stat, homogeneity_p, variances_homogeneous = homogeneity_variance_check(
        [group1_data, group2_data], [group1_name, group2_name]
    )

    use_ttest = both_normal and variances_homogeneous

    if use_ttest:
        add_to_markdown("Proceeding with Independent Two-Sample t-test.")
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)
        add_to_markdown(
            f"**Independent t-test Result:** t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}"
        )
    else:
        if not both_normal and not variances_homogeneous:
            add_to_markdown(
                "Neither normality nor homogeneity of variances assumptions met. Proceeding with Mann-Whitney U Test."
            )
        elif not both_normal:
            add_to_markdown(
                "Normality assumption not met. Proceeding with Mann-Whitney U Test."
            )
        elif not variances_homogeneous:
            add_to_markdown("Homogeneity of variances assumption not met.")

        if not variances_homogeneous and both_normal:
            add_to_markdown(
                "Note: Using t-test with `equal_var=False` (Welch's t-test) as variances are unequal but data is normal."
            )
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            add_to_markdown(
                f"**Welch's t-test Result:** t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}"
            )
        else:
            u_stat, p_value = stats.mannwhitneyu(
                group1_data, group2_data, alternative="two-sided"
            )
            add_to_markdown(
                f"**Mann-Whitney U Test Result:** U-statistic = {u_stat:.3f}, p-value = {p_value:.4f}"
            )
            if (len(group1_data) * len(group2_data)) > 0:
                effect_size_common_lang = u_stat / (len(group1_data) * len(group2_data))
                rank_biserial_corr = 1 - (2 * effect_size_common_lang)
                add_to_markdown(
                    f"  - Rank-Biserial Correlation (approx. effect size): {rank_biserial_corr:.3f}"
                )
            else:
                add_to_markdown(
                    "  - Effect size calculation skipped due to zero sample size in one group."
                )

    if p_value < alpha:
        add_to_markdown(
            f"- There is a statistically significant difference in {metric_desc} between {group1_name} and {group2_name} (p < {alpha})."
        )
    else:
        add_to_markdown(
            f"- There is no statistically significant difference in {metric_desc} between {group1_name} and {group2_name} (p >= {alpha})."
        )
    add_to_markdown("---")


def compare_metric_across_groups_anova_kruskal(
    df,
    group_col_name,
    value_col_names_list,
    group_col_desc,
    value_metric_base_desc,
    subfolder,
    alpha=0.05,
    min_samples_per_group_for_test=3,
):
    if isinstance(value_col_names_list, str):
        value_col_names_list = [value_col_names_list]

    if group_col_name not in df.columns:
        add_to_markdown(
            f"- *Group column `{group_col_name}` not found for '{group_col_desc}' vs '{value_metric_base_desc}' analysis. Skipping.*"
        )
        return

    df_analysis = df.copy()
    group_cat_col_temp = f"{group_col_name}_cat_temp_analysis"
    df_analysis[group_cat_col_temp] = (
        df_analysis[group_col_name].astype("category").astype(str)
    )

    unique_group_categories = [
        cat
        for cat in df_analysis[group_cat_col_temp].unique()
        if pd.notna(cat) and cat != "nan"
    ]

    if not unique_group_categories:
        add_to_markdown(
            f"- *No valid categories found in `{group_col_name}` for '{group_col_desc}' vs '{value_metric_base_desc}' analysis. Skipping.*"
        )
        return

    add_to_markdown(f"Impact of {group_col_desc} on {value_metric_base_desc}", level=4)

    for value_col_name in value_col_names_list:
        if value_col_name not in df_analysis.columns:
            add_to_markdown(
                f"- *Value column `{value_col_name}` not found. Skipping for this metric within {group_col_desc} comparison.*"
            )
            continue

        metric_specific_title = (
            f"{value_metric_base_desc} ({value_col_name.replace('_', ' ').title()})"
        )
        add_to_markdown(
            f"Comparing {metric_specific_title} across {group_col_desc}:",
            level=5,
        )

        # Prepare data for plotting and testing
        collated_data_for_metric = []
        for group_cat_val_str in unique_group_categories:
            series_data = df_analysis.loc[
                df_analysis[group_cat_col_temp] == group_cat_val_str, value_col_name
            ]
            series_data_numeric = pd.to_numeric(series_data, errors="coerce").dropna()

            if not series_data_numeric.empty:
                for val in series_data_numeric:
                    collated_data_for_metric.append(
                        {
                            group_col_desc: group_cat_val_str,
                            value_col_name: val,
                        }
                    )

        if not collated_data_for_metric:
            add_to_markdown(
                f"- *No data available for {metric_specific_title} across any {group_col_desc} categories.*"
            )
            add_to_markdown("---")
            continue

        current_metric_vs_group_df = pd.DataFrame(collated_data_for_metric)

        # Groups that have at least one data point for this metric
        plot_group_names = sorted(
            current_metric_vs_group_df[group_col_desc].unique().tolist()
        )

        if len(plot_group_names) < 2:
            add_to_markdown(
                f"- Only one {group_col_desc} category ('{plot_group_names[0]}' if any) has data for {metric_specific_title}. Cannot perform comparative analysis or plot."
            )
            add_to_markdown("---")
            continue

        # Visualization (using all groups that have data)
        fig_plot, ax_plot = plt.subplots(
            figsize=(max(8, len(plot_group_names) * 1.5), 6)
        )
        sns.boxplot(
            x=group_col_desc,
            y=value_col_name,
            data=current_metric_vs_group_df,
            order=plot_group_names,
            ax=ax_plot,
        )
        sns.stripplot(
            x=group_col_desc,
            y=value_col_name,
            data=current_metric_vs_group_df,
            order=plot_group_names,
            ax=ax_plot,
            color=".25",
            size=3,
            alpha=0.5,
        )
        ax_plot.set_title(f"{metric_specific_title} by {group_col_desc}")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        save_plot_and_add_to_markdown_single(
            fig_plot,
            f"{value_col_name}_by_{group_col_name.replace(' ', '_').lower()}_boxplot",
            subfolder,
            f"Fig: {metric_specific_title} by {group_col_desc}",
        )

        # Prepare samples for statistical testing (groups must meet min_samples_per_group_for_test)
        samples_for_stat_test = []
        group_names_for_stat_test = []
        for grp_name_str in plot_group_names:
            data_series = current_metric_vs_group_df[
                (current_metric_vs_group_df[group_col_desc] == grp_name_str)
            ][value_col_name].dropna()

            if len(data_series) >= min_samples_per_group_for_test:
                samples_for_stat_test.append(data_series)
                group_names_for_stat_test.append(grp_name_str)
            else:
                add_to_markdown(
                    f"- *Category '{grp_name_str}' of {group_col_desc} (N={len(data_series)}) excluded from statistical test for {metric_specific_title} due to sample size < {min_samples_per_group_for_test}.*"
                )

        if len(samples_for_stat_test) < 2:
            add_to_markdown(
                f"- Not enough {group_col_desc} categories (found {len(samples_for_stat_test)}) with sufficient data (N>={min_samples_per_group_for_test} per group) for {metric_specific_title} to perform statistical test."
            )
            add_to_markdown("---")
            continue

        # Perform statistical tests
        normality_results = normality_check(
            {
                name: data
                for name, data in zip(group_names_for_stat_test, samples_for_stat_test)
            }
        )
        all_normal = all(
            res["is_normal"]
            for res in normality_results.values()
            if isinstance(res["is_normal"], bool)
        )

        homog_stat, homog_p, var_homog = homogeneity_variance_check(
            samples_for_stat_test, group_names_for_stat_test, alpha=alpha
        )

        melt_data_for_posthoc = []
        for i, name_str in enumerate(group_names_for_stat_test):
            for val_num in samples_for_stat_test[i]:
                melt_data_for_posthoc.append({"group": name_str, "value": val_num})
        melt_df_for_posthoc = pd.DataFrame(melt_data_for_posthoc)

        if all_normal and var_homog:
            add_to_markdown(f"Proceeding with ANOVA for {metric_specific_title}.")
            f_stat, p_val = stats.f_oneway(*samples_for_stat_test)
            add_to_markdown(f"  - ANOVA: F={f_stat:.2f}, p={p_val:.4f}")
            if p_val < alpha:
                add_to_markdown(
                    f"    - Statistically significant difference found for {metric_specific_title} across {group_col_desc} categories (p < {alpha})."
                )
                if len(samples_for_stat_test) >= 2:
                    try:
                        tukey_results = pairwise_tukeyhsd(
                            melt_df_for_posthoc["value"],
                            melt_df_for_posthoc["group"],
                            alpha=alpha,
                        )
                        add_to_markdown("##### Tukey's HSD Post-hoc Test Results:")
                        add_to_markdown(f"```\n{tukey_results}\n```")
                        fig_tukey, ax_tukey = plt.subplots(
                            figsize=(10, max(4, len(group_names_for_stat_test) * 0.5))
                        )
                        tukey_results.plot_simultaneous(ax=ax_tukey)
                        ax_tukey.set_title(
                            f"Tukey's HSD: Pairwise Comparison for {metric_specific_title}"
                        )
                        save_plot_and_add_to_markdown_single(
                            fig_tukey,
                            f"{value_col_name}_by_{group_col_name.replace(' ', '_').lower()}_tukey_hsd",
                            subfolder,
                            f"Fig: Tukey's HSD for {metric_specific_title} by {group_col_desc}",
                        )
                    except Exception as e:
                        add_to_markdown(f"- *Error performing Tukey's HSD: {e}.*")
            else:
                add_to_markdown(
                    f"    - No statistically significant difference found (p >= {alpha})."
                )
        else:
            reason_list = []
            if not all_normal:
                reason_list.append("non-normality")
            if not var_homog:
                reason_list.append("heterogeneity of variances")
            reason_str = (
                ", ".join(reason_list)
                if reason_list
                else "undetermined assumption violation"
            )
            add_to_markdown(
                f"Assumptions for ANOVA not met ({reason_str}). Proceeding with Kruskal-Wallis Test for {metric_specific_title}."
            )
            h_stat, p_val = stats.kruskal(*samples_for_stat_test)
            add_to_markdown(f"  - Kruskal-Wallis: H={h_stat:.2f}, p={p_val:.4f}")
            if p_val < alpha:
                add_to_markdown(
                    f"    - Statistically significant difference found for {metric_specific_title} across {group_col_desc} categories (p < {alpha})."
                )
                if len(samples_for_stat_test) > 1:
                    try:
                        dunn_results = posthoc_dunn(
                            melt_df_for_posthoc,
                            val_col="value",
                            group_col="group",
                            p_adjust="bonferroni",
                        )
                        add_to_markdown(
                            "##### Dunn's Post-hoc Test Results (Bonferroni corrected):"
                        )
                        add_to_markdown(dunn_results.to_markdown())
                    except Exception as e:
                        add_to_markdown(f"- *Error performing Dunn's test: {e}.*")
            else:
                add_to_markdown(
                    f"    - No statistically significant difference found (p >= {alpha})."
                )
        add_to_markdown("---")


# --- Main Analysis Script ---
def advanced_inferential_analysis(df: pd.DataFrame):
    global markdown_content
    markdown_content = []

    script_start_time = datetime.datetime.now()
    add_to_markdown("Phase 3: Advanced Inferential Analysis", level=1)
    add_to_markdown(
        f"Execution Start Time: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_to_markdown("---")

    opex_cols_present = all(col in df.columns for col in OPEX_COMPOSITE_COLS[:-1])
    if not opex_cols_present:
        add_to_markdown(
            "OpEx composite scores not found, attempting to calculate them...", level=3
        )
        q_opex_ce = [f"q{i}_opex_ce_" for i in range(11, 18)]
        q_opex_cpi = [f"q{i}_opex_cpi_" for i in range(18, 28)]
        q_opex_ea = [f"q{i}_opex_ea_" for i in range(28, 34)]

        ce_cols_in_df = [
            col for col in df.columns if any(q_prefix in col for q_prefix in q_opex_ce)
        ]
        cpi_cols_in_df = [
            col for col in df.columns if any(q_prefix in col for q_prefix in q_opex_cpi)
        ]
        ea_cols_in_df = [
            col for col in df.columns if any(q_prefix in col for q_prefix in q_opex_ea)
        ]

        if ce_cols_in_df:
            df["opex_ce_composite"] = df[ce_cols_in_df].mean(axis=1, skipna=True)
        if cpi_cols_in_df:
            df["opex_cpi_composite"] = df[cpi_cols_in_df].mean(axis=1, skipna=True)
        if ea_cols_in_df:
            df["opex_ea_composite"] = df[ea_cols_in_df].mean(axis=1, skipna=True)
        opex_calculated_composites = [
            c
            for c in ["opex_ce_composite", "opex_cpi_composite", "opex_ea_composite"]
            if c in df.columns
        ]
        if len(opex_calculated_composites) > 0:
            df["opex_overall_composite"] = df[opex_calculated_composites].mean(
                axis=1, skipna=True
            )
            add_to_markdown("OpEx composite scores calculated and added to DataFrame.")
        else:
            add_to_markdown(
                "Could not calculate OpEx composite scores. Some analyses might fail."
            )

    # Ensure OpEx columns list is updated if they were calculated
    opex_composite_cols_present = [
        col for col in OPEX_COMPOSITE_COLS if col in df.columns
    ]

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

    # --- Step 1: Cross-Domain Comparison (Inferential Tests) ---
    add_to_markdown("Step 1: Cross-Domain Comparison (Inferential Statistics)", level=2)
    cross_domain_subfolder = "1_cross_domain_stats"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, cross_domain_subfolder))

    add_to_markdown(
        "Comparisons of Mean Likert Scores Across Departments (Maintenance, Quality, Production)",
        level=3,
    )

    for metric_title, q_key_suffix in comparable_likert_metrics.items():
        cols_map = {
            dept: config["q_map"].get(q_key_suffix)
            for dept, config in DEPT_CONFIGS.items()
        }
        compare_likert_anova_kruskal(df, cols_map, metric_title, cross_domain_subfolder)

    add_to_markdown(
        "Comparisons of Categorical Distributions Across Departments (Chi-Square)",
        level=3,
    )
    compare_categorical_chi_square(
        df,
        DEPT_CONFIGS,
        "da_tools_text",
        "Data Analysis Tools Used",
        cross_domain_subfolder,
    )
    compare_categorical_chi_square(
        df,
        DEPT_CONFIGS,
        "ai_tools_mc",
        "AI/GenAI Tools Used",
        cross_domain_subfolder,
    )
    compare_categorical_chi_square(
        df,
        DEPT_CONFIGS,
        "out_challenges_list_mc",
        "Challenges in Utilizing Data (Multi-choice)",
        cross_domain_subfolder,
    )

    # --- Step 2: Barriers Specific to AI/ML and Transparency (Inferential Tests) ---
    add_to_markdown(
        "Step 2: Barriers to AI/ML and Transparency (Inferential Statistics)", level=2
    )
    barriers_stats_subfolder = "2_barriers_stats"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, barriers_stats_subfolder))

    compare_paired_likert_wilcoxon_ttest(
        df,
        "q73_barriers_ps_ai_limit",
        "q74_barriers_ps_genai_limit",
        "AI/ML Privacy/Security Limits",
        "GenAI Privacy/Security Limits",
        barriers_stats_subfolder,
    )

    transparency_importance_cols = {
        "Maintenance": "q78_barriers_ai_transparency_importance_maint",
        "Quality": "q79_barriers_ai_transparency_importance_qual",
        "Production": "q80_barriers_ai_transparency_importance_prod",
    }
    compare_likert_anova_kruskal(
        df,
        transparency_importance_cols,
        "Importance of AI Transparency",
        barriers_stats_subfolder,
    )

    # --- Step 3: Connecting Data/AI/ML to Operational Excellence ---
    add_to_markdown("Step 3: Connecting Data/AI/ML to Operational Excellence", level=2)
    opex_conn_subfolder = "3_opex_connections"
    ensure_dir(os.path.join(OUTPUT_BASE_DIR, opex_conn_subfolder))

    add_to_markdown("Correlation Analysis (Spearman's Rho)", level=3)

    if not opex_composite_cols_present:
        add_to_markdown(
            "*OpEx composite scores not available. Skipping OpEx correlation analyses.*"
        )
    else:
        for dept_name, config in DEPT_CONFIGS.items():
            dept_practice_cols_keys = [
                "dg_collection_freq",
                "da_analysis_freq",
                "ai_usage",
                "dm_insights_drive_decisions",
                "tl_staff_trained",
                "tl_data_driven_culture",
                "out_measurable_improvements",
            ]
            dept_practice_cols = [
                config["q_map"].get(key)
                for key in dept_practice_cols_keys
                if config["q_map"].get(key) in df.columns
            ]

            if dept_practice_cols:
                perform_spearman_correlation_matrix(
                    df,
                    opex_composite_cols_present,
                    dept_practice_cols,
                    f"OpEx Composites vs {dept_name} Data & AI Practices",
                    opex_conn_subfolder,
                )

        barrier_likert_cols_keys = [
            "q73_barriers_ps_ai_limit",
            "q74_barriers_ps_genai_limit",
            "q77_barriers_ai_transparency_efforts",
        ]
        barrier_likert_cols = [
            col for col in barrier_likert_cols_keys if col in df.columns
        ]
        if barrier_likert_cols:
            perform_spearman_correlation_matrix(
                df,
                opex_composite_cols_present,
                barrier_likert_cols,
                "OpEx Composites vs General AI/ML Barriers",
                opex_conn_subfolder,
            )

    add_to_markdown("Group Comparison Analysis", level=3)
    # Example: High vs. Low AI Adopters (based on overall AI usage)
    # Create an overall AI adoption score (average AI usage across M, Q, P)
    ai_usage_cols_keys = [DEPT_CONFIGS[d]["q_map"]["ai_usage"] for d in DEPT_CONFIGS]
    ai_usage_cols = [col for col in ai_usage_cols_keys if col in df.columns]

    if ai_usage_cols:
        df["overall_ai_adoption_score"] = df[ai_usage_cols].mean(axis=1, skipna=True)
        median_ai_adoption = df["overall_ai_adoption_score"].median()

        if (
            pd.notna(median_ai_adoption)
            and median_ai_adoption > df["overall_ai_adoption_score"].min()
            and median_ai_adoption < df["overall_ai_adoption_score"].max()
        ):
            df["ai_adoption_group"] = np.select(
                [
                    df["overall_ai_adoption_score"] < median_ai_adoption,
                    df["overall_ai_adoption_score"] >= median_ai_adoption,
                ],
                ["Low AI Adopters", "High AI Adopters"],
                default="Unknown",
            )
            df.loc[df["overall_ai_adoption_score"].isna(), "ai_adoption_group"] = (
                "Unknown"
            )
            df["ai_adoption_group"] = df["ai_adoption_group"].astype("category")

            # Check if both groups exist and have enough data
            low_adopters_exist = (
                "Low AI Adopters" in df["ai_adoption_group"].cat.categories
            )
            high_adopters_exist = (
                "High AI Adopters" in df["ai_adoption_group"].cat.categories
            )

            if (
                low_adopters_exist
                and high_adopters_exist
                and opex_composite_cols_present
            ):
                add_to_markdown(
                    "Comparing OpEx Scores for High vs. Low AI Adopter Groups", level=4
                )
                for opex_score_col in opex_composite_cols_present:
                    compare_groups_mannwhitney_ttest(
                        df,
                        opex_score_col,
                        "ai_adoption_group",
                        "High AI Adopters",
                        "Low AI Adopters",
                        f"{opex_score_col.replace('_', ' ').title()}",
                        opex_conn_subfolder,
                    )
            else:
                add_to_markdown(
                    "- *Could not form distinct 'High AI Adopters' and 'Low AI Adopters' groups with sufficient data, or OpEx scores are missing. Skipping this comparison.*"
                )
        else:
            add_to_markdown(
                "- *Median AI adoption score is NaN, or does not allow for a meaningful split into High/Low groups. Cannot create AI adopter groups. Skipping this comparison.*"
            )
    else:
        add_to_markdown(
            "- *AI usage columns not found. Skipping AI Adopter group comparison.*"
        )

    compare_metric_across_groups_anova_kruskal(
        df,
        group_col_name="q8_company_size",
        value_col_names_list=opex_composite_cols_present,
        group_col_desc="Company Size",
        value_metric_base_desc="OpEx Score",
        subfolder=opex_conn_subfolder,
        alpha=0.05,
    )

    add_to_markdown(
        "Impact of Company Size on Aggregated Departmental Likert Metrics", level=3
    )
    aggregated_likert_column_details = []
    for original_metric_title, q_key_suffix in comparable_likert_metrics.items():
        current_metric_department_columns = []
        for dept_config in DEPT_CONFIGS.values():
            dept_specific_col_name = dept_config["q_map"].get(q_key_suffix)
            if dept_specific_col_name and dept_specific_col_name in df.columns:
                df[dept_specific_col_name] = pd.to_numeric(
                    df[dept_specific_col_name], errors="coerce"
                )
                current_metric_department_columns.append(dept_specific_col_name)

        if current_metric_department_columns:
            new_aggregated_col_name = f"agg_mean_{q_key_suffix}"
            df[new_aggregated_col_name] = df[current_metric_department_columns].mean(
                axis=1, skipna=True
            )
            aggregated_likert_column_details.append(
                (new_aggregated_col_name, original_metric_title)
            )

    if aggregated_likert_column_details:
        for agg_col_name, original_title_for_metric in aggregated_likert_column_details:
            compare_metric_across_groups_anova_kruskal(
                df,
                group_col_name="q8_company_size",
                value_col_names_list=[agg_col_name],
                group_col_desc="Company Size",
                value_metric_base_desc=f"Mean {original_title_for_metric}",
                subfolder=opex_conn_subfolder,
                alpha=0.05,
            )
    else:
        add_to_markdown(
            "- *No comparable Likert metrics could be aggregated across departments for company size comparison.*"
        )

    # --- Finalize Markdown ---
    add_to_markdown("\nEnd of Phase 3 Advanced Inferential Analysis Report.")
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
    return df


if __name__ == "__main__":
    ensure_dir(OUTPUT_BASE_DIR)

    try:
        df_survey = pd.read_csv("1_data_preparation/cleaned_survey_data.csv")
        print(f"Successfully loaded data. Shape: {df_survey.shape}")
    except FileNotFoundError:
        print("Error: The input data file was not found. Please check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if df_survey.empty:
        print("Loaded DataFrame is empty. Please check the CSV file.")
        sys.exit(1)

    df_survey_processed = advanced_inferential_analysis(df_survey.copy())
