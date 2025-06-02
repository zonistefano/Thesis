import datetime
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Union

import openai
import pandas as pd
from openai import OpenAI

# --- Helper Types ---
QuestionInfoDict = Dict[
    str, Union[str, List[str], Dict[str, int], bool, Optional[List[str]]]
]
QuestionTypeInfo = Dict[str, QuestionInfoDict]


class LLMCategorizationLogEntry(TypedDict):
    question_id: str
    original_text: str
    raw_llm_output: str
    final_categories: Union[str, List[str]]
    status: str


# --- Constants for LLM Processing ---
LLM_UNCLEAR_MARKER: str = "LLM_INTERNAL_UNCLEAR_COMMENT"
LLM_ERROR_MARKER: str = "LLM_INTERNAL_ERROR"
LLM_SKIPPED_MARKER: str = "LLM_INTERNAL_SKIPPED_NO_CLIENT"
LLM_DEFAULT_DELIMITER: str = ";"
LLM_MULTI_OUTPUT_JOIN_DELIMITER: str = "; "


# --- Configuration Class ---
@dataclass
class SurveyConfig:
    """Holds all static configuration for the survey data processing."""

    OPENROUTER_API_KEY: Optional[str] = os.getenv(
        "OPENROUTER_API_KEY",
        "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",
    )
    LLM_MODEL: str = "google/gemini-2.5-flash-preview"
    LLM_PLACEHOLDER_KEY: str = "sk-or-v1-YOUR-OPENROUTER-API-KEY"

    POSITION_CATEGORIES: List[str] = field(
        default_factory=lambda: [
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
    )
    INDUSTRY_SECTOR_CATEGORIES: List[str] = field(
        default_factory=lambda: [
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
    )
    COUNTRY_CATEGORIES: List[str] = field(
        default_factory=lambda: [
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
    )
    DA_TOOL_CATEGORIES: List[str] = field(
        default_factory=lambda: [
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
    )

    MC_STANDARD_OPTIONS_MAP: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "q38_maint_ai_tools_mc": [
                "ChatGPT (OpenAI)",
                "Copilot (Microsoft)",
                "Gemini (Google)",
                "SAP AI",
                "In-House Developed ML Solution",
                "In-House Developed AI Solution",
                "Other",
            ],
            "q51_qual_ai_tools_mc": [
                "ChatGPT (OpenAI)",
                "Copilot (Microsoft)",
                "Gemini (Google)",
                "SAP AI",
                "In-House Developed ML Solution",
                "In-House Developed AI Solution",
                "Other",
            ],
            "q64_prod_ai_tools_mc": [
                "ChatGPT (OpenAI)",
                "Copilot (Microsoft)",
                "Gemini (Google)",
                "SAP AI",
                "In-House Developed ML Solution",
                "In-House Developed AI Solution",
                "Other",
            ],
            "q46_maint_out_challenges_list_mc": [
                "Data quality issues",
                "Lack of skilled personnel or expertise in data analytics",
                "Integration difficulties",
                "High costs",
                "Difficulty in extracting insights from data",
                "Resistance to change, challenging culture change management",
                "Insufficient support from top management",
                "Other",
            ],
            "q59_qual_out_challenges_list_mc": [
                "Data quality issues",
                "Lack of skilled personnel or expertise in data analytics",
                "Integration difficulties",
                "High costs",
                "Difficulty in extracting insights from data",
                "Resistance to change, challenging culture change management",
                "Insufficient support from top management",
                "Other",
            ],
            "q72_prod_out_challenges_list_mc": [
                "Data quality issues",
                "Lack of skilled personnel or expertise in data analytics",
                "Integration difficulties",
                "High costs",
                "Difficulty in extracting insights from data",
                "Resistance to change, challenging culture change management",
                "Insufficient support from top management",
                "Other",
            ],
            "q75_barriers_aiml_general_mc": [
                "Unauthorized access to sensitive operational data",
                "Lack of transparency and Explainability",
                "Potential for biased or inaccurate predictions affecting operations",
                "Integration security issues with existing IT systems",
                "Difficulties in monitoring and auditing AI model behavior over time",
                "Other",
            ],
            "q76_barriers_ps_genai_concerns_mc": [
                "Intellectual Property (IP) Protection",
                "Regulatory Compliance",
                "Model Bias and Ethical Concerns",
                "Third-Party AI Service Risks",
                "Lack of Transparency and Explainability",
                "Other",
            ],
            "q81_barriers_ai_transparency_methods_mc": [
                "Clear documentation of AI models and data sources",
                "Explainable AI (XAI) tools and techniques",
                "Transparent data governance policies",
                "Third-party validation and certification",
                "Employee training and awareness programs",
                "Other",
            ],
        }
    )
    MC_ITEM_DESCRIPTION_MAP: Dict[str, str] = field(
        default_factory=lambda: {
            "q38_maint_ai_tools_mc": "Maintenance AI/GenAI tool",
            "q51_qual_ai_tools_mc": "Quality AI/GenAI tool",
            "q64_prod_ai_tools_mc": "Production AI/GenAI tool",
            "q46_maint_out_challenges_list_mc": "challenge in utilizing data for Maintenance",
            "q59_qual_out_challenges_list_mc": "challenge in utilizing data for Quality",
            "q72_prod_out_challenges_list_mc": "challenge in utilizing data for Production",
            "q75_barriers_aiml_general_mc": "general AI/ML barrier",
            "q76_barriers_ps_genai_concerns_mc": "GenAI privacy/security concern",
            "q81_barriers_ai_transparency_methods_mc": "AI transparency method",
        }
    )

    COLUMN_MAPPING_DEFINITION: Dict[str, str] = field(
        default_factory=lambda: {
            "Please provide your full name": "q1_full_name",
            "What is your current position or role within the company?": "q2_position",
            "To which department do you primarily belong?": "q3_department",
            "How many years of experience do you have in your current role?": "q4_experience_role",
            "How long have you been working with this company?": "q5_experience_company",
            "Please provide your company's name": "q6_company_name",
            "In which industry sector does your company primarily operate?": "q7_industry_sector",
            "What is the size of your company in terms of the number of employees?": "q8_company_size",
            "In which country is your company's primary headquarters located?": "q9_company_hq_country",
            "What is your company's approximate annual revenue?": "q10_company_revenue",
            "Cultural Enablers.Our organization demonstrates respect for every individual within organization (both management and nonmanagement)": "q11_opex_ce_respect_internal",
            "Cultural Enablers.Our organization demonstrates respect for every individual outside the organization (vendors, customers and stakeholders)": "q12_opex_ce_respect_external",
            "Cultural Enablers.Our organization is concerned about the health of their employees": "q13_opex_ce_employee_health",
            "Cultural Enablers.Our organization is concerned about the safety of their employees": "q14_opex_ce_employee_safety",
            "Cultural Enablers.Our organization is careful about rights of social community": "q15_opex_ce_social_community_rights",
            "Cultural Enablers.Our organization considers trainings as an investment for its future improvements": "q16_opex_ce_training_investment",
            "Cultural Enablers.Our organization considers hands-on exposure of its employees as a future investment": "q17_opex_ce_handson_investment",
            "Continuous Process Improvement.Our organization immediately provides products/services on customer's demand": "q18_opex_cpi_customer_demand_immediate",
            "Continuous Process Improvement.Products/services provided by our organization are flexible according to the demand of customers": "q19_opex_cpi_products_flexible",
            "Continuous Process Improvement.Our organization continuously monitors deterioration in products/services in order to maintain standards": "q20_opex_cpi_monitor_deterioration",
            "Continuous Process Improvement.Our organization ensures value addition at each step/processes of products/services": "q21_opex_cpi_value_addition",
            "Continuous Process Improvement.Our organization uses direct observations to evaluate the processes": "q22_opex_cpi_direct_observations",
            "Continuous Process Improvement.Employees in our organization are empowered to pursue elimination of defects in the processes": "q23_opex_cpi_employee_empowerment_defects",
            "Continuous Process Improvement.In our organization managers are empowered to stop a process when defect in a process is identified": "q24_opex_cpi_manager_empowerment_stop_process",
            "Continuous Process Improvement.Our organization takes counter measures to prevent the reoccurrence of defects in the process": "q25_opex_cpi_counter_measures_defects",
            "Continuous Process Improvement.In our organization improvement in our processes is an integral part of daily work": "q26_opex_cpi_improvement_daily_work",
            "Continuous Process Improvement.Our organization has effective tools for gathering reliable data": "q27_opex_cpi_tools_reliable_data",
            "Enterprise Alignment.In our organization, planning process extends respect to the inputs from different levels": "q28_opex_ea_planning_respect_inputs",
            "Enterprise Alignment.In our organization, planning process involves input from different levels of individuals": "q29_opex_ea_planning_involves_inputs",
            "Enterprise Alignment.In our organization, work description is defined for every individual at all levels": "q30_opex_ea_work_description_defined",
            "Enterprise Alignment.In our organization, work description is standardized for every individual at all levels": "q31_opex_ea_work_description_standardized",
            "Enterprise Alignment.Our organization defines relevant metrics/data necessary for designated users": "q32_opex_ea_defines_metrics_data",
            "Enterprise Alignment.Our organization encourages employees to adjust personal values to achieve organizational objectives": "q33_opex_ea_adjust_personal_values",
            "Data Generation.We regularly collect data related to maintenance.": "q34_maint_dg_collection_freq",
            "Data Analysis.Data collected is regularly analyzed to inform decisions.": "q35_maint_da_analysis_freq",
            "What tools or software do you currently use for maintenance data analysis?": "q36_maint_da_tools_text",
            "AI & GenAI Usage.We use AI and Generative AI (ChatGPT, Copilot, ...)\xa0for Maintenance.": "q37_maint_ai_usage",
            "Which AI/GenAI tools does your organization use for Maintenance?": "q38_maint_ai_tools_mc",
            "Usage in Decision Making.Data insights from maintenance analyses drive our operational decisions.": "q39_maint_dm_insights_drive_decisions",
            "Training and Literacy.Staff involved in maintenance are trained to analyze and utilize data effectively.": "q40_maint_tl_staff_trained",
            "Training and Literacy.There is a strong culture of data-driven decision-making within our maintenance teams.": "q41_maint_tl_data_driven_culture",
            "Outcome.Data usage in maintenance has led to measurable improvements.": "q42_maint_out_measurable_improvements",
            "Outcome.Data-driven insights help us to identify areas for continuous improvement in our maintenance processes.": "q43_maint_out_insights_ci",
            "Outcome.Data helps us in minimizing the cost of maintenance by optimizing resources.": "q44_maint_out_data_minimizes_cost",
            "Outcome.Our organization faces challenges in effectively utilizing data for maintenance improvement.": "q45_maint_out_challenges_utilizing_data",
            "What challenges do you face in utilizing data for maintenance?": "q46_maint_out_challenges_list_mc",
            "Data Generation.We regularly collect data related to quality.": "q47_qual_dg_collection_freq",
            "Data Analysis.Data collected is regularly analyzed to inform decisions.1": "q48_qual_da_analysis_freq",
            "What tools or software do you currently use for quality data analysis?": "q49_qual_da_tools_text",
            "AI & GenAI Usage.We use AI and Generative AI (ChatGPT, Copilot, ...)\xa0for Quality.": "q50_qual_ai_usage",
            "Which AI/GenAI tools does your organization use for Quality?": "q51_qual_ai_tools_mc",
            "Usage in Decision Making.Data insights from quality analyses drive our operational decisions.": "q52_qual_dm_insights_drive_decisions",
            "Training and Literacy.Staff involved in quality are trained to analyze and utilize data effectively.": "q53_qual_tl_staff_trained",
            "Training and Literacy.There is a strong culture of data-driven decision-making within our quality teams.": "q54_qual_tl_data_driven_culture",
            "Outcome.Data usage in quality has led to measurable improvements.": "q55_qual_out_measurable_improvements",
            "Outcome.Data-driven insights help us to identify areas for continuous improvement in our quality processes.": "q56_qual_out_insights_ci",
            "Outcome.Data helps us in minimizing the cost of quality by optimizing resources.": "q57_qual_out_data_minimizes_cost",
            "Outcome.Our organization faces challenges in effectively utilizing data for quality improvement.": "q58_qual_out_challenges_utilizing_data",
            "What challenges do you face in utilizing data for quality?": "q59_qual_out_challenges_list_mc",
            "Data Generation.We regularly collect data related to production.": "q60_prod_dg_collection_freq",
            "Data Analysis.Data collected is regularly analyzed to inform decisions.2": "q61_prod_da_analysis_freq",
            "What tools or software do you currently use for production data analysis?": "q62_prod_da_tools_text",
            "AI & GenAI Usage.We use AI and Generative AI (ChatGPT, Copilot, ...)\xa0for Production.": "q63_prod_ai_usage",
            "Which AI/GenAI tools does your organization use for Production?": "q64_prod_ai_tools_mc",
            "Usage in Decision Making.Data insights from production analyses drive our operational decisions.": "q65_prod_dm_insights_drive_decisions",
            "Training and Literacy.Staff involved in production are trained to analyze and utilize data effectively.": "q66_prod_tl_staff_trained",
            "Training and Literacy.There is a strong culture of data-driven decision-making within our production teams.": "q67_prod_tl_data_driven_culture",
            "Outcome.Data usage in production has led to measurable improvements.": "q68_prod_out_measurable_improvements",
            "Outcome.Data-driven insights help us to identify areas for continuous improvement in our production processes.": "q69_prod_out_insights_ci",
            "Outcome.Data helps us in minimizing the cost of production by optimizing resources.": "q70_prod_out_data_minimizes_cost",
            "Outcome.Our organization faces challenges in effectively utilizing data for production improvement.": "q71_prod_out_challenges_utilizing_data",
            "What challenges do you face in utilizing data for production?": "q72_prod_out_challenges_list_mc",
            "Privacy/Security.Concerns about data privacy and/or security limit our ability to use AI/ML effectively.": "q73_barriers_ps_ai_limit",
            "Privacy/Security.Concerns about data privacy and/or security limit our ability to use Generative AI effectively.": "q74_barriers_ps_genai_limit",
            "What barriers does your organization face regarding the usage of AI/ML?": "q75_barriers_aiml_general_mc",
            "What specific privacy and/or security concerns does your organization face regarding the usage of Generative AI (ChatGPT, Copilot, ...)?": "q76_barriers_ps_genai_concerns_mc",
            "Transparency.We make sure that our AI models are transparent and understandable for business purposes.": "q77_barriers_ai_transparency_efforts",
            "How important is for AI models to be transparent and Interpretable for ___\n.Maintenance": "q78_barriers_ai_transparency_importance_maint",
            "How important is for AI models to be transparent and Interpretable for ___\n.Quality": "q79_barriers_ai_transparency_importance_qual",
            "How important is for AI models to be transparent and Interpretable for ___\n.Production": "q80_barriers_ai_transparency_importance_prod",
            "What methods does your organization use to improve transparency for AI and Data Driven Decision Making?": "q81_barriers_ai_transparency_methods_mc",
        }
    )

    STANDARDIZED_NULL_LIKE_ENTRIES: List[str] = field(
        default_factory=lambda: [
            "n/a",
            "na",
            "none",
            "not applicable",
            "-",
            ".",
            "no other",
            "nothing",
            "",
            "not specified",
            "unknown",
            "no tools",
            "no specific tools",
            "/",
            "null",
            "nan",
            "no answer",
            "skip",
            "skipped",
            "prefer not to say",
            "niente",
            "nessuno",
            "nessuna",
            "nulla",
            "non usato",
            "noone",
            "no",
            "anonymous",
            "no_tool_specified",
        ]
    )
    STANDARDIZED_NULL_LIKE_ENTRIES_LOWERCASE: List[str] = field(init=False)

    DEPT_VALUES: List[str] = field(
        default_factory=lambda: [
            "Maintenance",
            "Quality Assurance",
            "Production",
            "Supply Chain/Logistics",
            "Sales/Marketing",
            "IT",
            "Finance/Accounting",
            "Human Resources",
            "Research & Development (R&D)",
            "Other",
        ]
    )
    EXP_ROLE_VALUES: List[str] = field(
        default_factory=lambda: [
            "Less than 1 year",
            "1-3 years",
            "4-6 years",
            "7-10 years",
            "More than 10 years",
        ]
    )
    COMPANY_SIZE_VALUES: List[str] = field(
        default_factory=lambda: [
            "1-50",
            "51-250",
            "251-1000",
            "1001-5000",
            "More than 5000",
        ]
    )
    COMPANY_REVENUE_VALUES: List[str] = field(
        default_factory=lambda: [
            "<$1 million",
            "$1 million - $10 million",
            "$10 million - $50 million",
            "$50 million - $100 million",
            "$100 million - $500 million",
            ">$500 million",
        ]
    )

    STD_LIKERT_5_POINT_LABELS: List[str] = field(
        default_factory=lambda: [
            "Strongly disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly agree",
        ]
    )
    STD_LIKERT_5_POINT_MAP: Dict[str, int] = field(init=False)

    TRANSP_IMPORTANCE_LIKERT_LABELS: List[str] = field(
        default_factory=lambda: [
            "Very unimportant",
            "Unimportant",
            "Neutral",
            "Important",
            "Very important",
        ]
    )
    TRANSP_IMPORTANCE_LIKERT_MAP: Dict[str, int] = field(init=False)

    COLUMN_MAPPING: Dict[str, str] = field(init=False)
    MULTI_SELECT_DELIMITER: str = ";"
    OUTPUT_BASE_DIR: str = "1_data_preparation"

    def __post_init__(self) -> None:
        self.STANDARDIZED_NULL_LIKE_ENTRIES_LOWERCASE = [
            entry.lower() for entry in self.STANDARDIZED_NULL_LIKE_ENTRIES
        ]
        self.STD_LIKERT_5_POINT_MAP = {
            label: i + 1 for i, label in enumerate(self.STD_LIKERT_5_POINT_LABELS)
        }
        self.TRANSP_IMPORTANCE_LIKERT_MAP = {
            label: i + 1 for i, label in enumerate(self.TRANSP_IMPORTANCE_LIKERT_LABELS)
        }
        self.COLUMN_MAPPING = {
            key.strip(): value for key, value in self.COLUMN_MAPPING_DEFINITION.items()
        }


# --- Utility Functions ---
def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def dataframe_column_sort_key(col_name: str) -> Tuple[Union[int, float], int, str, str]:
    if col_name == "respondent_id":
        return (-2, 0, "", col_name)
    if col_name == "submission_id":
        return (-1, 0, "", col_name)

    match = re.match(r"(q\d+)(?:_([a-zA-Z0-9_]+))?", col_name)
    if match:
        q_num = int(match.group(1)[1:])
        suffix_str = match.group(2) if match.group(2) else ""
        suffix_order = 99
        if not suffix_str:
            suffix_order = 0
        return (q_num, suffix_order, suffix_str, col_name)
    return (float("inf"), 0, col_name, col_name)


def _md_escape_pipe(
    text: Any,
) -> str:
    text_str = str(text) if not isinstance(text, str) else text
    return text_str.replace("|", r"\|").replace("\n", " ").replace("\xa0", " ")


# --- LLM Service Class ---
class LLMService:
    def __init__(self, api_key: Optional[str], model_name: str, placeholder_key: str):
        self.model_name: str = model_name
        self.client: Optional[OpenAI] = None
        self.cache: Dict[Tuple[str, str, str], str] = {}
        self.stats: Dict[str, int] = {
            "api_calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "skipped_no_client": 0,
            "categories_created_dynamic": 0,
        }
        self.categorization_log: List[LLMCategorizationLogEntry] = []

        if api_key and api_key != placeholder_key:
            try:
                self.client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1", api_key=api_key
                )
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        else:
            print("Warning: LLM API Key not set or placeholder. LLM features disabled.")

    def is_client_available(self) -> bool:
        return self.client is not None

    def call_llm(
        self,
        prompt: str,
        system_message: str = "You are a helpful data analysis assistant.",
        max_retries: int = 3,
        delay: int = 2,
    ) -> str:
        if not self.client:
            self.stats["skipped_no_client"] += 1
            return LLM_SKIPPED_MARKER

        cache_key = (self.model_name, system_message, prompt)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]

        self.stats["api_calls"] += 1
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )
                if (
                    chat_completion.choices
                    and chat_completion.choices[0].message
                    and chat_completion.choices[0].message.content
                ):
                    response_text = chat_completion.choices[0].message.content.strip()
                    self.cache[cache_key] = response_text
                    time.sleep(0.5)
                    return response_text
                raise ValueError(
                    "LLM response structure unexpected or content is None."
                )
            except Exception as e:
                print(
                    f"LLM API call warning (Attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    self.stats["errors"] += 1
                    self.cache[cache_key] = LLM_ERROR_MARKER
                    return LLM_ERROR_MARKER
        return LLM_ERROR_MARKER

    def add_log_entry(
        self,
        question_id: str,
        original_text: str,
        raw_llm_output: str,
        final_categories: Union[str, List[str]],
        status: str,
    ) -> None:
        self.categorization_log.append(
            {
                "question_id": question_id,
                "original_text": original_text,
                "raw_llm_output": raw_llm_output,
                "final_categories": final_categories,
                "status": status,
            }
        )


# --- Report Manager Class ---
class ReportManager:
    def __init__(self, script_start_time: datetime.datetime):
        self.entries: List[str] = []
        self.script_start_time: datetime.datetime = script_start_time
        self.llm_log_for_report: List[LLMCategorizationLogEntry] = []
        self.add_entry("# Data Preparation Script Report")
        self.add_entry(
            f"Execution Start Time: {self.script_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def add_entry(self, entry: str) -> None:
        self.entries.append(entry)

    def add_separator(self) -> None:
        self.add_entry("---")

    def add_section_heading(self, title: str) -> None:
        self.add_entry(
            f"\n## Section {len([e for e in self.entries if '## Section' in e]) + 1}: {title}"
        )

    def add_llm_categorization_log(self, log: List[LLMCategorizationLogEntry]) -> None:
        self.llm_log_for_report = log

    def _generate_llm_log_table(self) -> str:
        if not self.llm_log_for_report:
            return "No LLM categorizations performed or logged."

        table_md = [
            "| Question ID | Original Text (Truncated) | Raw LLM Output | Final Category/ies | Status |",
            "|---|---|---|---|---|",
        ]
        for log_entry in self.llm_log_for_report:
            orig_text = _md_escape_pipe(log_entry["original_text"])
            orig_text_trunc = (
                orig_text[:50] + "..." if len(orig_text) > 50 else orig_text
            )

            raw_output = _md_escape_pipe(log_entry["raw_llm_output"])
            raw_output_trunc = (
                raw_output[:50] + "..." if len(raw_output) > 50 else raw_output
            )

            final_cat = log_entry["final_categories"]
            if isinstance(final_cat, list):
                final_cat_str = _md_escape_pipe(
                    LLM_MULTI_OUTPUT_JOIN_DELIMITER.join(final_cat)
                )
            elif final_cat == "":
                final_cat_str = "(Cleared)"
            else:
                final_cat_str = _md_escape_pipe(str(final_cat))

            table_md.append(
                f"| {log_entry['question_id']} | {orig_text_trunc} | {raw_output_trunc} | {final_cat_str} | {log_entry['status']} |"
            )
        return "\n".join(table_md)

    def generate_final_report_text(self) -> str:
        llm_summary_idx = -1
        for i, entry_text in enumerate(self.entries):
            if "LLM Processing Summary" in entry_text:
                llm_summary_idx = i
                break

        if self.llm_log_for_report and llm_summary_idx != -1:
            llm_table_section_heading = "\n## LLM Categorization Details"
            llm_table_content = self._generate_llm_log_table()
            entries_before_summary = self.entries[:llm_summary_idx]
            entries_from_summary_onwards = self.entries[llm_summary_idx:]
            self.entries = (
                entries_before_summary
                + [llm_table_section_heading, llm_table_content]
                + entries_from_summary_onwards
            )

        script_end_time = datetime.datetime.now()
        final_entries = [
            "---",
            f"Execution End Time: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Execution Time: {script_end_time - self.script_start_time}",
        ]

        current_report = "\n\n".join(self.entries)
        final_report = current_report + "\n\n" + "\n\n".join(final_entries)
        return final_report

    def save_report_to_file(self, file_path: str) -> None:
        report_content = self.generate_final_report_text()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\nReport saved to: {file_path}")
        except Exception as e:
            print(f"Error saving report: {e}")


# --- LLM-based Categorization Logic ---
def llm_categorize_text(
    series: pd.Series,
    question_id: str,
    cat_list: List[str],
    item_desc: str,
    llm: LLMService,
    config: SurveyConfig,
    allow_multiple_categories: bool = True,
) -> pd.Series:
    unique_vals = series.dropna().unique()
    mapping: Dict[Any, Union[str, List[str]]] = {}

    generic_responses = ["Other", "Unclear/Comment"]
    valid_llm_responses_for_prompt = cat_list + generic_responses

    prompt_multi_category_instruction = f"- If it fits one or more categories, list them separated by a SEMICOLON ('{LLM_DEFAULT_DELIMITER}'), exactly as they are written. Prefer a single, most fitting category.\n"
    prompt_single_category_instruction = "- If it fits one or more categories, respond with ONLY the single, most fitting category.\n"
    prompt_category_instruction = (
        prompt_multi_category_instruction
        if allow_multiple_categories
        else prompt_single_category_instruction
    )

    prompt_template = (
        f"Categorize the following '{item_desc}': '{{val}}'.\n"
        f"Choose from these valid categories: {', '.join(cat_list)}.\n"
        f"{prompt_category_instruction}"
        f"- If it doesn't fit any specific category but is a valid response for '{item_desc}', respond with 'Other'.\n"
        f"- If the input is unclear, a comment, a refusal to answer, or expresses no specific {item_desc}, respond with 'Unclear/Comment'.\n"
        f"- If the input indicates not specified, unknown, or N/A for '{item_desc}', or is otherwise uninformative, respond with 'Unclear/Comment'.\n"
        f"Respond with the category name(s) ONLY. No explanations."
    )

    for val_orig in unique_vals:
        val_s = str(val_orig).strip()
        status_for_log: str
        processed_llm_categories: Union[str, List[str]] = ""
        raw_llm_resp: str = ""

        if (
            not val_s
            or val_s.lower() in config.STANDARDIZED_NULL_LIKE_ENTRIES_LOWERCASE
        ):
            raw_llm_resp = "Pre-emptive: Cleared as null-like"
            chosen_categories = [LLM_UNCLEAR_MARKER]
            status_for_log = "Cleared (Original was null-like)"
        else:
            raw_llm_resp = llm.call_llm(prompt_template.format(val=val_s))

            if raw_llm_resp == LLM_SKIPPED_MARKER:
                chosen_categories = [LLM_SKIPPED_MARKER]
                status_for_log = "Skipped (No LLM Client)"
            elif raw_llm_resp == LLM_ERROR_MARKER:
                chosen_categories = [LLM_ERROR_MARKER]
                status_for_log = "LLM Error"
            elif raw_llm_resp == "Unclear/Comment":
                chosen_categories = [LLM_UNCLEAR_MARKER]
                status_for_log = "Unclear/Comment from LLM"
            else:
                status_for_log = "Categorized by LLM"
                llm_response_parts_to_parse: List[str]

                if not allow_multiple_categories:
                    first_part = raw_llm_resp.split(LLM_DEFAULT_DELIMITER)[0].strip()
                    if LLM_DEFAULT_DELIMITER in raw_llm_resp and first_part:
                        status_for_log = "Categorized by LLM (forced single)"
                    llm_response_parts_to_parse = [first_part] if first_part else []
                else:
                    llm_response_parts_to_parse = [
                        c.strip()
                        for c in raw_llm_resp.split(LLM_DEFAULT_DELIMITER)
                        if c.strip()
                    ]

                returned_cats_raw = llm_response_parts_to_parse

                chosen_categories = []
                valid_llm_resp_lower = {
                    resp.lower() for resp in valid_llm_responses_for_prompt
                }
                canon_map = {
                    resp.lower(): resp for resp in valid_llm_responses_for_prompt
                }

                for r_cat_raw in returned_cats_raw:
                    r_cat_lower = r_cat_raw.lower()
                    if r_cat_lower in valid_llm_resp_lower:
                        chosen_categories.append(canon_map[r_cat_lower])
                    else:
                        chosen_categories.append("Other")
                        llm.stats["errors"] = llm.stats.get("errors", 0) + 1
                        llm.stats["llm_hallucinations_to_other"] = (
                            llm.stats.get("llm_hallucinations_to_other", 0) + 1
                        )
                        if status_for_log == "Categorized by LLM":
                            status_for_log = "Mapped to Other (unrecognized LLM)"
                        elif status_for_log == "Categorized by LLM (forced single)":
                            status_for_log = (
                                "Forced single, then Other (unrecognized LLM)"
                            )

                if (
                    not chosen_categories and returned_cats_raw
                ):  # If LLM responded but nothing was valid
                    chosen_categories = ["Other"]
                    if status_for_log == "Categorized by LLM":
                        status_for_log = (
                            "Mapped to Other (no valid cats in LLM response)"
                        )
                    elif status_for_log == "Categorized by LLM (forced single)":
                        status_for_log = "Forced single, then Other (no valid cats)"
                elif (
                    not returned_cats_raw
                ):  # If LLM response was empty or just delimiters
                    chosen_categories = [
                        "Other"
                    ]  # Default to Other if LLM provides no parsable categories
                    status_for_log = "Mapped to Other (empty/invalid LLM output)"

        if LLM_UNCLEAR_MARKER in chosen_categories:
            mapping[val_orig] = LLM_UNCLEAR_MARKER
            processed_llm_categories = ""

            if status_for_log not in [
                "Cleared (Original was null-like)",
                "Unclear/Comment from LLM",
            ]:
                status_for_log = "Unclear/Removed by LLM based on processing"

        elif LLM_ERROR_MARKER in chosen_categories:
            mapping[val_orig] = "LLM Processing Error"
            processed_llm_categories = "LLM Processing Error"
        elif LLM_SKIPPED_MARKER in chosen_categories:
            mapping[val_orig] = "LLM Skipped"
            processed_llm_categories = "LLM Skipped"
        else:
            final_chosen_cats = [
                cat for cat in chosen_categories if cat != "Not Specified"
            ]
            if not final_chosen_cats and chosen_categories:
                mapping[val_orig] = LLM_UNCLEAR_MARKER
                processed_llm_categories = ""
                if status_for_log not in [
                    "Cleared (Original was null-like)",
                    "Unclear/Comment from LLM",
                ]:
                    status_for_log = "Considered Unclear after category processing"
            elif not final_chosen_cats:
                mapping[val_orig] = LLM_UNCLEAR_MARKER
                processed_llm_categories = ""
                if status_for_log not in [
                    "Cleared (Original was null-like)",
                    "Unclear/Comment from LLM",
                ]:
                    status_for_log = "Mapped to Unclear (no categories found/valid)"
            else:
                mapping[val_orig] = final_chosen_cats
                processed_llm_categories = final_chosen_cats

        llm.add_log_entry(
            question_id=question_id,
            original_text=val_s,
            raw_llm_output=raw_llm_resp,
            final_categories=processed_llm_categories,
            status=status_for_log,
        )

    final_mapping_for_series: Dict[Any, str] = {}
    for k, v_list_or_str_marker in mapping.items():
        if isinstance(v_list_or_str_marker, list):
            final_mapping_for_series[k] = LLM_MULTI_OUTPUT_JOIN_DELIMITER.join(
                v_list_or_str_marker
            )
        else:
            final_mapping_for_series[k] = v_list_or_str_marker

    return series.map(final_mapping_for_series).fillna("")


def llm_categorize_dynamic_other(
    series_other_text: pd.Series,
    mc_q_id: str,
    std_opts_for_q: List[str],
    item_desc: str,
    llm: LLMService,
    config: SurveyConfig,
) -> pd.Series:
    unique_texts = series_other_text.dropna()[
        series_other_text.str.strip() != ""
    ].unique()
    mapping: Dict[str, Union[str, List[str]]] = {}

    newly_identified_categories_for_this_question: List[str] = []

    opts_for_prompt = [opt for opt in std_opts_for_q if opt.lower() != "other"]

    for val_orig in unique_texts:
        val_clean = str(val_orig).strip()
        raw_llm_resp_for_log: str
        final_cats_for_map: Union[str, List[str]]
        status_for_log: str

        if (
            not val_clean
            or val_clean.lower() in config.STANDARDIZED_NULL_LIKE_ENTRIES_LOWERCASE
        ):
            raw_llm_resp_for_log = "Pre-emptive: Cleared as null-like"
            final_cats_for_map = LLM_UNCLEAR_MARKER
            status_for_log = "Cleared (Original was null-like)"
        else:
            prompt_parts = [
                f"The 'Other' response for a question about '{item_desc}' is: '{val_clean}'.",
                f"Review '{val_clean}' carefully. Your goal is to map it or define a new category.",
                "Response options:",
            ]
            if opts_for_prompt:
                prompt_parts.append(
                    f"1. If it clearly maps to one of these standard options, state that option's name: {', '.join(opts_for_prompt)}."
                )
            if newly_identified_categories_for_this_question:
                prompt_parts.append(
                    f"2. If it clearly maps to one of these previously identified new categories for this question, state that category's name: {', '.join(newly_identified_categories_for_this_question)}."
                )
            prompt_parts.extend(
                [
                    f"3. If it represents one or more distinct, new '{item_desc}'s not covered above, provide concise, plural noun phrases for these new categories, separated by a SEMICOLON ('{LLM_DEFAULT_DELIMITER}'). Prefer a single new category if the text implies a single concept.",
                    f"4. If '{val_clean}' is vague, a non-answer, a comment, an opinion, or simply restates 'Other' without specifics, respond ONLY with 'Unclear/Comment'.",
                    "Respond with the chosen category name(s) or new category phrase(s) ONLY. No explanations.",
                ]
            )
            prompt = "\n".join(prompt_parts)
            raw_llm_resp_for_log = llm.call_llm(prompt)

            if raw_llm_resp_for_log == LLM_SKIPPED_MARKER:
                final_cats_for_map = LLM_SKIPPED_MARKER
                status_for_log = "Skipped (No LLM Client)"
            elif raw_llm_resp_for_log == LLM_ERROR_MARKER:
                final_cats_for_map = LLM_ERROR_MARKER
                status_for_log = "LLM Error"
            elif raw_llm_resp_for_log == "Unclear/Comment":
                final_cats_for_map = LLM_UNCLEAR_MARKER
                status_for_log = "Unclear/Comment from LLM"
            else:
                potential_cats_from_llm = [
                    c.strip()
                    for c in raw_llm_resp_for_log.split(LLM_DEFAULT_DELIMITER)
                    if c.strip()
                ]

                processed_cats_for_this_val: List[str] = []
                current_status_parts: List[str] = []

                all_known_categories_at_this_point = (
                    opts_for_prompt + newly_identified_categories_for_this_question
                )

                all_known_plus_other = all_known_categories_at_this_point + (
                    ["Other"] if "Other" in std_opts_for_q else []
                )

                all_known_lower = {k.lower(): k for k in all_known_plus_other}

                for p_cat in potential_cats_from_llm:
                    p_cat_lower = p_cat.lower()
                    if p_cat_lower in all_known_lower:
                        mapped_cat = all_known_lower[p_cat_lower]
                        processed_cats_for_this_val.append(mapped_cat)
                        current_status_parts.append(f"Mapped to '{mapped_cat}'")
                    else:  # New category suggested by LLM
                        is_truly_new_among_dynamic_cats = True
                        for (
                            existing_dyn_cat
                        ) in newly_identified_categories_for_this_question:
                            if p_cat.lower() == existing_dyn_cat.lower():
                                processed_cats_for_this_val.append(existing_dyn_cat)
                                current_status_parts.append(
                                    f"Mapped to new dynamic '{existing_dyn_cat}'"
                                )
                                is_truly_new_among_dynamic_cats = False
                                break

                        if is_truly_new_among_dynamic_cats:
                            if p_cat not in [
                                LLM_UNCLEAR_MARKER,
                                LLM_ERROR_MARKER,
                                LLM_SKIPPED_MARKER,
                            ]:
                                processed_cats_for_this_val.append(p_cat)
                                newly_identified_categories_for_this_question.append(
                                    p_cat
                                )
                                llm.stats["categories_created_dynamic"] += 1
                                current_status_parts.append(
                                    f"New category created: '{p_cat}'"
                                )
                            else:
                                current_status_parts.append(
                                    f"LLM suggested invalid category '{p_cat}', mapped to Other"
                                )
                                if "Other" in all_known_lower.values():
                                    processed_cats_for_this_val.append("Other")

                if not processed_cats_for_this_val:
                    if "Other" in std_opts_for_q:
                        final_cats_for_map = ["Other"]
                        status_for_log = "LLM Other Text - Parsed as generic Other"
                    else:
                        final_cats_for_map = LLM_UNCLEAR_MARKER
                        status_for_log = (
                            "LLM Other Text - Uncategorizable and no 'Other' option"
                        )

                else:
                    final_cats_for_map = processed_cats_for_this_val
                    status_for_log = (
                        "; ".join(current_status_parts)
                        if current_status_parts
                        else "Categorized by LLM (Other Text)"
                    )

        final_val_for_log: Union[str, List[str]]
        output_for_series_map: str

        if final_cats_for_map == LLM_UNCLEAR_MARKER:
            output_for_series_map = LLM_UNCLEAR_MARKER
            final_val_for_log = ""
        elif final_cats_for_map == LLM_ERROR_MARKER:
            output_for_series_map = "LLM Processing Error"
            final_val_for_log = "LLM Processing Error"
        elif final_cats_for_map == LLM_SKIPPED_MARKER:
            output_for_series_map = "LLM Skipped"
            final_val_for_log = "LLM Skipped"
        else:
            output_for_series_map = LLM_MULTI_OUTPUT_JOIN_DELIMITER.join(
                final_cats_for_map
            )
            final_val_for_log = final_cats_for_map

        mapping[val_orig] = output_for_series_map

        llm.add_log_entry(
            question_id=mc_q_id,
            original_text=val_clean,
            raw_llm_output=raw_llm_resp_for_log,
            final_categories=final_val_for_log,
            status=status_for_log,
        )

    return series_other_text.map(mapping).fillna("")


# --- Core Data Processing Functions ---
def _load_raw_data(file_path: str, reporter: ReportManager) -> Optional[pd.DataFrame]:
    reporter.add_section_heading(f"Load the Survey Data from '{file_path}'")
    try:
        df_raw = pd.read_excel(file_path, sheet_name="Sheet1")
        reporter.add_entry(
            f"Data loaded successfully. Initial shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns."
        )
        if df_raw.empty:
            reporter.add_entry(
                "Error: DataFrame is empty after loading. File found, but no data or loading issue."
            )
            return None
        return df_raw
    except FileNotFoundError:
        reporter.add_entry(f"Error: File {file_path} not found. Script terminated.")
        return None
    except Exception as e:
        reporter.add_entry(f"Error loading data: {e}. Script terminated.")
        return None


def _initialize_columns_and_question_info(
    df_raw: pd.DataFrame,
    config: SurveyConfig,
    reporter: ReportManager,
    q_info: QuestionTypeInfo,
) -> Tuple[pd.DataFrame, QuestionTypeInfo]:
    reporter.add_section_heading("Column Renaming & Info Structure Initialization")
    df = df_raw.copy()
    original_cols = df.columns.tolist()
    df.columns = [str(col).strip() if pd.notnull(col) else "" for col in df.columns]
    reporter.add_entry(
        f"DataFrame column names stripped of leading/trailing whitespace (if any changes made: {original_cols != df.columns.tolist()})."
    )

    # Rename columns based on the mapping
    cols_before_rename = set(df.columns)
    df.rename(columns=config.COLUMN_MAPPING, inplace=True)
    renamed_count = sum(
        1
        for old, new in config.COLUMN_MAPPING.items()
        if old in cols_before_rename and new in df.columns and old != new
    )
    reporter.add_entry(
        f"Applied column renaming based on `config.COLUMN_MAPPING`: {renamed_count} columns were successfully renamed."
    )

    # Drop unmapped
    intended_final_cols = set(config.COLUMN_MAPPING.values())
    cols_to_drop = [col for col in df.columns if col not in intended_final_cols]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        reporter.add_entry(
            f"Dropped {len(cols_to_drop)} unmapped columns (e.g., {', '.join(map(str, cols_to_drop[:3]))}{'...' if len(cols_to_drop) > 3 else ''})."
        )

    # Build question_type_info
    for orig_q_text_stripped, new_name in config.COLUMN_MAPPING.items():
        if new_name not in df.columns:
            continue
        entry: QuestionInfoDict = {"original_question_text": orig_q_text_stripped}

        text_question_prefixes = ("q1_", "q2_", "q3_", "q6_", "q7_", "q9_")
        if new_name.startswith(text_question_prefixes) or "tools_text" in new_name:
            entry["type"] = "Text"
            if "tools_text" in new_name:
                entry["category"] = "Tools Text"
        elif new_name in [
            "q4_experience_role",
            "q5_experience_company",
            "q8_company_size",
            "q10_company_revenue",
        ]:
            entry["type"] = "Multiple Choice (Single)"
            if new_name == "q3_department":
                entry["possible_values"] = config.DEPT_VALUES
            if new_name == "q4_experience_role":
                entry["possible_values"] = config.EXP_ROLE_VALUES
            elif new_name == "q5_experience_company":
                entry["possible_values"] = config.EXP_ROLE_VALUES
            elif new_name == "q8_company_size":
                entry["possible_values"] = config.COMPANY_SIZE_VALUES
            elif new_name == "q10_company_revenue":
                entry["possible_values"] = config.COMPANY_REVENUE_VALUES
            if "Other" in (entry.get("possible_values", []) or []):  # type: ignore
                entry["has_other_specify"] = True
        elif "_mc" in new_name:
            entry["type"] = "Multiple Choice (Multi)"
            if new_name in config.MC_STANDARD_OPTIONS_MAP:
                entry["possible_values"] = config.MC_STANDARD_OPTIONS_MAP[new_name]
            if "Other" in (entry.get("possible_values", []) or []):  # type: ignore
                entry["has_other_specify"] = True
        elif new_name in [
            "q78_barriers_ai_transparency_importance_maint",
            "q79_barriers_ai_transparency_importance_qual",
            "q80_barriers_ai_transparency_importance_prod",
        ]:
            entry["type"] = "Likert"
            entry["possible_values"] = config.TRANSP_IMPORTANCE_LIKERT_LABELS
            entry["value_map_to_numeric"] = config.TRANSP_IMPORTANCE_LIKERT_MAP
        else:
            entry["type"] = "Likert"
            entry["possible_values"] = config.STD_LIKERT_5_POINT_LABELS
            entry["value_map_to_numeric"] = config.STD_LIKERT_5_POINT_MAP

        q_info[new_name] = entry

    reporter.add_entry(
        f"`question_type_info` dictionary created with {len(q_info)} entries for the processed columns."
    )
    counts_by_type = Counter(
        str(info.get("type", "Unknown")) for info in q_info.values()
    )
    for q_type, count in counts_by_type.items():
        reporter.add_entry(f"  - Type '{q_type}': {count} columns")
    return df, q_info


def _anonymize_data(
    df: pd.DataFrame, reporter: ReportManager, q_info: QuestionTypeInfo
) -> Tuple[pd.DataFrame, QuestionTypeInfo]:
    reporter.add_section_heading("Anonymization Steps")
    # Add respondent_id
    df.insert(0, "respondent_id", range(1, len(df) + 1))
    q_info["respondent_id"] = {
        "original_question_text": "Generated Respondent ID",
        "type": "Identifier",
    }
    reporter.add_entry("Added 'respondent_id' column as the first column.")

    # Drop PII columns
    pii_cols_to_drop = ["q1_full_name"]
    actual_pii_dropped = []
    for col_to_drop in pii_cols_to_drop:
        if col_to_drop in df.columns:
            df.drop(columns=[col_to_drop], inplace=True)
            if col_to_drop in q_info:
                del q_info[col_to_drop]
            actual_pii_dropped.append(col_to_drop)
    if actual_pii_dropped:
        reporter.add_entry(
            f"Dropped PII columns for anonymity: {', '.join(actual_pii_dropped)}."
        )
    else:
        reporter.add_entry(
            "No predefined PII columns (e.g., q1_full_name) found to drop."
        )

    if "q6_company_name" in df.columns:
        reporter.add_entry(
            "Note: 'q6_company_name' is present in the data. Consider anonymization or careful handling if this data is sensitive for reporting."
        )
    return df, q_info


def _preprocess_data_integrity_and_types(
    df: pd.DataFrame,
    reporter: ReportManager,
    q_info: QuestionTypeInfo,
    config: SurveyConfig,
) -> pd.DataFrame:
    reporter.add_section_heading(
        "Initial Data Integrity Checks (NaNs) and Type Coercion"
    )
    initial_nans = df.isnull().sum().sum()
    reporter.add_entry(f"Total NaN values before this step: {initial_nans}")

    # NaN Handling based on inferred question type from q_info
    for col, info in q_info.items():
        if col not in df.columns:
            continue

        col_type = str(info.get("type", "Unknown"))

        if col_type == "Text" or info.get("category") == "Tools Text":
            # For pure text or tools text, fill NaN with '' (blank)
            df[col] = df[col].fillna("")
        elif col_type == "Multiple Choice (Single)":
            # For single-select MC, fill NaN with '' (blank)
            df[col] = df[col].fillna("")
        elif col_type == "Multiple Choice (Multi)":
            # For multi-select MC, fill NaN with empty string (common for no selection)
            df[col] = df[col].fillna("")
        # Likert NaNs are generally preserved at this stage; they might indicate non-response.
        # Numeric conversion later will handle them or errors if non-numeric.

    reporter.add_entry(
        "Applied type-specific NaN filling: '' (blank) for Text & MC-Single & MC-Multi. Likert NaNs preserved for now."
    )

    # Type Verification and Likert Formatting
    likert_coercion_warnings: List[str] = []
    for col, info in q_info.items():
        if col not in df.columns:
            continue
        col_type = str(info.get("type", "Unknown"))

        if col_type == "Likert":
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                value_map = info.get("value_map_to_numeric")
                if isinstance(value_map, dict):
                    df[col] = (
                        df[col].astype(str).str.strip().map(value_map).fillna(df[col])
                    )

            # Attempt to convert to numeric, coercing errors to NaN
            original_nan_count = df[col].isnull().sum()
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                new_nan_count = df[col].isnull().sum()
                if new_nan_count > original_nan_count:
                    likert_coercion_warnings.append(
                        f"Coercion to numeric for Likert column '{col}' introduced {new_nan_count - original_nan_count} new NaN(s) "
                        f"(values could not be converted to numbers)."
                    )
        elif (
            col_type
            in [
                "Text",
                "Multiple Choice (Single)",
                "Multiple Choice (Multi)",
            ]
            or info.get("category") == "Tools Text"
        ):
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str)

    reporter.add_entry(
        "Ensured Likert columns are numeric (text labels mapped if defined, then coerced). Text/MC columns ensured string type initially."
    )
    if likert_coercion_warnings:
        for warn_msg in likert_coercion_warnings:
            reporter.add_entry(f"  Warning: {warn_msg}")

    current_nans = df.isnull().sum().sum()
    reporter.add_entry(f"Total NaN values after this step: {current_nans}.")
    return df


def _validated_ss_mc_responses(
    df: pd.DataFrame,
    reporter: ReportManager,
    q_info: QuestionTypeInfo,
) -> pd.DataFrame:
    reporter.add_section_heading("Validating Single-Select MC Responses")
    mc_single_unexpected_warnings: List[str] = []
    for col, info in q_info.items():
        if col not in df.columns or info.get("type") != "Multiple Choice (Single)":
            continue
        df[col] = df[col].astype(str).str.strip()

        has_other_specify = info.get("has_other_specify", False)
        possible_vals_list = info.get("possible_values")
        if not isinstance(possible_vals_list, list):
            possible_vals_list = []

        if not has_other_specify:
            for val in df[col]:
                val_str = str(val).strip()
                if val_str not in possible_vals_list and val_str != "":
                    mc_single_unexpected_warnings.append(
                        f"MC-Single '{col}' contains unexpected value: {val_str} (and is not blank)."
                    )

    reporter.add_entry("Processed MC-Single columns: validated responses.")
    if mc_single_unexpected_warnings:
        for warn_msg in mc_single_unexpected_warnings:
            reporter.add_entry(f"  Note: {warn_msg}")
    return df


def _process_multiselect_columns(
    df: pd.DataFrame,
    reporter: ReportManager,
    q_info: QuestionTypeInfo,
    config: SurveyConfig,
) -> Tuple[pd.DataFrame, QuestionTypeInfo]:
    reporter.add_section_heading(
        "Processing Free-Text 'Tools' and Multi-Select Questions (Initial Parsing)"
    )
    multi_select_cols = [
        c
        for c, i in q_info.items()
        if i.get("type") == "Multiple Choice (Multi)" and c in df.columns
    ]
    for col_name in multi_select_cols:
        info_dict = q_info[col_name]
        df[col_name] = df[col_name].astype(str)

        parsed_options_master_list: List[List[str]] = []
        other_text_master_list: List[str] = []

        pv_raw = info_dict.get("possible_values", [])
        possible_values_this_q = pv_raw if isinstance(pv_raw, list) else []

        std_opts_no_other_canon = [
            v for v in possible_values_this_q if v.lower() != "other"
        ]
        std_opts_no_other_lower_set = {v.lower() for v in std_opts_no_other_canon}
        lower_to_canon_map = {v.lower(): v for v in std_opts_no_other_canon}

        for original_entry_str in df[col_name]:
            current_selected_std_options: Set[str] = set()
            current_collected_other_texts: List[str] = []

            if not original_entry_str.strip():
                parsed_options_master_list.append([])
                other_text_master_list.append("")
                continue

            raw_choices_from_entry = [
                opt.strip()
                for opt in original_entry_str.split(config.MULTI_SELECT_DELIMITER)
                if opt.strip()
            ]

            for choice_original_case in raw_choices_from_entry:
                choice_lower = choice_original_case.lower()

                if choice_lower in std_opts_no_other_lower_set:
                    current_selected_std_options.add(lower_to_canon_map[choice_lower])
                elif (
                    choice_lower not in config.STANDARDIZED_NULL_LIKE_ENTRIES_LOWERCASE
                ):
                    current_collected_other_texts.append(choice_original_case)

            parsed_options_master_list.append(
                sorted(list(current_selected_std_options))
            )
            other_text_master_list.append(
                LLM_MULTI_OUTPUT_JOIN_DELIMITER.join(
                    sorted(list(set(current_collected_other_texts)))
                )
            )

        # Create intermediate _list column
        list_col_name = f"{col_name}_list"
        df[list_col_name] = parsed_options_master_list
        q_info[list_col_name] = {
            "original_question_text": f"Parsed standard options from: {info_dict.get('original_question_text', col_name)}",
            "type": "Processed List (from MC-Multi)",
            "parent_question": col_name,
            "possible_values": possible_values_this_q,
        }

        # Create intermediate _other_text column if "Other" is an option typically
        create_other_text_col = info_dict.get("has_other_specify", False) or (
            "Other" in possible_values_this_q
        )
        if create_other_text_col:
            other_text_col_name = f"{col_name}_other_text"
            df[other_text_col_name] = other_text_master_list
            q_info[other_text_col_name] = {
                "original_question_text": f"Custom 'Other' text from: {info_dict.get('original_question_text', col_name)}",
                "type": "Text (Other Specify)",
                "parent_question": col_name,
            }

    if multi_select_cols:
        reporter.add_entry(
            f"Processed {len(multi_select_cols)} multi-select questions: created intermediate `_list` and `_other_text` columns for further processing."
        )
    return df, q_info


def _apply_llm_categorization_tasks(
    df: pd.DataFrame,
    llm: LLMService,
    reporter: ReportManager,
    q_info: QuestionTypeInfo,
    config: SurveyConfig,
) -> Tuple[pd.DataFrame, QuestionTypeInfo]:
    reporter.add_section_heading("LLM-based Categorization of Text Responses")
    if not llm.is_client_available():
        reporter.add_entry(
            "LLM client not available. Skipping all LLM categorization tasks."
        )
        return df, q_info

    cols_with_potential_unclear_marker: List[str] = []
    columns_to_drop_after_llm: List[str] = []
    processed_parent_mc_cols_for_combination: Dict[str, pd.Series] = {}

    # --- LLM Categorization for standard Text questions (e.g., position, industry, country) ---
    text_q_map_for_llm = {
        "q2_position": (config.POSITION_CATEGORIES, "job position"),
        "q3_department": (config.DEPT_VALUES, "department"),
        "q7_industry_sector": (config.INDUSTRY_SECTOR_CATEGORIES, "industry sector"),
        "q9_company_hq_country": (
            config.COUNTRY_CATEGORIES,
            "country name",
        ),
    }
    for col_id, (categories, item_description) in text_q_map_for_llm.items():
        if col_id in df.columns and q_info.get(col_id, {}).get("type") == "Text":
            allow_multiple = False

            df[col_id] = llm_categorize_text(
                df[col_id],
                col_id,
                categories,
                item_description,
                llm,
                config,
                allow_multiple_categories=allow_multiple,
            )

            q_info_entry = q_info.get(col_id, {})
            q_info_entry["original_question_text"] = (
                f"LLM Categorized: {q_info_entry.get('original_question_text', col_id)}"
            )
            q_info_entry["possible_values_llm"] = categories + [
                "Other",
                LLM_UNCLEAR_MARKER,
                "LLM Processing Error",
                "LLM Skipped",
            ]
            q_info_entry["is_llm_categorized"] = True

            reporter.add_entry(
                f"  - LLM categorized and updated Text column '{col_id}' (Allow multiple: {allow_multiple})."
            )
            cols_with_potential_unclear_marker.append(col_id)

    # --- LLM Categorization for free-text 'Tools' questions ---
    tools_text_q_map_for_llm = {
        "q36_maint_da_tools_text": (
            config.DA_TOOL_CATEGORIES,
            "maintenance data analysis tool",
        ),
        "q49_qual_da_tools_text": (
            config.DA_TOOL_CATEGORIES,
            "quality data analysis tool",
        ),
        "q62_prod_da_tools_text": (
            config.DA_TOOL_CATEGORIES,
            "production data analysis tool",
        ),
    }
    for col_id, (categories, item_description) in tools_text_q_map_for_llm.items():
        if (
            col_id in df.columns
            and q_info.get(col_id, {}).get("category") == "Tools Text"
        ):
            df[col_id] = llm_categorize_text(
                df[col_id], col_id, categories, item_description, llm, config
            )

            q_info_entry = q_info.get(col_id, {})
            q_info_entry["original_question_text"] = (
                f"LLM Categorized Tools Text: {q_info_entry.get('original_question_text', col_id)}"
            )
            q_info_entry["possible_values_llm"] = categories + [
                "Other",
                LLM_UNCLEAR_MARKER,
                "LLM Processing Error",
                "LLM Skipped",
            ]
            q_info_entry["is_llm_categorized"] = True

            reporter.add_entry(
                f"  - LLM categorized and updated Tools Text column '{col_id}'."
            )
            cols_with_potential_unclear_marker.append(col_id)

    # --- LLM Categorization for 'Other' text derived from Multi-Select questions ---
    for other_text_col_name in list(q_info.keys()):
        info_dict = q_info.get(other_text_col_name, {})
        if (
            info_dict.get("type") == "Text (Other Specify)"
            and other_text_col_name in df.columns
        ):
            parent_mc_q_name = str(info_dict.get("parent_question", ""))

            if parent_mc_q_name and parent_mc_q_name in config.MC_ITEM_DESCRIPTION_MAP:
                parent_q_info = q_info.get(parent_mc_q_name, {})
                std_options_for_parent_q_raw = parent_q_info.get("possible_values", [])
                std_options_for_parent_q = (
                    std_options_for_parent_q_raw
                    if isinstance(std_options_for_parent_q_raw, list)
                    else []
                )

                item_desc_for_parent_q = config.MC_ITEM_DESCRIPTION_MAP[
                    parent_mc_q_name
                ]

                # Store the LLM categorized 'other' text series for later combination
                temp_llm_other_series = llm_categorize_dynamic_other(
                    df[other_text_col_name],
                    parent_mc_q_name,
                    std_options_for_parent_q,
                    item_desc_for_parent_q,
                    llm,
                    config,
                )
                processed_parent_mc_cols_for_combination[parent_mc_q_name] = (
                    temp_llm_other_series
                )

                reporter.add_entry(
                    f"  - LLM dynamically categorized 'Other' text from '{other_text_col_name}' (for parent '{parent_mc_q_name}') for later combination."
                )

    # After the loop, combine standard options with LLM-categorized 'other' options for MC questions
    for (
        parent_mc_col,
        temp_llm_other_series,
    ) in processed_parent_mc_cols_for_combination.items():
        standard_options_list_col = f"{parent_mc_col}_list"
        other_text_col = f"{parent_mc_col}_other_text"

        temp_apply_col_name = f"{parent_mc_col}_llm_other_temp_for_apply"
        df[temp_apply_col_name] = temp_llm_other_series

        def combine_mc_options(
            row_series: pd.Series, std_list_col: str, llm_temp_col: str
        ) -> List[str]:
            std_opts = (
                row_series[std_list_col]
                if std_list_col in row_series
                and isinstance(row_series[std_list_col], list)
                else []
            )
            llm_other_str = row_series[llm_temp_col]

            llm_other_opts_processed = []
            if isinstance(llm_other_str, str) and llm_other_str:
                # Split the string from LLM output
                raw_split_opts = [
                    opt.strip()
                    for opt in llm_other_str.split(LLM_MULTI_OUTPUT_JOIN_DELIMITER)
                    if opt.strip()
                ]
                # Filter out any markers or non-category strings
                for opt in raw_split_opts:
                    if opt not in [
                        LLM_UNCLEAR_MARKER,
                        LLM_ERROR_MARKER,
                        LLM_SKIPPED_MARKER,
                        "LLM Processing Error",
                        "LLM Skipped",
                    ]:
                        llm_other_opts_processed.append(opt)

            combined_options = std_opts + llm_other_opts_processed
            return sorted(list(set(combined_options)))

        df[parent_mc_col] = df.apply(
            combine_mc_options,
            args=(standard_options_list_col, temp_apply_col_name),
            axis=1,
        )

        q_info_entry = q_info.get(parent_mc_col, {})
        q_info_entry["original_question_text"] = (
            f"Consolidated Multi-Select: {q_info_entry.get('original_question_text', parent_mc_col)}"
        )
        q_info_entry["type"] = "Multi-Select"
        q_info_entry["is_llm_processed_mc"] = True

        reporter.add_entry(
            f"  - Combined standard and LLM-categorized 'Other' options into '{parent_mc_col}'. Final value is a Python list."
        )

        # Mark intermediate columns for deletion
        if standard_options_list_col in df.columns:
            columns_to_drop_after_llm.append(standard_options_list_col)
            if standard_options_list_col in q_info:
                del q_info[standard_options_list_col]
        if other_text_col in df.columns:
            columns_to_drop_after_llm.append(other_text_col)
            if other_text_col in q_info:
                del q_info[other_text_col]
        if temp_apply_col_name in df.columns:
            columns_to_drop_after_llm.append(temp_apply_col_name)

    # Drop the collected intermediate columns
    if columns_to_drop_after_llm:
        unique_cols_to_drop = list(set(columns_to_drop_after_llm))
        df.drop(columns=unique_cols_to_drop, inplace=True, errors="ignore")
        reporter.add_entry(
            f"  - Dropped intermediate multi-select helper columns: {', '.join(unique_cols_to_drop)}."
        )

    # Replace any instance of LLM_UNCLEAR_MARKER with empty string in text-categorized columns
    unclear_replacement_count = 0
    for col_to_clean in cols_with_potential_unclear_marker:
        if col_to_clean in df.columns and df[col_to_clean].dtype == "object":
            original_sum_is_marker = (df[col_to_clean] == LLM_UNCLEAR_MARKER).sum()
            if original_sum_is_marker > 0:
                df[col_to_clean] = df[col_to_clean].replace(LLM_UNCLEAR_MARKER, "")
                unclear_replacement_count += original_sum_is_marker
    if unclear_replacement_count > 0:
        reporter.add_entry(
            f"Replaced {unclear_replacement_count} instances of internal '{LLM_UNCLEAR_MARKER}' with empty strings in LLM-categorized text columns."
        )

    # Update q_info for LLM-categorized columns to reflect that LLM_UNCLEAR_MARKER became ""
    for col_id_cleaned in cols_with_potential_unclear_marker:
        if col_id_cleaned in q_info:
            info_entry = q_info[col_id_cleaned]
            if (
                info_entry.get("is_llm_categorized")
                and "possible_values_llm" in info_entry
            ):
                current_possible_values = info_entry["possible_values_llm"]
                if isinstance(current_possible_values, list):
                    new_possible_values = [
                        val
                        for val in current_possible_values
                        if val != LLM_UNCLEAR_MARKER
                    ]
                    if "" not in new_possible_values:
                        new_possible_values.append("")
                    info_entry["possible_values_llm"] = new_possible_values

    reporter.add_llm_categorization_log(llm.categorization_log)
    return df, q_info


def _finalize_dataframe(df: pd.DataFrame, reporter: ReportManager) -> pd.DataFrame:
    reporter.add_section_heading("Final DataFrame Structure and NaN Summary")
    try:
        df.columns = df.columns.astype(str)
        df_sorted = df[sorted(df.columns, key=dataframe_column_sort_key)]
    except Exception as e:
        reporter.add_entry(
            f"Note: Could not sort columns with custom key due to error: {e}. Using default sort."
        )
        df_sorted = df[sorted(df.columns)]

    reporter.add_entry(
        f"Final DataFrame shape: {df_sorted.shape[0]} rows, {df_sorted.shape[1]} columns. Columns are sorted."
    )

    final_nans = df_sorted.isnull().sum().sum()
    cols_with_nans_count = df_sorted.isnull().any().sum()
    reporter.add_entry(
        f"Final missing values (NaNs): {final_nans} total NaNs across {cols_with_nans_count} columns."
    )
    if cols_with_nans_count > 0:
        missing_cols_summary = df_sorted.isnull().sum()
        cols_with_nans_list = missing_cols_summary[
            missing_cols_summary > 0
        ].index.tolist()
        reporter.add_entry(
            f"  Columns containing NaNs include (e.g.): {', '.join(cols_with_nans_list[:5])}{'...' if len(cols_with_nans_list) > 5 else ''}"
        )
        reporter.add_entry(
            "    (NaNs are often expected in Likert scales for non-responses or due to numeric coercion issues)."
        )
    return df_sorted


def _generate_column_analysis_guide_markdown(
    df: pd.DataFrame, q_info: QuestionTypeInfo, reporter: ReportManager
) -> None:
    reporter.add_section_heading("Data Structure and Column Analysis Guide")
    reporter.add_entry(
        "This table provides an overview of each column in the cleaned dataset, including its origin, type, data type in the DataFrame, and notes for analysis. Pipes `|` within content are escaped as `\\|`."
    )

    md_table_rows = [
        "| Column Name | Original Question / Description | Question Type (Processed) | Pandas Dtype | Possible Values / Scale Info | Notes & Parent Question |",
        "|---|---|---|---|---|---|",
    ]

    for col_name in df.columns:
        info = q_info.get(col_name, {})

        orig_q_text = _md_escape_pipe(str(info.get("original_question_text", "N/A")))
        processed_q_type = _md_escape_pipe(str(info.get("type", "Unknown")))
        pandas_dtype = _md_escape_pipe(str(df[col_name].dtype))

        pv_details_str = "-"
        possible_vals = info.get("possible_values")
        possible_vals_llm = info.get("possible_values_llm")
        value_map_num = info.get("value_map_to_numeric")

        is_llm_categorized_text = info.get("is_llm_categorized", False)
        is_llm_processed_mc = info.get("is_llm_processed_mc", False)

        if (
            processed_q_type == "Likert"
            and isinstance(possible_vals, list)
            and isinstance(value_map_num, dict)
        ):
            scale_desc = f"{len(possible_vals)}-pt: {', '.join(map(_md_escape_pipe, possible_vals))}. Mapped {min(value_map_num.values())}-{max(value_map_num.values())}."
            pv_details_str = scale_desc
        elif is_llm_categorized_text and isinstance(possible_vals_llm, list):
            example_cats_display = []
            for val_llm in possible_vals_llm:
                if val_llm == "":
                    example_cats_display.append("(Blank)")
                else:
                    example_cats_display.append(str(val_llm))

            example_cats = example_cats_display[:4]
            pv_details_str = f"LLM Cats (e.g.): {_md_escape_pipe(', '.join(example_cats))}{'...' if len(example_cats_display) > 4 else ''}"
        elif processed_q_type == "Multi-Select" or is_llm_processed_mc:
            pv_details_str = (
                "List of selected/categorized options (standard + dynamic from LLM)."
            )
            if isinstance(possible_vals, list) and possible_vals:
                pv_details_str += f" Std opts e.g.: {_md_escape_pipe(', '.join(map(str, possible_vals[:2])))}..."
        elif isinstance(possible_vals, list) and possible_vals:
            example_vals_display = []
            for val_item in possible_vals:
                if val_item == "":
                    example_vals_display.append("(Blank)")
                else:
                    example_vals_display.append(str(val_item))
            example_vals = example_vals_display[:3]

            pv_details_str = _md_escape_pipe(", ".join(example_vals)) + (
                "..." if len(possible_vals) > 3 else ""
            )

        analysis_notes = "Standard handling applied."
        if col_name == "respondent_id":
            analysis_notes = "Unique respondent identifier."
        elif processed_q_type == "Likert":
            analysis_notes = (
                "Ordinal data, numeric. NaNs indicate non-response or coercion error."
            )
        elif processed_q_type == "Multi-Select" or is_llm_processed_mc:
            analysis_notes = "Python list of strings. Includes standard selections and LLM-processed 'Other' entries. Explode for frequency."
        elif is_llm_categorized_text:
            analysis_notes = "Text content categorized by LLM. May contain multiple values joined by delimiter if applicable for question. Blank indicates cleared/unclear."

        parent_q = info.get("parent_question")
        if parent_q:
            analysis_notes += f" (Parent: {_md_escape_pipe(str(parent_q))})"

        md_table_rows.append(
            f"| {_md_escape_pipe(col_name)} | {orig_q_text} | {processed_q_type} | {pandas_dtype} | {pv_details_str} | {analysis_notes} |"
        )

    reporter.add_entry("\n".join(md_table_rows))


def _save_processed_data(
    df: pd.DataFrame,
    config: SurveyConfig,
    reporter: ReportManager,
    q_info: QuestionTypeInfo,
) -> None:
    reporter.add_section_heading("Saving Processed Data")
    ensure_dir(config.OUTPUT_BASE_DIR)

    excel_path = os.path.join(config.OUTPUT_BASE_DIR, "cleaned_survey_data.xlsx")
    csv_path = os.path.join(config.OUTPUT_BASE_DIR, "cleaned_survey_data.csv")
    pickle_path = os.path.join(config.OUTPUT_BASE_DIR, "cleaned_survey_data.pkl")
    json_defs_path = os.path.join(config.OUTPUT_BASE_DIR, "question_definitions.json")

    try:
        # For Excel, convert list[str] columns to delimited strings for compatibility
        df_excel = df.copy()
        for col in df_excel.columns:
            if df_excel[col].apply(lambda x: isinstance(x, list)).any():
                df_excel[col] = df_excel[col].apply(
                    lambda x: LLM_MULTI_OUTPUT_JOIN_DELIMITER.join(x)
                    if isinstance(x, list)
                    else x
                )

        df_excel.to_excel(excel_path, index=False, engine="openpyxl")
        reporter.add_entry(
            f"Cleaned data saved to Excel: '{excel_path}' (lists joined by '{LLM_MULTI_OUTPUT_JOIN_DELIMITER}')"
        )

        # For CSV, also join lists
        df_csv = df.copy()
        for col in df_csv.columns:
            if df_csv[col].apply(lambda x: isinstance(x, list)).any():
                df_csv[col] = df_csv[col].apply(
                    lambda x: LLM_MULTI_OUTPUT_JOIN_DELIMITER.join(x)
                    if isinstance(x, list)
                    else x
                )
        df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        reporter.add_entry(
            f"Cleaned data saved to CSV: '{csv_path}' (lists joined by '{LLM_MULTI_OUTPUT_JOIN_DELIMITER}')"
        )

        # Pickle saves Python objects directly, so lists are preserved.
        df.to_pickle(pickle_path)
        reporter.add_entry(
            f"Cleaned data saved to Pickle: '{pickle_path}' (lists preserved)"
        )

        question_definitions = []
        for col_name in df.columns:
            info = q_info.get(col_name, {})

            original_question_text_raw = str(info.get("original_question_text", "N/A"))
            original_question = original_question_text_raw.replace(
                " (LLM categorized)", ""
            )
            original_question = original_question.replace("LLM categorized", "").strip()

            temp_orig_q_for_desc = original_question
            if not temp_orig_q_for_desc or temp_orig_q_for_desc == "N/A":
                short_desc = "N/A"
            else:
                words = temp_orig_q_for_desc.split()
                if len(words) <= 4:
                    short_desc = " ".join(words)
                else:
                    short_desc = " ".join(words[:4]) + "..."

            question_type = str(info.get("type", "Unknown"))

            current_pvs = info.get("possible_values")
            if info.get("is_llm_categorized", False):
                llm_pvs = info.get("possible_values_llm")
                if llm_pvs is not None:
                    current_pvs = llm_pvs

            if current_pvs is not None and not isinstance(current_pvs, list):
                current_pvs = (
                    list(current_pvs)
                    if hasattr(current_pvs, "__iter__")
                    and not isinstance(current_pvs, str)
                    else None
                )

            likert_map_data = None
            if question_type == "Likert":
                likert_map_data = info.get("value_map_to_numeric")

            question_definitions.append(
                {
                    "column_name": col_name,
                    "short_description": short_desc,
                    "original_question": original_question,
                    "question_type": question_type,
                    "possible_values": current_pvs,
                    "likert_map": likert_map_data,
                }
            )

        import json  # Assuming json module is available.

        with open(json_defs_path, "w", encoding="utf-8") as f:
            json.dump(question_definitions, f, indent=4, ensure_ascii=False)
        reporter.add_entry(f"Question definitions saved to JSON: '{json_defs_path}'")

    except Exception as e:
        reporter.add_entry(
            f"Error during saving of processed data or question definitions: {e}"
        )


# --- Main Data Preparation Orchestrator ---
def prepare_data(
    file_path: str, config: SurveyConfig
) -> Tuple[Optional[pd.DataFrame], ReportManager]:
    script_start_time = datetime.datetime.now()
    reporter = ReportManager(script_start_time)

    llm_service = LLMService(
        config.OPENROUTER_API_KEY, config.LLM_MODEL, config.LLM_PLACEHOLDER_KEY
    )
    reporter.add_entry(
        f"LLM Service Client: {'Initialized successfully.' if llm_service.is_client_available() else 'NOT initialized (API key issue or placeholder used). LLM features will be skipped or limited.'}"
    )
    reporter.add_separator()

    q_info: QuestionTypeInfo = {}

    # --- Core Processing Steps ---
    df_raw = _load_raw_data(file_path, reporter)
    if df_raw is None:
        return None, reporter

    df, q_info = _initialize_columns_and_question_info(df_raw, config, reporter, q_info)
    df, q_info = _anonymize_data(df, reporter, q_info)
    df = _preprocess_data_integrity_and_types(df, reporter, q_info, config)
    df = _validated_ss_mc_responses(df, reporter, q_info)
    df, q_info = _process_multiselect_columns(df, reporter, q_info, config)

    df, q_info = _apply_llm_categorization_tasks(
        df, llm_service, reporter, q_info, config
    )

    df_final = _finalize_dataframe(df, reporter)

    _generate_column_analysis_guide_markdown(df_final, q_info, reporter)

    reporter.add_section_heading("LLM Processing Summary (API Calls, Errors, etc.)")
    if (
        llm_service.is_client_available()
        or llm_service.stats["api_calls"] > 0
        or llm_service.stats["skipped_no_client"] > 0
        or llm_service.stats["errors"] > 0
    ):
        for stat_name, stat_value in llm_service.stats.items():
            reporter.add_entry(
                f"- {stat_name.replace('_', ' ').capitalize()}: {stat_value}"
            )
    else:
        reporter.add_entry(
            "No LLM calls made or LLM-related activity recorded (client may be unavailable or no tasks required LLM)."
        )

    _save_processed_data(df_final, config, reporter, q_info)

    return df_final, reporter


if __name__ == "__main__":
    survey_file_path = "survey_responses.xlsx"

    if not os.path.exists(survey_file_path):
        print(f"Error: Input survey file '{survey_file_path}' not found.")

        sys.exit(1)

    app_config = SurveyConfig()

    processed_df, report_manager_instance = prepare_data(survey_file_path, app_config)

    if processed_df is not None:
        print("\n--- Data preparation script finished successfully. ---")
        print(f"Processed DataFrame shape: {processed_df.shape}")
    else:
        print(
            "\n--- Data preparation script encountered an error and could not complete. ---"
        )

    ensure_dir(app_config.OUTPUT_BASE_DIR)
    report_file_path = os.path.join(
        app_config.OUTPUT_BASE_DIR, "data_preparation_report.md"
    )
    report_manager_instance.save_report_to_file(report_file_path)
