import yaml
from mimecsdi.mime_csdi import MIME_CSDI
import os

if not os.path.exists("/simulations/SRS/mimecsdi"):
    os.makedirs("/simulations/SRS/mimecsdi")

if not os.path.exists("/simulations/Balance/mimecsdi"):
    os.makedirs("/simulations/Balance/mimecsdi")

if not os.path.exists("/simulations/Neyman/mimecsdi"):
    os.makedirs("/simulations/Neyman/mimecsdi")

data_info_srs = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION",
        "URBAN", "INCOME", "MARRIAGE",
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "HYPERTENSION",
        "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
        "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
        "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "EVENT", "EVENT_STAR", "R"
    ],
    "num_vars": [
        "X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
        "Creatinine", "eGFR", "Urea", "Potassium", "Sodium",
        "Chloride", "Bicarbonate", "Calcium", "Magnesium", "Phosphate",
        "Triglyceride", "HDL", "LDL", "Hb", "HCT",
        "RBC", "WBC", "Platelet", "MCV", "RDW",
        "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
        "ALT", "ALP", "GGT", "Bilirubin", "Albumin",
        "Globulin", "Protein", "Glucose", "F_Glucose", "HbA1c",
        "Insulin", "Ferritin", "SBP", "Temperature", "HR",
        "SpO2", "WEIGHT", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
        "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR",
        "C", "C_STAR", "T_I", "T_I_STAR", "W"
    ],
    "phase2_vars": [
        "SMOKE", "ALC", "EXER", "INCOME", "EDU",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE",
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "Glucose", "F_Glucose", "HbA1c", "T_I", "EVENT", "C"
    ],
    "phase1_vars": [
        "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR", "EDU_STAR",
        "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
        "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
        "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR", "T_I_STAR", "EVENT_STAR", "C_STAR"
    ]
}

data_info_balance = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION",
        "URBAN", "INCOME", "MARRIAGE",
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "HYPERTENSION",
        "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
        "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
        "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "EVENT", "EVENT_STAR", "STRATA", "R"
    ],
    "num_vars": [
        "X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
        "Creatinine", "eGFR", "Urea", "Potassium", "Sodium",
        "Chloride", "Bicarbonate", "Calcium", "Magnesium", "Phosphate",
        "Triglyceride", "HDL", "LDL", "Hb", "HCT",
        "RBC", "WBC", "Platelet", "MCV", "RDW",
        "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
        "ALT", "ALP", "GGT", "Bilirubin", "Albumin",
        "Globulin", "Protein", "Glucose", "F_Glucose", "HbA1c",
        "Insulin", "Ferritin", "SBP", "Temperature", "HR",
        "SpO2", "WEIGHT", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
        "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR",
        "C", "C_STAR", "T_I", "T_I_STAR", "W"
    ],
    "phase2_vars": [
        "SMOKE", "ALC", "EXER", "INCOME", "EDU",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE",
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "Glucose", "F_Glucose", "HbA1c", "T_I", "EVENT", "C"
    ],
    "phase1_vars": [
        "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR", "EDU_STAR",
        "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
        "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
        "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR", "T_I_STAR", "EVENT_STAR", "C_STAR"
    ]
}

data_info_neyman = {
    "weight_var": "W",
    "cat_vars": [
        "SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION",
        "URBAN", "INCOME", "MARRIAGE",
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "HYPERTENSION",
        "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
        "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
        "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "EVENT", "EVENT_STAR", "STRATA", "R"
    ],
    "num_vars": [
        "X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
        "Creatinine", "eGFR", "Urea", "Potassium", "Sodium",
        "Chloride", "Bicarbonate", "Calcium", "Magnesium", "Phosphate",
        "Triglyceride", "HDL", "LDL", "Hb", "HCT",
        "RBC", "WBC", "Platelet", "MCV", "RDW",
        "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
        "ALT", "ALP", "GGT", "Bilirubin", "Albumin",
        "Globulin", "Protein", "Glucose", "F_Glucose", "HbA1c",
        "Insulin", "Ferritin", "SBP", "Temperature", "HR",
        "SpO2", "WEIGHT", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
        "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR",
        "C", "C_STAR", "T_I", "T_I_STAR", "W"
    ],
    "phase2_vars": [
        "SMOKE", "ALC", "EXER", "INCOME", "EDU",
        "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE",
        "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
        "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
        "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
        "Glucose", "F_Glucose", "HbA1c", "T_I", "EVENT", "C"
    ],
    "phase1_vars": [
        "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR", "EDU_STAR",
        "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR",
        "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
        "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
        "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
        "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR", "T_I_STAR", "EVENT_STAR", "C_STAR"
    ]
}

with open("mimecsdi/config/survival.yaml", "r") as f:
    config = yaml.safe_load(f)

for i in range(1, 31):
    digit = str(i).zfill(4)
    file_path_srs = "F:/phd-thesis/data/Sample/SRS/" + digit + ".csv"
    file_path_bal = "F:/phd-thesis/data/Sample/Balance/" + digit + ".csv"
    file_path_ney = "F:/phd-thesis/data/Sample/Neyman/" + digit + ".csv"

    save_path_srs = "F:/phd-thesis/simulations/SRS/mimecsdi/" + digit + ".xlsx"
    save_path_bal = "F:/phd-thesis/simulations/Balance/mimecsdi/" + digit + ".xlsx"
    save_path_ney = "F:/phd-thesis/simulations/Neyman/mimecsdi/" + digit + ".xlsx"

    mime_mod_srs = MIME_CSDI(config, data_info_srs)
    mime_mod_srs.fit(file_path_srs)
    mime_mod_srs.impute(save_path=save_path_srs)

    mime_mod_bal = MIME_CSDI(config, data_info_balance)
    mime_mod_bal.fit(file_path_bal)
    mime_mod_bal.impute(save_path=save_path_bal)

    mime_mod_ney = MIME_CSDI(config, data_info_neyman)
    mime_mod_ney.fit(file_path_ney)
    mime_mod_ney.impute(save_path=save_path_ney)


