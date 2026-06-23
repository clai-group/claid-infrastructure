"""
feature_extraction_gnn.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DURATION CONSTANTS
# ============================================================

DURATION_ORDER = {
    '0-14 days':  1,
    '15-30 days': 2,
    '30-90 days': 3,
    '90+ days':   4
}
DURATIONS = ['0-14 days', '15-30 days', '30-90 days', '90+ days']


# ============================================================
# PHENX → SYSTEMS MAPPING
# Built from your complete 420-phenotype list.
# Primary system = first entry.
# Multiple entries = genuinely multisystem conditions.
# ============================================================

PHENX_TO_SYSTEMS = {

    # ── PROCEDURES ───────────────────────────────────────────
    'Administration of albumin and globulin':                                                           ['Procedure'],
    'Administration of antibiotics':                                                                    ['Procedure'],
    'Administration of anti-inflammatory agents':                                                       ['Procedure'],
    'Administration of nutritional and electrolytic substances':                                        ['Procedure'],
    'Administration of thrombolytics and platelet inhibitors':                                          ['Procedure'],
    'Administration of diagnostic substances, NEC':                                                     ['Procedure'],
    'Administration of therapeutic substances, NEC':                                                    ['Procedure'],
    'Administration and transfusion of bone marrow, stem cells, pancreatic islet cells, and t-cells':  ['Procedure'],
    'Chemotherapy':                                                                                     ['Procedure', 'Oncologic'],
    'Radiation Therapy':                                                                                ['Procedure', 'Oncologic'],
    'Encounter for antineoplastic therapies':                                                           ['Procedure', 'Oncologic'],
    'Encounter for prophylactic or other procedures':                                                   ['Procedure'],
    'Implant, device or graft related encounter':                                                       ['Procedure'],
    'Vaccinations':                                                                                     ['Procedure'],
    'COVID-19 vaccinations':                                                                            ['Procedure'],
    'Regional anesthesia':                                                                              ['Procedure'],
    'Rh immunoglobulin and other serum infusion':                                                       ['Procedure'],
    'Intravenous induction of labor':                                                                   ['Procedure', 'Pregnancy'],
    'Cervical ripening':                                                                                ['Procedure', 'Pregnancy'],
    'Infusion of vasopressor':                                                                          ['Procedure', 'Cardiovascular'],
    'Transfusion of blood and blood products':                                                          ['Procedure', 'Hematologic'],
    'Transfusion of plasma':                                                                            ['Procedure', 'Hematologic'],
    'Transfusion of clotting factors':                                                                  ['Procedure', 'Hematologic'],
    'Peritoneal dialysis':                                                                              ['Procedure', 'Renal/GU'],
    'Irrigation (diagnostic and therapeutic)':                                                          ['Procedure'],
    'Nuclear Medicine':                                                                                 ['Procedure'],
    'Mental Health Therapy':                                                                            ['Procedure', 'Psych'],
    'Extracorporeal or Systemic Assistance and Performance':                                            ['Procedure'],
    'Extracorporeal or Systemic Therapies':                                                             ['Procedure'],
    'Cardiovascular Procedures':                                                                        ['Procedure', 'Cardiovascular'],
    'Respiratory System Procedures':                                                                    ['Procedure', 'Respiratory'],
    'Gastrointestinal System Procedures':                                                               ['Procedure', 'GI'],
    'Central Nervous System Procedures':                                                                ['Procedure', 'Neurology'],
    'Peripheral Nervous System Procedures':                                                             ['Procedure', 'Neurology'],
    'Eye Procedures':                                                                                   ['Procedure', 'Eyes & Ears'],
    'Ear, Nose, and Throat Procedures':                                                                 ['Procedure', 'Ears'],
    'Musculoskeletal, Subcutaneous Tissue, and Fascia Procedures':                                      ['Procedure', 'MSK'],
    'Lymphatic and Hemic System Procedures':                                                            ['Procedure', 'Hematologic'],
    'Urinary System Procedures':                                                                        ['Procedure', 'Renal/GU'],
    'Female Reproductive System Procedures':                                                            ['Procedure', 'Gynecology/Pelvis'],
    'Male Reproductive System Procedures':                                                              ['Procedure', 'Renal/GU'],
    'Skin and Breast Procedures':                                                                       ['Procedure', 'Skin'],
    'Hepatobiliary and Pancreas Procedures':                                                            ['Procedure', 'GI'],
    'Endocrine Procedures':                                                                             ['Procedure', 'Endocrine'],
    'General Region Procedures':                                                                        ['Procedure'],
    'Pregnancy-Related Procedures':                                                                     ['Procedure', 'Pregnancy'],
    'Acquired absence of limb or organ':                                                                ['Procedure'],

    # ── INJURIES & ADVERSE EFFECTS ───────────────────────────
    'Injury to nerves, muscles and tendons':                                                            ['Procedure', 'MSK'],
    'Injury, sequela':                                                                                  ['Procedure'],
    'Injury to blood vessels':                                                                          ['Procedure', 'Cardiovascular'],
    'Superficial injury; contusion':                                                                    ['Procedure'],
    'Amputation of a limb, subsequent encounter':                                                       ['Procedure', 'MSK'],
    'Amputation of a limb, initial encounter':                                                          ['Procedure', 'MSK'],
    'Dislocation':                                                                                      ['MSK'],
    'Sprains and strains':                                                                              ['MSK'],
    'Other specified injury':                                                                           ['Procedure'],
    'Other specified injury, subsequent encounter':                                                     ['Procedure'],
    'Other unspecified injuries, subsequent encounter':                                                 ['Procedure'],
    'fracture':                                                                                         ['MSK'],
    'Fracture of head and neck':                                                                        ['MSK'],
    'Traumatic brain injury (TBI); concussion':                                                         ['Neurology'],
    'Spinal cord injury (SCI), subsequent encounter':                                                   ['Neurology', 'MSK'],
    'Spinal cord injury (SCI), initial encounter':                                                      ['Neurology', 'MSK'],
    'Internal organ injury':                                                                            ['Procedure'],
    'Poisoning/toxic effect/adverse effects/underdosing, sequela':                                      ['Procedure'],
    'procedural complication':                                                                          ['Procedure'],
    'Effect of foreign body entering opening, subsequent encounter':                                    ['Procedure'],
    'Adverse effects of drugs and medicaments, initial encounter':                                      ['Procedure'],
    'Adverse effects of drugs and medicaments, subsequent encounter':                                   ['Procedure'],
    'Poisoning by drugs, subsequent encounter':                                                         ['Procedure'],
    'External cause codes: unspecified mechanism':                                                      ['Procedure'],
    'Drug induced or toxic related condition':                                                          ['Procedure'],
    'Birth trauma':                                                                                     ['Pediatric/Neonatal'],
    'Allergic reactions':                                                                               ['Immune'],
    'Allergic reactions, subsequent encounter':                                                         ['Immune'],

    # ── CARDIOVASCULAR ───────────────────────────────────────
    'Coronary atherosclerosis and other heart disease':                                                 ['Cardiovascular'],
    'Heart failure':                                                                                    ['Cardiovascular'],
    'Cardiac dysrhythmias':                                                                             ['Cardiovascular'],
    'Cardiac arrest and ventricular fibrillation':                                                      ['Cardiovascular'],
    'Acute myocardial infarction':                                                                      ['Cardiovascular'],
    'Complications of acute myocardial infarction':                                                     ['Cardiovascular'],
    'Conduction disorders':                                                                             ['Cardiovascular'],
    'Myocarditis and cardiomyopathy':                                                                   ['Cardiovascular'],
    'Pericarditis and pericardial disease':                                                             ['Cardiovascular'],
    'Endocarditis and endocardial disease':                                                             ['Cardiovascular'],
    'Nonrheumatic and unspecified valve disorders':                                                     ['Cardiovascular'],
    'Chronic rheumatic heart disease':                                                                  ['Cardiovascular'],
    'Acute rheumatic heart disease':                                                                    ['Cardiovascular'],
    'Other and ill-defined heart disease':                                                              ['Cardiovascular'],
    'Cardiac and circulatory congenital anomalies':                                                     ['Cardiovascular', 'Pediatric/Neonatal'],
    'Cardiac cancers':                                                                                  ['Cardiovascular', 'Oncologic'],
    'Arterial dissections':                                                                             ['Cardiovascular'],
    'Aortic; peripheral; and visceral artery aneurysms':                                                ['Cardiovascular'],
    'Aortic and peripheral arterial embolism or thrombosis':                                            ['Cardiovascular'],
    'Peripheral and visceral vascular disease':                                                         ['Cardiovascular'],
    'Acute phlebitis; thrombophlebitis and thromboembolism':                                            ['Cardiovascular'],
    'Chronic phlebitis; thrombophlebitis and thromboembolism':                                          ['Cardiovascular'],
    'Postthrombotic syndrome and venous insufficiency/hypertension':                                    ['Cardiovascular'],
    'Other specified diseases of veins and lymphatics':                                                 ['Cardiovascular'],
    'Varicose veins of lower extremity':                                                                ['Cardiovascular'],
    'Hemorrhoids':                                                                                      ['Cardiovascular', 'GI'],
    'Hypotension':                                                                                      ['Cardiovascular'],
    'Shock':                                                                                            ['Cardiovascular', 'Systemic'],
    'HTN':                                                                                              ['Cardiovascular'],
    'Essential hypertension':                                                                           ['Cardiovascular'],
    'HTN; heart':                                                                                       ['Cardiovascular'],
    'HTN; brain':                                                                                       ['Cardiovascular', 'Neurology'],
    'HTN; renal':                                                                                       ['Cardiovascular', 'Renal/GU'],
    'HTN; endocrine':                                                                                   ['Cardiovascular', 'Endocrine'],
    'HTN; eye':                                                                                         ['Cardiovascular', 'Eyes & Ears'],
    'Other specified and unspecified circulatory disease':                                              ['Cardiovascular'],
    'Circulatory signs and symptoms':                                                                   ['Cardiovascular'],
    'Postprocedural or postoperative circulatory system complication':                                  ['Procedure', 'Cardiovascular'],
    'Pulmonary heart disease':                                                                          ['Cardiovascular', 'Respiratory'],
    'Acute pulmonary embolism':                                                                         ['Respiratory', 'Cardiovascular'],
    'Cerebral infarction':                                                                              ['Neurology', 'Cardiovascular'],
    'Acute hemorrhagic cerebrovascular disease':                                                        ['Neurology', 'Cardiovascular'],
    'Other and ill-defined cerebrovascular disease':                                                    ['Neurology', 'Cardiovascular'],
    'Occlusion or stenosis of precerebral or cerebral arteries without infarction':                    ['Neurology', 'Cardiovascular'],
    'Sequela of hemorrhagic cerebrovascular disease':                                                   ['Neurology', 'Cardiovascular'],
    'Sequela of cerebral infarction and other cerebrovascular disease':                                 ['Neurology', 'Cardiovascular'],
    'Transient cerebral ischemia':                                                                      ['Neurology', 'Cardiovascular'],
    'Syncope':                                                                                          ['Neurology', 'Cardiovascular'],
    'Gangrene':                                                                                         ['Cardiovascular', 'Skin', 'MSK'],

    # ── RESPIRATORY ──────────────────────────────────────────
    'Pneumonia (except that caused by tuberculosis)':                                                   ['Respiratory', 'Infectious'],
    'Influenza':                                                                                        ['Respiratory', 'Infectious'],
    'Acute bronchitis':                                                                                 ['Respiratory', 'Infectious'],
    'Viral infection; acute bronchitis':                                                                ['Respiratory', 'Infectious'],
    'Viral infection; respiratory':                                                                     ['Respiratory', 'Infectious'],
    'Other specified upper respiratory infections':                                                     ['Respiratory', 'Infectious'],
    'Sinusitis':                                                                                        ['Respiratory', 'Infectious'],
    'Acute and chronic tonsillitis':                                                                    ['Respiratory', 'Infectious'],
    'Chronic obstructive pulmonary disease and bronchiectasis':                                         ['Respiratory'],
    'Asthma':                                                                                           ['Respiratory'],
    'Aspiration pneumonitis':                                                                           ['Respiratory'],
    'Pleurisy, pleural effusion and pulmonary collapse':                                                ['Respiratory'],
    'Pneumothorax':                                                                                     ['Respiratory'],
    'Mediastinal disorders':                                                                            ['Respiratory'],
    'Lung disease due to external agents':                                                              ['Respiratory'],
    'Other specified and unspecified lower respiratory disease':                                        ['Respiratory'],
    'Other specified and unspecified upper respiratory disease':                                        ['Respiratory'],
    'Respiratory failure; insufficiency; arrest':                                                       ['Respiratory'],
    'Respiratory signs and symptoms':                                                                   ['Respiratory'],
    'Respiratory cancers':                                                                              ['Respiratory', 'Oncologic'],
    'Respiratory congenital malformations':                                                             ['Respiratory', 'Pediatric/Neonatal'],
    'Tuberculosis':                                                                                     ['Respiratory', 'Infectious'],
    'Postprocedural or postoperative respiratory system complication':                                  ['Procedure', 'Respiratory'],
    'Respiratory perinatal condition':                                                                  ['Pediatric/Neonatal', 'Respiratory'],
    'Respiratory distress syndrome':                                                                    ['Pediatric/Neonatal', 'Respiratory'],
    'Mesothelioma':                                                                                     ['Oncologic', 'Respiratory'],

    # ── GASTROINTESTINAL ─────────────────────────────────────
    'Appendicitis and other appendiceal conditions':                                                    ['GI'],
    'Abdominal hernia':                                                                                 ['GI'],
    'Regional enteritis and ulcerative colitis':                                                        ['GI', 'Immune'],
    'Noninfectious gastroenteritis':                                                                    ['GI'],
    'Gastritis and duodenitis':                                                                         ['GI'],
    'Gastrointestinal hemorrhage':                                                                      ['GI'],
    'Gastroduodenal ulcer':                                                                             ['GI'],
    'Other specified and unspecified disorders of stomach and duodenum':                                ['GI'],
    'Intestinal obstruction and ileus':                                                                 ['GI'],
    'Diverticulosis and diverticulitis':                                                                ['GI'],
    'Anal and rectal conditions':                                                                       ['GI'],
    'Esophageal disorders':                                                                             ['GI'],
    'Dysphagia':                                                                                        ['GI', 'Neurology'],
    'Nausea and vomiting':                                                                              ['GI'],
    'Abdominal symptoms':                                                                               ['GI'],
    'Intestinal infection':                                                                             ['GI', 'Infectious'],
    'Foodborne intoxications':                                                                          ['GI', 'Infectious'],
    'Peritonitis and intra-abdominal abscess':                                                          ['GI', 'Infectious'],
    'Biliary tract disease':                                                                            ['GI'],
    'Gastrointestinal and biliary perforation':                                                         ['GI'],
    'Noninfectious hepatitis':                                                                          ['GI'],
    'Hepatic failure':                                                                                  ['GI'],
    'Hepatitis':                                                                                        ['GI', 'Infectious'],
    'Other specified and unspecified liver disease':                                                    ['GI'],
    'Pancreatic disorders (excluding diabetes)':                                                        ['GI'],
    'Other specified and unspecified gastrointestinal disorders':                                       ['GI'],
    'Gastrointestinal cancers':                                                                         ['GI', 'Oncologic'],
    'Digestive congenital anomalies':                                                                   ['GI', 'Pediatric/Neonatal'],
    'Neonatal digestive and feeding disorders':                                                         ['Pediatric/Neonatal', 'GI'],
    'Postprocedural or postoperative digestive system complication':                                    ['Procedure', 'GI'],
    'Cystic fibrosis':                                                                                  ['Respiratory', 'GI', 'Endocrine'],

    # ── NEUROLOGY ────────────────────────────────────────────
    'Epilepsy; convulsions':                                                                            ['Neurology'],
    'Headache; including migraine':                                                                     ['Neurology'],
    'Coma; stupor; and brain damage':                                                                   ['Neurology'],
    'Paralysis (other than cerebral palsy)':                                                            ['Neurology'],
    'Cerebral palsy':                                                                                   ['Neurology', 'Pediatric/Neonatal'],
    "Parkinson`s disease":                                                                              ['Neurology'],
    'Multiple sclerosis':                                                                               ['Neurology', 'Immune'],
    'Polyneuropathies':                                                                                 ['Neurology'],
    'Nerve and nerve root disorders':                                                                   ['Neurology'],
    'Nervous system pain and pain syndromes':                                                           ['Neurology'],
    'Myopathies':                                                                                       ['Neurology', 'MSK'],
    'Neurocognitive disorders':                                                                         ['Neurology', 'Psych'],
    'Neurodevelopmental disorders':                                                                     ['Neurology', 'Psych', 'Pediatric/Neonatal'],
    'CNS abscess':                                                                                      ['Neurology', 'Infectious'],
    'Encephalitis':                                                                                     ['Neurology', 'Infectious'],
    'Meningitis':                                                                                       ['Neurology', 'Infectious'],
    'Viral infection; encephalitis':                                                                    ['Neurology', 'Infectious'],
    'Viral infection; CNS/PNS':                                                                         ['Neurology', 'Infectious'],
    'Viral infection; meningitis':                                                                      ['Neurology', 'Infectious'],
    'Nervous system congenital anomalies':                                                              ['Neurology', 'Pediatric/Neonatal'],
    'Other nervous system disorders':                                                                   ['Neurology'],
    'Nervous system signs and symptoms':                                                                ['Neurology'],
    'Nervous system cancers':                                                                           ['Neurology', 'Oncologic'],
    'Postprocedural or postoperative nervous system complication':                                      ['Procedure', 'Neurology'],

    # ── ENDOCRINE / METABOLIC ────────────────────────────────
    'Diabetes mellitus':                                                                                ['Endocrine'],
    'Diabetes mellitus, Type 1':                                                                        ['Endocrine'],
    'Diabetes mellitus, Type 2':                                                                        ['Endocrine', 'Cardiovascular', 'Neurology', 'Renal/GU', 'Eyes & Ears', 'Skin'],
    'Diabetes mellitus; nerve':                                                                         ['Endocrine', 'Neurology'],
    'Diabetes mellitus; eye':                                                                           ['Endocrine', 'Eyes & Ears'],
    'Diabetes mellitus; renal complication':                                                            ['Endocrine', 'Renal/GU'],
    'Diabetes mellitus; joint':                                                                         ['Endocrine', 'MSK'],
    'Diabetes mellitus; brain damage':                                                                  ['Endocrine', 'Neurology'],
    'Diabetes mellitus; skin':                                                                          ['Endocrine', 'Skin'],
    'Diabetes mellitus; childbirth':                                                                    ['Endocrine', 'Pregnancy'],
    'Thyroid disorders':                                                                                ['Endocrine'],
    'Pituitary disorders':                                                                              ['Endocrine'],
    'Disorders of lipid metabolism':                                                                    ['Endocrine', 'Cardiovascular'],
    'Other specified and unspecified endocrine disorders':                                              ['Endocrine'],
    'Endocrine system cancers':                                                                         ['Endocrine', 'Oncologic'],
    'Malignant neuroendocrine tumors':                                                                  ['Endocrine', 'Oncologic'],
    'Obesity':                                                                                          ['Endocrine', 'Cardiovascular', 'MSK'],
    'Nutritional deficiencies':                                                                         ['Endocrine', 'Systemic'],
    'Sequela of malnutrition and other nutritional deficiencies':                                       ['Endocrine', 'Systemic'],
    'Malnutrition':                                                                                     ['Endocrine', 'Systemic'],
    'Other specified and unspecified nutritional and metabolic disorders':                               ['Endocrine', 'Systemic'],
    'Fluid and electrolyte disorders':                                                                  ['Endocrine', 'Systemic'],
    'Postprocedural or postoperative endocrine or metabolic complication':                              ['Procedure', 'Endocrine'],

    # ── MUSCULOSKELETAL ──────────────────────────────────────
    'Osteoarthritis':                                                                                   ['MSK'],
    'Rheumatoid arthritis and related disease':                                                         ['MSK', 'Immune'],
    'Systemic lupus erythematosus and connective tissue disorders':                                     ['MSK', 'Immune', 'Cardiovascular', 'Renal/GU'],
    'Other specified connective tissue disease':                                                        ['MSK', 'Immune'],
    'Spondylopathies/spondyloarthropathy (including infective)':                                        ['MSK'],
    'Low back pain':                                                                                    ['MSK'],
    'Musculoskeletal pain, not low back pain':                                                          ['MSK'],
    'Other specified joint disorders':                                                                  ['MSK'],
    'Crystal arthropathies (excluding gout)':                                                           ['MSK'],
    'Gout':                                                                                             ['MSK', 'Endocrine'],
    'Traumatic arthropathy':                                                                            ['MSK'],
    'Neurogenic/neuropathic arthropathy':                                                               ['MSK', 'Neurology'],
    'Infective arthritis':                                                                              ['MSK', 'Infectious'],
    'Immune-mediated/reactive arthropathies':                                                           ['MSK', 'Immune'],
    'Juvenile arthritis':                                                                               ['MSK', 'Immune', 'Pediatric/Neonatal'],
    'Other specified chronic arthropathy':                                                              ['MSK'],
    'Osteoporosis':                                                                                     ['MSK'],
    'Aseptic necrosis and osteonecrosis':                                                               ['MSK'],
    'pathological fracture':                                                                            ['MSK'],
    'Pathological, stress and atypical fractures, sequela':                                             ['MSK'],
    'Stress fracture':                                                                                  ['MSK'],
    'Biomechanical lesions':                                                                            ['MSK'],
    'Muscle disorders':                                                                                 ['MSK'],
    'Tendon and synovial disorders':                                                                    ['MSK'],
    'Acquired deformities (excluding foot)':                                                            ['MSK'],
    'Acquired foot deformities':                                                                        ['MSK'],
    'Scoliosis and other postural dorsopathic deformities':                                             ['MSK'],
    'Osteomyelitis':                                                                                    ['MSK', 'Infectious'],
    'Other specified bone disease and musculoskeletal deformities':                                     ['MSK'],
    'Musculoskeletal congenital conditions':                                                            ['MSK', 'Pediatric/Neonatal'],
    'Postprocedural or postoperative musculoskeletal system complication':                              ['Procedure', 'MSK'],
    'Vasculitis':                                                                                       ['Immune', 'Cardiovascular'],

    # ── RENAL / GENITOURINARY ────────────────────────────────
    'Acute and unspecified renal failure':                                                              ['Renal/GU'],
    'Chronic kidney disease':                                                                           ['Renal/GU'],
    'Nephritis; nephrosis; renal sclerosis':                                                            ['Renal/GU'],
    'Other specified and unspecified diseases of kidney and ureters':                                   ['Renal/GU'],
    'Calculus of urinary tract':                                                                        ['Renal/GU'],
    'Urinary tract infections':                                                                         ['Renal/GU', 'Infectious'],
    'Urinary incontinence':                                                                             ['Renal/GU'],
    'Genitourinary signs and symptoms':                                                                 ['Renal/GU'],
    'Hematuria':                                                                                        ['Renal/GU'],
    'Proteinuria':                                                                                      ['Renal/GU'],
    'Vesicoureteral reflux':                                                                            ['Renal/GU'],
    'Other specified and unspecified diseases of bladder and urethra':                                  ['Renal/GU'],
    'Urinary system cancers':                                                                           ['Renal/GU', 'Oncologic'],
    'Genitourinary congenital anomalies':                                                               ['Renal/GU', 'Pediatric/Neonatal'],
    'Postprocedural or postoperative genitourinary system complication':                                ['Procedure', 'Renal/GU'],
    'Other specified male genital disorders':                                                           ['Renal/GU'],
    'Male infertility':                                                                                 ['Renal/GU'],
    'Erectile dysfunction':                                                                             ['Renal/GU'],
    'Hyperplasia of prostate':                                                                          ['Renal/GU'],
    'Inflammatory conditions of male genital organs':                                                   ['Renal/GU', 'Infectious'],
    'Male reproductive system cancers':                                                                 ['Renal/GU', 'Oncologic'],
    'Sexually transmitted infections (excluding HIV and hepatitis)':                                    ['Infectious', 'Renal/GU'],
    'Viral infection; STI':                                                                             ['Infectious', 'Renal/GU'],

    # ── GYNECOLOGY / PELVIS ──────────────────────────────────
    'Other specified female genital disorders':                                                         ['Gynecology/Pelvis'],
    'Menstrual disorders':                                                                              ['Gynecology/Pelvis'],
    'Menopausal disorders':                                                                             ['Gynecology/Pelvis'],
    'Endometriosis':                                                                                    ['Gynecology/Pelvis'],
    'Female infertility':                                                                               ['Gynecology/Pelvis'],
    'Prolapse of female genital organs':                                                                ['Gynecology/Pelvis'],
    'Nonmalignant breast conditions':                                                                   ['Gynecology/Pelvis'],
    'Inflammatory diseases of female pelvic organs':                                                    ['Gynecology/Pelvis', 'Infectious'],
    'Female reproductive system cancers':                                                               ['Gynecology/Pelvis', 'Oncologic'],
    'Breast cancer':                                                                                    ['Gynecology/Pelvis', 'Oncologic'],

    # ── PREGNANCY ────────────────────────────────────────────
    'Uncomplicated pregnancy, delivery or puerperium':                                                  ['Pregnancy'],
    'Induced abortion and complications of termination of pregnancy':                                   ['Pregnancy'],
    'Molar pregnancy and other abnormal products of conception':                                        ['Pregnancy'],
    'Spontaneous abortion and complications of spontaneous abortion':                                   ['Pregnancy'],
    'Complications following ectopic and/or molar pregnancy':                                           ['Pregnancy'],
    'Supervision of high-risk pregnancy':                                                               ['Pregnancy'],
    'Malposition, disproportion or other labor complications':                                          ['Pregnancy'],
    'OB-related trauma to perineum and vulva':                                                          ['Pregnancy'],
    'Early or threatened labor':                                                                        ['Pregnancy'],
    'Hemorrhage after first trimester':                                                                 ['Pregnancy'],
    'Early, first or unspecified trimester hemorrhage':                                                 ['Pregnancy'],
    'Multiple gestation':                                                                               ['Pregnancy'],
    'Polyhydramnios and other problems of amniotic cavity':                                             ['Pregnancy'],
    'Maternal intrauterine infection':                                                                  ['Pregnancy', 'Infectious'],
    'Prolonged pregnancy':                                                                              ['Pregnancy'],
    'Contraceptive and procreative management':                                                         ['Pregnancy', 'Gynecology/Pelvis'],
    'Other specified complications in pregnancy':                                                       ['Pregnancy'],
    'Hypertension and hypertensive-related conditions complicating pregnancy; childbirth; and the puerperium': ['Pregnancy', 'Cardiovascular'],
    'Complications specified during childbirth':                                                        ['Pregnancy'],
    'Complications specified during the puerperium':                                                    ['Pregnancy'],
    'Maternal care for abnormality of pelvic organs':                                                   ['Pregnancy'],
    'Maternal care related to fetal conditions':                                                        ['Pregnancy'],
    'Maternal care related to disorders of the placenta and placental implantation':                    ['Pregnancy'],

    # ── EYES & EARS ──────────────────────────────────────────
    'Blindness and vision defects':                                                                     ['Eyes & Ears'],
    'Glaucoma':                                                                                         ['Eyes & Ears'],
    'Cataract and other lens disorders':                                                                ['Eyes & Ears'],
    'Refractive error':                                                                                 ['Eyes & Ears'],
    'Retinal and vitreous conditions':                                                                  ['Eyes & Ears'],
    'Cornea and external disease':                                                                      ['Eyes & Ears'],
    'Uveitis and ocular inflammation':                                                                  ['Eyes & Ears', 'Immune'],
    'Strabismus':                                                                                       ['Eyes & Ears'],
    'Neuro-ophthalmology':                                                                              ['Eyes & Ears', 'Neurology'],
    'Oculofacial plastics and orbital conditions':                                                      ['Eyes & Ears'],
    'Other specified eye disorders':                                                                    ['Eyes & Ears'],
    'Congenital malformations of eye, ear, face, neck':                                                 ['Eyes & Ears', 'Pediatric/Neonatal'],
    'Postprocedural or postoperative eye complication':                                                 ['Procedure', 'Eyes & Ears'],
    'Hearing loss':                                                                                     ['Ears'],
    'Otitis media':                                                                                     ['Ears', 'Infectious'],
    'Diseases of middle ear and mastoid (except otitis media)':                                         ['Ears'],
    'Diseases of inner ear and related conditions':                                                     ['Ears'],
    'Other specified and unspecified disorders of the ear':                                             ['Ears'],
    'Postprocedural or postoperative ear and/or mastoid process complication':                          ['Procedure', 'Ears'],

    # ── PSYCHIATRY ───────────────────────────────────────────
    'Depressive disorders':                                                                             ['Psych'],
    'Bipolar and related disorders':                                                                    ['Psych'],
    'Other specified and unspecified mood disorders':                                                   ['Psych'],
    'Anxiety and fear-related disorders':                                                               ['Psych'],
    'Trauma- and stressor-related disorders':                                                           ['Psych'],
    'Obsessive-compulsive and related disorders':                                                       ['Psych'],
    'Schizophrenia spectrum and other psychotic disorders':                                             ['Psych'],
    'Personality disorders':                                                                            ['Psych'],
    'Feeding and eating disorders':                                                                     ['Psych', 'Endocrine'],
    'Somatic disorders':                                                                                ['Psych'],
    'Disruptive, impulse-control and conduct disorders':                                                ['Psych'],
    'Mental and substance use disorders in remission':                                                  ['Psych'],
    'Miscellaneous mental and behavioral disorders/conditions':                                         ['Psych'],
    'Suicidal ideation/attempt/intentional self-harm':                                                  ['Psych'],
    'Opioid-related disorders':                                                                         ['Psych'],
    'Stimulant-related disorders':                                                                      ['Psych'],
    'Cannabis-related disorders':                                                                       ['Psych'],
    'Sedative-related disorders':                                                                       ['Psych'],
    'Inhalant-related disorders':                                                                       ['Psych'],
    'Alcohol-related disorders':                                                                        ['Psych'],
    'Other specified substance-related disorders':                                                      ['Psych'],
    'Tobacco-related disorders':                                                                        ['Psych', 'Respiratory'],
    'Symptoms of mental and substance use conditions':                                                  ['Psych'],

    # ── SKIN ─────────────────────────────────────────────────
    'Other specified and unspecified skin disorders':                                                   ['Skin'],
    'Other specified inflammatory condition of skin':                                                   ['Skin', 'Immune'],
    'Contact dermatitis':                                                                               ['Skin', 'Immune'],
    'Skin and subcutaneous tissue infections':                                                          ['Skin', 'Infectious'],
    'Non-pressure ulcer of skin':                                                                       ['Skin', 'Cardiovascular'],
    'Pressure ulcer of skin':                                                                           ['Skin'],
    'Skin/Subcutaneous signs and symptoms':                                                             ['Skin'],
    'Skin cancers':                                                                                     ['Skin', 'Oncologic'],
    'Postprocedural or postoperative skin complication':                                                ['Procedure', 'Skin'],

    # ── SYSTEMIC / GENERAL ───────────────────────────────────
    'Abnormal findings without diagnosis':                                                              ['Systemic'],
    'Other general signs and symptoms':                                                                 ['Systemic'],
    'General sensation/perception signs and symptoms':                                                  ['Systemic', 'Neurology', 'Psych'],
    'Malaise and fatigue':                                                                              ['Systemic'],
    'Sleep wake disorders':                                                                             ['Systemic', 'Psych'],
    'Fever':                                                                                            ['Systemic', 'Infectious'],
    'Nonspecific chest pain':                                                                           ['Systemic', 'Cardiovascular'],
    'Immunity disorders':                                                                               ['Immune', 'Systemic'],
    'Conditions due to neoplasm or the treatment of neoplasm':                                          ['Oncologic', 'Systemic'],

    # ── HEMATOLOGIC ──────────────────────────────────────────
    'Coagulation and hemorrhagic disorders':                                                            ['Hematologic'],
    'Diseases of white blood cells':                                                                    ['Hematologic'],
    'Hemolytic anemia':                                                                                 ['Hematologic'],
    'Sickle cell trait/anemia':                                                                         ['Hematologic'],
    'Nutritional anemia':                                                                               ['Hematologic', 'Endocrine'],
    'Aplastic anemia':                                                                                  ['Hematologic'],
    'Acute posthemorrhagic anemia':                                                                     ['Hematologic'],
    'Other specified and unspecified hematologic conditions':                                           ['Hematologic'],
    'Myelodysplastic syndrome (MDS)':                                                                   ['Hematologic', 'Oncologic'],
    'Leukemia':                                                                                         ['Hematologic', 'Oncologic'],
    'Non-Hodgkin lymphoma':                                                                             ['Hematologic', 'Oncologic'],
    'Hodgkin lymphoma':                                                                                 ['Hematologic', 'Oncologic'],
    'Multiple myeloma':                                                                                 ['Hematologic', 'Oncologic'],
    'Postprocedural or postoperative complications of the spleen':                                      ['Procedure', 'Hematologic'],

    # ── ONCOLOGIC ────────────────────────────────────────────
    'Head and neck cancers':                                                                            ['Oncologic'],
    'Malignant neoplasm, unspecified':                                                                  ['Oncologic'],
    'Cancer of other sites':                                                                            ['Oncologic'],
    'Sarcoma':                                                                                          ['Oncologic', 'MSK'],
    'Benign neoplasms':                                                                                 ['Oncologic'],
    'Secondary malignancies':                                                                           ['Oncologic'],
    'Neoplasms of unspecified nature or uncertain behavior':                                            ['Oncologic'],
    'Bone cancer':                                                                                      ['Oncologic', 'MSK'],

    # ── INFECTIOUS ───────────────────────────────────────────
    'Bacterial infections':                                                                             ['Infectious'],
    'Viral infection':                                                                                  ['Infectious'],
    'Fungal infections':                                                                                ['Infectious'],
    'Parasitic, other specified and unspecified infections':                                            ['Infectious'],
    'HIV infection':                                                                                    ['Infectious', 'Immune'],
    'Septicemia':                                                                                       ['Infectious', 'Cardiovascular'],
    'Resistance to antimicrobial drugs':                                                                ['Infectious'],
    'Sequela of specified infectious disease conditions':                                               ['Infectious'],
    'Any dental condition including traumatic injury':                                                  ['Infectious'],

    # ── PEDIATRIC / NEONATAL ─────────────────────────────────
    'Newborn affected by maternal conditions or complications of labor/delivery':                       ['Pediatric/Neonatal'],
    'Short gestation; low birth weight; and fetal growth retardation':                                  ['Pediatric/Neonatal'],
    'Hemolytic jaundice and perinatal jaundice':                                                        ['Pediatric/Neonatal', 'Hematologic'],
    'Neonatal acidemia and hypoxia':                                                                    ['Pediatric/Neonatal'],
    'Other specified and unspecified perinatal conditions':                                             ['Pediatric/Neonatal'],
    'Perinatal infections':                                                                             ['Pediatric/Neonatal', 'Infectious'],

    # ── COVID ────────────────────────────────────────────────
    'COVID1': ['Infectious', 'Respiratory', 'Systemic'],
    'COVID2': ['Infectious', 'Respiratory', 'Systemic'],
    'COVID3': ['Infectious', 'Respiratory', 'Systemic'],
    'COVID4': ['Infectious', 'Respiratory', 'Systemic'],
    'COVID5': ['Infectious', 'Respiratory', 'Systemic'],
    'COVID6': ['Infectious', 'Respiratory', 'Systemic'],
}

# ── Derived lookups ───────────────────────────────────────────
PHENX_PRIMARY_SYSTEM   = {k: v[0] for k, v in PHENX_TO_SYSTEMS.items()}
MULTISYSTEM_PHENOTYPES = {k for k, v in PHENX_TO_SYSTEMS.items() if len(v) > 1}
VAGUE_PHENOTYPES       = {k for k, v in PHENX_TO_SYSTEMS.items()
                          if 'Systemic' in v and len(v) == 1}
PHENOTYPE_SPECIFICITY  = {k: 1.0 / len(v) for k, v in PHENX_TO_SYSTEMS.items()}
ALL_SYSTEMS            = sorted({s for systems in PHENX_TO_SYSTEMS.values() for s in systems})

ACUTE_KEYWORDS   = ['acute', 'sudden', 'initial encounter', 'injury', 'arrest',
                     'infarction', 'hemorrhage', 'rupture', 'fracture', 'crisis',
                     'poisoning', 'embolism', 'failure', 'trauma']
CHRONIC_KEYWORDS = ['chronic', 'long-term', 'late effect', 'sequela', 'history of',
                     'persistent', 'long-standing', 'disease', 'disorder']
VAGUE_KEYWORDS   = ['unspecified', 'other specified', 'ill-defined', 'abnormal findings',
                     'signs and symptoms', 'other general', 'nec', 'not elsewhere']


# ============================================================
# DATA-DRIVEN SYSTEM RELATEDNESS
# ============================================================

def compute_system_relatedness(df):
    """
    Empirical system-to-system relatedness from the corpus.
    Normalized co-occurrence rate of significant associations
    across body system pairs. Fully data-driven — no hardcoded weights.
    """
    sig = df[df['p.adjust'] < 0.05].copy()
    sig['src_sys'] = sig['startPhen_def'].map(PHENX_PRIMARY_SYSTEM).fillna('Unknown')
    sig['tgt_sys'] = sig['endPhenx_def'].map(PHENX_PRIMARY_SYSTEM).fillna('Unknown')

    cross = sig.groupby(['src_sys', 'tgt_sys']).size().reset_index(name='n_cross')
    sys_total = (pd.concat([sig['src_sys'].rename('system'),
                             sig['tgt_sys'].rename('system')])
                 .value_counts().rename('n_total').reset_index())
    sys_total_dict = dict(zip(sys_total['system'], sys_total['n_total']))

    raw = {}
    for _, row in cross.iterrows():
        sa, sb, n = row['src_sys'], row['tgt_sys'], row['n_cross']
        raw[(sa, sb)] = n / np.sqrt(sys_total_dict.get(sa, 1) * sys_total_dict.get(sb, 1))

    cross_scores = [v for (a, b), v in raw.items() if a != b]
    max_cross    = max(cross_scores) if cross_scores else 1.0
    normalized   = {k: (0.95 * v / max_cross) if k[0] != k[1] else 1.0
                    for k, v in raw.items()}

    sym = {(s, s): 1.0 for s in ALL_SYSTEMS + ['Unknown']}
    for (sa, sb), score in normalized.items():
        sym[(sa, sb)] = score
        if (sb, sa) not in normalized:
            sym[(sb, sa)] = score
    return sym


# ============================================================
# HELPERS
# ============================================================

def get_primary_system(condition):
    if pd.isna(condition):
        return 'Unknown'
    c = str(condition)
    if c in PHENX_PRIMARY_SYSTEM:
        return PHENX_PRIMARY_SYSTEM[c]
    cl = c.lower()
    if any(k in cl for k in ['procedure', 'procedures', 'transfusion',
                               'administration', 'vaccination', 'encounter for', 'implant']):
        return 'Procedure'
    if any(k in cl for k in ['pregnancy', 'maternal', 'liveborn', 'obstetric', 'puerperium']):
        return 'Pregnancy'
    if any(k in cl for k in ['cancer', 'malignant', 'neoplasm', 'tumor',
                               'lymphoma', 'leukemia', 'myeloma', 'sarcoma']):
        return 'Oncologic'
    if any(k in cl for k in ['neonatal', 'perinatal', 'newborn', 'birth']):
        return 'Pediatric/Neonatal'
    return 'Unknown'

def get_all_systems(condition):
    if pd.isna(condition):
        return ['Unknown']
    c = str(condition)
    return PHENX_TO_SYSTEMS.get(c, [get_primary_system(condition)])

def is_multisystem(condition):    return str(condition) in MULTISYSTEM_PHENOTYPES
def is_vague_condition(condition):
    if pd.isna(condition): return True
    c = str(condition)
    return c in VAGUE_PHENOTYPES or any(p in c.lower() for p in VAGUE_KEYWORDS)
def is_acute(condition):
    if pd.isna(condition): return False
    return any(k in str(condition).lower() for k in ACUTE_KEYWORDS)
def is_chronic(condition):
    if pd.isna(condition): return False
    return any(k in str(condition).lower() for k in CHRONIC_KEYWORDS)
def get_phenotype_specificity(condition):
    if pd.isna(condition): return 0.0
    c = str(condition)
    if c in PHENOTYPE_SPECIFICITY:
        score = PHENOTYPE_SPECIFICITY[c]
        return score * 0.3 if any(p in c.lower() for p in VAGUE_KEYWORDS) else score
    cl = c.lower()
    if any(v in cl for v in ['unspecified', 'other specified', 'abnormal findings']): return 0.1
    return 0.6 if len(c.split()) >= 4 else 0.4 if len(c.split()) >= 2 else 0.2


# ============================================================
# CLASS 1: CORE STATISTICAL SIGNAL
# ============================================================

def class1_statistical(df):
    f, rho, padj = pd.DataFrame(index=df.index), df['rho'], df['p.adjust']
    rho_abs = rho.abs()
    f['rho']             = rho
    f['rho_abs']         = rho_abs
    f['rho_squared']     = rho ** 2
    f['neg_log10_padj']  = -np.log10(padj.replace(0, 1e-300))
    f['signal_strength'] = rho_abs * f['neg_log10_padj']
    prior_odds           = 0.35 / 0.65
    posterior_odds       = prior_odds * (1 - padj) / (padj + 1e-300)
    f['posterior_prob']  = posterior_odds / (1 + posterior_odds)
    f['bayes_factor']    = np.log1p(rho_abs / (padj + 1e-300))
    f['is_sig_001']      = (padj < 0.001).astype(int)
    return f


# ============================================================
# CLASS 2: DIRECTIONAL & BIDIRECTIONAL
# ============================================================

def class2_directional(df):
    f        = pd.DataFrame(index=df.index)
    rev_idx  = df.set_index(['endPhenx_def', 'startPhen_def', 'duration'])[['rho', 'p.adjust']]
    keys     = list(zip(df['startPhen_def'], df['endPhenx_def'], df['duration']))
    rev_rho  = np.array([rev_idx['rho'].get(k, np.nan)      for k in keys])
    rev_padj = np.array([rev_idx['p.adjust'].get(k, np.nan) for k in keys])
    fwd_abs  = df['rho'].abs().values
    rev_abs  = np.where(np.isnan(rev_rho), 0.0, np.abs(rev_rho))

    f['has_reverse']              = (~np.isnan(rev_rho)).astype(int)
    f['rho_reverse_abs']          = rev_abs
    f['directional_dominance']    = (fwd_abs - rev_abs) / (fwd_abs + rev_abs + 1e-6)
    f['direction_confidence']     = fwd_abs * (1 - rev_abs / (fwd_abs + 1e-6))
    both_sig                      = ((df['p.adjust'].values < 0.05) &
                                     (~np.isnan(rev_padj)) & (rev_padj < 0.05))
    f['both_directions_sig']      = both_sig.astype(int)
    f['asymmetric_bidirectional'] = (both_sig & (fwd_abs > rev_abs * 1.5)).astype(int)
    f['forward_only']             = ((df['p.adjust'].values < 0.05) & np.isnan(rev_rho)).astype(int)
    f['rho_sign_concordance']     = (np.sign(df['rho'].values) *
                                     np.sign(np.where(np.isnan(rev_rho), 0, rev_rho)))
    phen_pairs = set(zip(df['startPhen_def'], df['endPhenx_def']))
    f['has_reverse_phen'] = [
        1 if (r['endPhenx_def'], r['startPhen_def']) in phen_pairs else 0
        for _, r in df[['startPhen_def', 'endPhenx_def']].iterrows()
    ]
    return f


# ============================================================
# CLASS 3: CROSS-DURATION TEMPORAL PROFILE
# ============================================================

def class3_temporal(df):
    """
    Pivots corpus by (start, end) to get rho profile across 4 duration windows.
    Since every (start, end, duration) is unique, aggfunc='first' does no aggregation.
    NaN = pair not observed at that duration window (not tested or filtered out).
    All downstream computations are NaN-safe.
    """
    f = pd.DataFrame(index=df.index)

    pivot_rho = df.pivot_table(
        index=['startPhen_def', 'endPhenx_def'],
        columns='duration', values='rho', aggfunc='first'
    ).reindex(columns=DURATIONS)

    pivot_sig = (df.pivot_table(
        index=['startPhen_def', 'endPhenx_def'],
        columns='duration', values='p.adjust', aggfunc='first'
    ).reindex(columns=DURATIONS) < 0.05).astype(float)

    keys    = df[['startPhen_def', 'endPhenx_def']].copy()
    rho_mat = keys.merge(pivot_rho.reset_index(),
                         on=['startPhen_def', 'endPhenx_def'], how='left')[DURATIONS].values
    sig_mat = keys.merge(pivot_sig.reset_index(),
                         on=['startPhen_def', 'endPhenx_def'], how='left')[DURATIONS].values
    sig_mat     = np.where(np.isnan(sig_mat), 0.0, sig_mat)  # absent = not significant
    rho_abs_mat = np.abs(rho_mat)
    n_observed  = (~np.isnan(rho_mat)).sum(axis=1)

    dur_ord = df['duration'].map(DURATION_ORDER).fillna(0).astype(int)
    f['duration_ordinal']     = dur_ord
    f['n_observed_durations'] = n_observed

    f['n_sig_durations']    = np.nansum(sig_mat, axis=1)
    f['prop_sig_durations'] = np.where(n_observed > 0,
                                        f['n_sig_durations'] / n_observed, 0.0)

    # Temporal slope (NaN if < 2 windows observed)
    x, x_c = np.array([1,2,3,4], dtype=float), None
    x_c = x - x.mean()

    def slope(row):
        valid = ~np.isnan(row)
        if valid.sum() < 2: return np.nan
        return np.dot(x_c[valid], row[valid]) / (np.dot(x_c[valid], x_c[valid]) + 1e-6)

    slopes = np.array([slope(rho_abs_mat[i]) for i in range(len(df))])
    f['rho_temporal_slope']          = np.nan_to_num(slopes, nan=0.0)
    f['rho_temporal_slope_observed'] = (~np.isnan(slopes)).astype(int)

    # Current rho vs pair's own history
    pair_mean = np.nanmean(rho_abs_mat, axis=1)
    pair_sd   = np.nanstd(rho_abs_mat,  axis=1)
    f['rho_z_vs_pair_own'] = np.where(
        (n_observed > 1) & (pair_sd > 0),
        (df['rho'].abs().values - pair_mean) / (pair_sd + 1e-6), 0.0)

    # Temporal range
    safe_max = np.nanmax(np.where(np.isnan(rho_abs_mat), -999, rho_abs_mat), axis=1)
    safe_min = np.nanmin(np.where(np.isnan(rho_abs_mat),  999, rho_abs_mat), axis=1)
    f['rho_temporal_range'] = np.where(n_observed >= 2, safe_max - safe_min, 0.0)

    # Peak duration
    safe_mat = np.where(np.isnan(rho_abs_mat), -999, rho_abs_mat)
    peak_idx = np.argmax(safe_mat, axis=1)
    f['peak_rho_duration'] = np.where(n_observed >= 1, peak_idx + 1, 0)
    f['is_peak_duration']  = (f['peak_rho_duration'] == dur_ord).astype(int)

    # Pattern flags (requires ≥ 3 observed windows)
    has_3 = n_observed >= 3
    f['pattern_early_peak'] = np.where(has_3,
        ((rho_abs_mat[:, 0] >= rho_abs_mat[:, 1]) &
         (rho_abs_mat[:, 1] >= rho_abs_mat[:, 2])).astype(int), 0)
    f['pattern_late_peak']  = np.where(has_3,
        ((rho_abs_mat[:, 3] >= rho_abs_mat[:, 2]) &
         (rho_abs_mat[:, 2] >= rho_abs_mat[:, 1])).astype(int), 0)
    f['pattern_plateau']    = (f['n_sig_durations'] == 4).astype(int)

    # Duration × rho interactions (always computable)
    f['rho_x_short_lag'] = df['rho'].abs() * (dur_ord == 1).astype(float)
    f['rho_x_long_lag']  = df['rho'].abs() * (dur_ord == 4).astype(float)
    f['rho_x_duration']  = df['rho'].abs() * dur_ord.astype(float)
    return f


# ============================================================
# CLASS 4: CORPUS-RELATIVE EMPIRICAL PRIORS
# ============================================================

def class4_empirical_priors(df):
    f   = pd.DataFrame(index=df.index)
    sig = df[df['p.adjust'] < 0.05].copy()

    src_rho   = df.groupby('startPhen_def')['rho'].agg(['mean','std']).rename(
                    columns={'mean':'src_mean','std':'src_sd'})
    tgt_rho   = df.groupby('endPhenx_def')['rho'].agg(['mean','std']).rename(
                    columns={'mean':'tgt_mean','std':'tgt_sd'})
    src_sig_r = (sig.groupby('startPhen_def').size() / df.groupby('startPhen_def').size()).fillna(0)
    tgt_sig_r = (sig.groupby('endPhenx_def').size() / df.groupby('endPhenx_def').size()).fillna(0)

    tmp = df[['startPhen_def','endPhenx_def','rho','p.adjust']].copy()
    tmp = tmp.merge(src_rho.reset_index(), on='startPhen_def', how='left')
    tmp = tmp.merge(tgt_rho.reset_index(), on='endPhenx_def',  how='left')
    tmp.index = df.index

    f['rho_z_vs_source']  = (df['rho'].abs() - tmp['src_mean'].abs()) / (tmp['src_sd'] + 1e-6)
    f['rho_z_vs_target']  = (df['rho'].abs() - tmp['tgt_mean'].abs()) / (tmp['tgt_sd'] + 1e-6)
    f['source_sig_rate']  = df['startPhen_def'].map(src_sig_r).fillna(0)
    f['target_sig_rate']  = df['endPhenx_def'].map(tgt_sig_r).fillna(0)

    expected = f['source_sig_rate'] * f['target_sig_rate']
    f['association_surprise'] = np.log1p(
        (df['p.adjust'] < 0.05).astype(float) / (expected + 1e-6))

    src_fan = sig.groupby('startPhen_def')['endPhenx_def'].nunique()
    tgt_fan = sig.groupby('endPhenx_def')['startPhen_def'].nunique()
    f['source_fan_out']     = df['startPhen_def'].map(src_fan).fillna(0)
    f['target_fan_in']      = df['endPhenx_def'].map(tgt_fan).fillna(0)
    f['source_selectivity'] = 1.0 / (np.log1p(f['source_fan_out']) + 1)
    f['target_sink_score']  = np.log1p(f['target_fan_in'])

    dur_rho = df.groupby('duration')['rho'].agg(['mean','std'])
    def rho_z_dur(row):
        d = row['duration']
        if d not in dur_rho.index: return 0.0
        return (abs(row['rho']) - dur_rho.loc[d,'mean']) / (dur_rho.loc[d,'std'] + 1e-6)
    f['rho_z_within_duration'] = df.apply(rho_z_dur, axis=1)
    return f


# ============================================================
# CLASS 5: CLINICAL ONTOLOGY
# ============================================================

def class5_clinical_ontology(df, system_relatedness):
    f     = pd.DataFrame(index=df.index)
    start, end = df['startPhen_def'], df['endPhenx_def']
    s_sys = start.apply(get_primary_system)
    e_sys = end.apply(get_primary_system)
    s_all = start.apply(get_all_systems)
    e_all = end.apply(get_all_systems)

    s_proc = s_sys == 'Procedure'
    e_proc = e_sys == 'Procedure'
    f['dx_to_dx'] = (~s_proc & ~e_proc).astype(int)
    f['px_to_dx'] = ( s_proc & ~e_proc).astype(int)
    f['dx_to_px'] = (~s_proc &  e_proc).astype(int)

    f['same_primary_system']      = (s_sys == e_sys).astype(int)
    f['different_primary_system'] = (s_sys != e_sys).astype(int)
    f['start_is_multisystem']     = start.apply(is_multisystem).astype(int)
    f['end_is_multisystem']       = end.apply(is_multisystem).astype(int)

    def sys_ov(sl, el):
        ss, es = set(sl), set(el)
        i, u = ss & es, ss | es
        return len(i), len(i)/len(u) if u else 0.0

    ovs = [sys_ov(s, e) for s, e in zip(s_all, e_all)]
    f['n_systems_overlap'] = [o[0] for o in ovs]
    f['system_jaccard']    = [o[1] for o in ovs]

    f['system_relatedness'] = [
        system_relatedness.get((s, e), system_relatedness.get((e, s), 0.0))
        for s, e in zip(s_sys, e_sys)]
    f['high_relatedness']     = (f['system_relatedness'] >= 0.6).astype(int)
    f['start_is_systemic']    = (s_sys == 'Systemic').astype(int)
    f['end_is_systemic']      = (e_sys == 'Systemic').astype(int)
    f['systemic_to_specific'] = (f['start_is_systemic'] & f['different_primary_system']).astype(int)

    s_spec = start.apply(get_phenotype_specificity)
    e_spec = end.apply(get_phenotype_specificity)
    f['start_specificity'] = s_spec
    f['end_specificity']   = e_spec
    f['avg_specificity']   = (s_spec + e_spec) / 2
    f['both_specific']     = ((s_spec > 0.5) & (e_spec > 0.5)).astype(int)
    f['start_is_vague']    = start.apply(is_vague_condition).astype(int)
    f['end_is_vague']      = end.apply(is_vague_condition).astype(int)
    f['any_vague']         = (f['start_is_vague'] | f['end_is_vague']).astype(int)

    s_acute   = start.apply(is_acute)
    e_acute   = end.apply(is_acute)
    s_chronic = start.apply(is_chronic)
    f['start_is_acute'] = s_acute.astype(int)
    f['end_is_acute']   = e_acute.astype(int)
    dur_ord = df['duration'].map(DURATION_ORDER).fillna(0).astype(int)
    f['acute_long_lag_implausible'] = (s_acute  & (dur_ord == 4)).astype(int)
    f['chronic_long_lag_plausible'] = (s_chronic & (dur_ord >= 3)).astype(int)
    return f


# ============================================================
# CLASS 6: NETWORK POSITION
# ============================================================

def build_reference_graph(df, p_threshold=0.05):
    sig = df[df['p.adjust'] < p_threshold]
    G   = nx.DiGraph()
    for _, row in sig.iterrows():
        src, tgt = row['startPhen_def'], row['endPhenx_def']
        w = -np.log10(row['p.adjust'] + 1e-300) * abs(row['rho'])
        if G.has_edge(src, tgt):
            if w > G[src][tgt]['weight']: G[src][tgt]['weight'] = w
        else:
            G.add_edge(src, tgt, weight=w)
    return G


def compute_graph_stats(G):
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    pagerank    = nx.pagerank(G, alpha=0.85, max_iter=200, weight='weight')
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    in_deg      = dict(G.in_degree())
    out_deg     = dict(G.out_degree())
    all_pr      = np.array(list(pagerank.values()))
    all_bw      = np.array(list(betweenness.values()))
    thresholds  = {'pr_p75': np.percentile(all_pr, 75),
                   'bw_p90': np.percentile(all_bw, 90)}
    trans_ent   = {}
    for node in G.nodes():
        edges   = list(G.out_edges(node, data=True))
        total_w = sum(d['weight'] for _,_,d in edges)
        if not edges or total_w == 0:
            trans_ent[node] = 0.0; continue
        probs = [d['weight']/total_w for _,_,d in edges]
        trans_ent[node] = -sum(p * np.log2(p + 1e-300) for p in probs)
    return {'pagerank': pagerank, 'betweenness': betweenness,
            'in_deg': in_deg, 'out_deg': out_deg,
            'transition_entropy': trans_ent, 'thresholds': thresholds}


def class6_network(df, G, gs):
    f        = pd.DataFrame(index=df.index)
    thr      = gs['thresholds']
    src_list = df['startPhen_def'].tolist()
    tgt_list = df['endPhenx_def'].tolist()
    get      = lambda d, k, dv=0.0: d.get(k, dv)

    src_out = np.array([get(gs['out_deg'], s, 0) for s in src_list], dtype=float)
    src_in  = np.array([get(gs['in_deg'],  s, 0) for s in src_list], dtype=float)
    tgt_out = np.array([get(gs['out_deg'], t, 0) for t in tgt_list], dtype=float)
    tgt_in  = np.array([get(gs['in_deg'],  t, 0) for t in tgt_list], dtype=float)

    f['source_driver_score']   = src_out / (src_in + src_out + 1)
    f['target_sink_score_net'] = tgt_in  / (tgt_in + tgt_out + 1)
    f['cascade_potential']     = f['source_driver_score'] * (tgt_out / (tgt_in + tgt_out + 1))
    f['terminal_event']        = f['source_driver_score'] * f['target_sink_score_net']

    src_pr = np.array([get(gs['pagerank'], s, 0.0) for s in src_list])
    tgt_pr = np.array([get(gs['pagerank'], t, 0.0) for t in tgt_list])
    f['target_pagerank']      = tgt_pr
    f['source_pagerank']      = src_pr
    f['target_high_pagerank'] = (tgt_pr > thr['pr_p75']).astype(int)

    src_bw = np.array([get(gs['betweenness'], s, 0.0) for s in src_list])
    tgt_bw = np.array([get(gs['betweenness'], t, 0.0) for t in tgt_list])
    f['target_betweenness']        = tgt_bw
    f['target_is_bridge']          = (tgt_bw > thr['bw_p90']).astype(int)
    f['source_is_bridge']          = (src_bw > thr['bw_p90']).astype(int)
    f['source_transition_entropy'] = [get(gs['transition_entropy'], s, 0.0) for s in src_list]
    f['target_transition_entropy'] = [get(gs['transition_entropy'], t, 0.0) for t in tgt_list]
    f['local_transition_prob']     = np.where(
        src_out > 0,
        [int(G.has_edge(s,t)) / max(get(gs['out_deg'], s, 1), 1)
         for s,t in zip(src_list, tgt_list)], 0.0)
    f['edge_in_reference']   = [int(G.has_edge(s,t)) for s,t in zip(src_list,tgt_list)]
    f['has_reciprocal_edge'] = [int(G.has_edge(t,s)) for s,t in zip(src_list,tgt_list)]
    return f


# ============================================================
# MAIN PIPELINE
# ============================================================

def extract_all_features(input_file, output_file, verbose=True):
    def log(msg):
        if verbose: print(msg)

    log("=" * 70)
    log("FEATURE EXTRACTION PIPELINE")
    log("=" * 70)

    log(f"\n[1/8] Loading {input_file} ...")
    df = pd.read_csv(input_file)
    log(f"      {len(df):,} rows × {df.shape[1]} columns")

    for col in ['startPhen_def', 'endPhenx_def', 'duration', 'rho', 'p.adjust']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    log("\n[2/8] Computing data-driven system relatedness ...")
    sys_rel = compute_system_relatedness(df)
    log(f"      {len(sys_rel)} system pairs")

    log("\n[3/8] Class 1 — Core Statistical Signal ...")
    c1 = class1_statistical(df);         log(f"      → {c1.shape[1]} features")

    log("\n[4/8] Class 2 — Directional & Bidirectional ...")
    c2 = class2_directional(df);         log(f"      → {c2.shape[1]} features")

    log("\n[5/8] Class 3 — Cross-Duration Temporal Profile ...")
    c3 = class3_temporal(df);            log(f"      → {c3.shape[1]} features")

    log("\n[6/8] Class 4 — Corpus-Relative Empirical Priors ...")
    c4 = class4_empirical_priors(df);    log(f"      → {c4.shape[1]} features")

    log("\n[7/8] Class 5 — Clinical Ontology ...")
    c5 = class5_clinical_ontology(df, sys_rel); log(f"      → {c5.shape[1]} features")

    log("\n[8/8] Building graph + Class 6 — Network Position ...")
    G   = build_reference_graph(df, p_threshold=0.05)
    gs  = compute_graph_stats(G)
    c6  = class6_network(df, G, gs);    log(f"      → {c6.shape[1]} features")

    log("\nCombining and saving ...")
    id_cols    = [c for c in ['sequence','startPhen_def','endPhenx_def','duration']
                  if c in df.columns]
    label_cols = [c for c in ['label'] if c in df.columns]

    out = pd.concat([df[id_cols + label_cols], c1, c2, c3, c4, c5, c6], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.to_csv(output_file, index=False)

    n_feat = out.shape[1] - len(id_cols) - len(label_cols)
    log(f"\n{'=' * 70}")
    log(f"Output:   {output_file}")
    log(f"Shape:    {out.shape[0]:,} rows × {out.shape[1]} columns")
    log(f"Features: {n_feat} (excluding identifiers and label)")
    log(f"{'=' * 70}")

    if 'label' in df.columns and verbose:
        num_cols = out.select_dtypes(include=[np.number]).columns
        lbl      = (out['label'] == out['label'].dropna().unique()[-1]).astype(int)
        corrs    = out[num_cols].corrwith(lbl).abs().sort_values(ascending=False)
        log("\nTop 15 feature correlations with label:")
        log(corrs.head(15).to_string())

    return out


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    
    INPUT_FILE  = 'data/corrs.csv'
    SUBSET_FILE = 'data/labels_10k.csv'
    VAL_FILE    = 'data/validation_dataset.csv'
    OUTPUT_FILE = 'data/features_gnn.csv'
    
    all_df = extract_all_features(INPUT_FILE, OUTPUT_FILE)
    
    subset_labels = pd.read_csv(SUBSET_FILE)[['sequence', 'duration', 'label']]
    subset_df = all_df.merge(
        subset_labels,
        on=['sequence', 'duration'],
        how='inner'
    )
    print(f'{subset_df.shape} after merging with subset labels')
    subset_df.to_csv(OUTPUT_FILE.replace('.csv', '_10k.csv'), index=False)
    
    val_labels = pd.read_csv(VAL_FILE)[['sequence', 'duration', 'label_jt', 'label_bp1']]

    # Keep all columns from all_df + add validation labels
    val_df = all_df.merge(
        val_labels,
        on=['sequence', 'duration'],
        how='inner'
    )
    print(f'{val_df.shape} after merging with validation labels')

    val_df.to_csv(OUTPUT_FILE.replace('.csv', '_validation.csv'), index=False)
