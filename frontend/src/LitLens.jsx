import { useState, useEffect, useRef, useCallback } from 'react';

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: INJECTED STYLES
   Only fonts, resets, keyframes, media queries, and pseudo-element styles
   go here. Everything else is inline CSS-in-JS.
   ═══════════════════════════════════════════════════════════════════════════ */
const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Lora:ital,wght@0,400;0,600;1,400&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:16px;-webkit-text-size-adjust:100%}
body{
  font-family:'Lora',Georgia,serif;
  background:#08090d;
  color:#e4e5e7;
  line-height:1.6;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
  overflow-x:hidden;
}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:3px}
::selection{background:#6366f1;color:#fff}
*:focus-visible{outline:2px solid #6366f1;outline-offset:2px}
input,textarea,button{font-family:inherit}

@keyframes fadeInUp{
  from{opacity:0;transform:translateY(12px)}
  to{opacity:1;transform:translateY(0)}
}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes shimmer{
  0%{background-position:-200% center}
  100%{background-position:200% center}
}
@keyframes pulseGlow{
  0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.25)}
  50%{box-shadow:0 0 14px 4px rgba(99,102,241,.25)}
}
@keyframes pulseDot{
  0%,100%{transform:scale(1);opacity:1}
  50%{transform:scale(1.5);opacity:.6}
}
@keyframes slideDown{
  from{opacity:0;transform:translateY(-16px)}
  to{opacity:1;transform:translateY(0)}
}
@keyframes fillBar{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes borderSolid{
  from{border-style:dashed}
  to{border-style:solid}
}

@media(max-width:1200px){
  .ll-layout{grid-template-columns:240px 1fr !important}
}
@media(max-width:900px){
  .ll-layout{grid-template-columns:1fr !important;grid-template-rows:auto 1fr !important}
  .ll-sidebar{border-right:none !important;border-bottom:1px solid rgba(255,255,255,.06) !important;max-height:260px !important;overflow-y:auto !important}
}
@media(max-width:600px){
  .ll-metrics{grid-template-columns:1fr 1fr !important}
  .ll-features{grid-template-columns:1fr !important}
  .ll-contra-cols{grid-template-columns:1fr !important}
}
`;

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: DESIGN TOKENS
   ═══════════════════════════════════════════════════════════════════════════ */
const T = {
  bgPrimary: '#08090d',
  bgSurface: '#111318',
  bgElevated: '#1a1c24',
  bgHover: '#1f2130',
  textPrimary: '#e4e5e7',
  textSecondary: '#8b8d94',
  textTertiary: '#52545c',
  accentIndigo: '#6366f1',
  accentIndigoHover: '#818cf8',
  accentIndigoGlow: 'rgba(99,102,241,.25)',
  accentGreen: '#22c55e',
  accentGreenDim: 'rgba(34,197,94,.12)',
  accentRed: '#ef4444',
  accentRedDim: 'rgba(239,68,68,.12)',
  accentAmber: '#f59e0b',
  accentAmberDim: 'rgba(245,158,11,.12)',
  accentBlue: '#3b82f6',
  accentBlueDim: 'rgba(59,130,246,.12)',
  borderSubtle: 'rgba(255,255,255,.06)',
  borderStrong: 'rgba(255,255,255,.12)',
  fontDisplay: "'Playfair Display',Georgia,serif",
  fontMono: "'IBM Plex Mono','Courier New',monospace",
  fontBody: "'Lora',Georgia,serif",
};

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: MOCK DATA
   ═══════════════════════════════════════════════════════════════════════════ */
const MOCK = {
  researchQuestion: 'What factors influence patient adherence to medication in chronic disease management?',
  domain: 'Public Health',
  report: { n_papers: 14, n_claims: 23, n_contradictions: 3, n_gaps: 5, executive_summary: 'Analysis of 14 papers spanning 2021–2023 reveals that medication adherence in chronic disease is driven by a complex interplay of socioeconomic barriers, medication complexity, and patient-provider communication quality. While digital interventions show short-term promise, evidence for sustained impact remains contested. Significant gaps persist around culturally tailored interventions, long-term longitudinal tracking, and the role of caregiver networks.' },
  papers: [
    { title: 'Patient Adherence in Chronic Disease: A Systematic Review', authors: 'Chen, Wang, Li', year: '2023' },
    { title: 'Digital Health Interventions for Medication Adherence', authors: 'Thompson, Baker', year: '2022' },
    { title: 'Socioeconomic Factors Affecting Treatment Compliance', authors: 'Rodriguez, Martinez', year: '2023' },
    { title: 'The Role of Patient-Provider Communication', authors: 'Johnson, Williams', year: '2021' },
    { title: 'Medication Complexity and Patient Outcomes', authors: 'Park, Kim, Lee', year: '2022' },
    { title: 'Behavioral Nudges in Healthcare: An RCT', authors: 'Anderson, Davis', year: '2023' },
    { title: 'Cultural Competency and Treatment Adherence', authors: 'Nguyen, Patel', year: '2022' },
    { title: 'Long-term Adherence Patterns in Diabetes Management', authors: 'Brown, Taylor', year: '2021' },
    { title: 'Self-Efficacy and Health Behavior Change', authors: 'Garcia, Lopez', year: '2023' },
    { title: 'Insurance Coverage and Medication Access', authors: 'Wilson, Clark', year: '2022' },
    { title: 'Telehealth Impact on Chronic Disease Management', authors: 'Lee, Zhang', year: '2023' },
    { title: 'Side Effect Burden and Treatment Discontinuation', authors: 'Miller, Jones', year: '2021' },
    { title: 'Family Support Systems in Patient Adherence', authors: 'Davis, Moore', year: '2022' },
    { title: 'AI-Powered Adherence Monitoring: A Pilot Study', authors: 'Kumar, Singh', year: '2023' },
  ],
  claims: [
    { claim_text: 'Medication complexity significantly reduces adherence rates in multi-drug regimens', source_paper: 'Park, Kim, Lee (2022)', claim_type: 'finding', evidence_strength: 'strong', theme: 'Medication Factors' },
    { claim_text: 'Digital reminder systems improve short-term adherence by 15-23%', source_paper: 'Thompson, Baker (2022)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Digital Interventions' },
    { claim_text: 'Socioeconomic status is the strongest independent predictor of non-adherence', source_paper: 'Rodriguez, Martinez (2023)', claim_type: 'finding', evidence_strength: 'strong', theme: 'Socioeconomic Barriers' },
    { claim_text: 'Patient-provider communication quality correlates with r=0.42 to adherence', source_paper: 'Johnson, Williams (2021)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Communication' },
    { claim_text: 'Self-efficacy mediates the relationship between health literacy and adherence behavior', source_paper: 'Garcia, Lopez (2023)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Psychological Factors' },
    { claim_text: 'Side effect burden accounts for 34% of voluntary treatment discontinuation', source_paper: 'Miller, Jones (2021)', claim_type: 'finding', evidence_strength: 'strong', theme: 'Medication Factors' },
    { claim_text: 'Insurance coverage gaps create cyclical non-adherence patterns', source_paper: 'Wilson, Clark (2022)', claim_type: 'finding', evidence_strength: 'strong', theme: 'Socioeconomic Barriers' },
    { claim_text: 'Family support networks improve adherence by 18% in elderly populations', source_paper: 'Davis, Moore (2022)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Social Support' },
    { claim_text: 'Telehealth consultations have no significant effect on long-term adherence', source_paper: 'Lee, Zhang (2023)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Digital Interventions' },
    { claim_text: 'Behavioral nudge interventions show diminishing returns after 6 months', source_paper: 'Anderson, Davis (2023)', claim_type: 'finding', evidence_strength: 'strong', theme: 'Behavioral Interventions' },
    { claim_text: 'Culturally tailored interventions outperform generic programs by 2.1x', source_paper: 'Nguyen, Patel (2022)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Cultural Factors' },
    { claim_text: 'AI-powered monitoring achieves 91% detection rate for non-adherence events', source_paper: 'Kumar, Singh (2023)', claim_type: 'finding', evidence_strength: 'weak', theme: 'Digital Interventions' },
    { claim_text: 'Simplified once-daily regimens improve adherence by 26% over multi-dose', source_paper: 'Chen, Wang, Li (2023)', claim_type: 'finding', evidence_strength: 'strong', theme: 'Medication Factors' },
    { claim_text: 'Long-term diabetes adherence declines 8% annually without intervention', source_paper: 'Brown, Taylor (2021)', claim_type: 'finding', evidence_strength: 'moderate', theme: 'Temporal Patterns' },
  ],
  contradictions: [
    {
      topic: 'Sustained efficacy of digital reminder systems',
      paper_a: { title: 'Digital Health Interventions for Medication Adherence', year: '2022', position: 'Digital reminders produce statistically significant and sustained adherence improvements over 12 months (p<0.01).' },
      paper_b: { title: 'Behavioral Nudges in Healthcare: An RCT', year: '2023', position: 'Digital nudge effects diminish to non-significance after 6 months, suggesting habituation rather than behavior change.' },
      severity: 'partial disagreement',
      possible_explanation: 'Thompson (2022) measured self-reported adherence while Anderson (2023) used pharmacy refill data. The discrepancy may reflect reporting bias in self-reported measures.',
    },
    {
      topic: 'Primary barrier to medication adherence',
      paper_a: { title: 'Socioeconomic Factors Affecting Treatment Compliance', year: '2023', position: 'Cost and insurance coverage are the dominant barriers, explaining 41% of variance in adherence.' },
      paper_b: { title: 'Medication Complexity and Patient Outcomes', year: '2022', position: 'Regimen complexity is the strongest modifiable predictor, independent of cost considerations.' },
      severity: 'direct contradiction',
      possible_explanation: 'Rodriguez (2023) studied a low-income urban population where cost barriers dominate, while Park (2022) studied insured patients in an integrated health system where cost is controlled.',
    },
    {
      topic: 'Impact of telehealth on chronic disease management',
      paper_a: { title: 'Telehealth Impact on Chronic Disease Management', year: '2023', position: 'Telehealth consultations improve medication adherence through increased access and convenience.' },
      paper_b: { title: 'Long-term Adherence Patterns in Diabetes Management', year: '2021', position: 'No significant difference in adherence rates between telehealth and in-person cohorts at 18-month follow-up.' },
      severity: 'different context',
      possible_explanation: 'Lee (2023) evaluated a post-COVID telehealth-native population, while Brown (2021) studied patients transitioned mid-treatment to telehealth during the pandemic — different adoption contexts.',
    },
  ],
  methodology: [
    { paper_title: 'Chen, Wang, Li (2023)', year: '2023', study_design: 'Systematic Review', sample_size: '48 studies', data_collection_method: 'Database search', statistical_methods: 'Meta-analysis', key_strength: 'Comprehensive scope', key_limitation: 'Heterogeneous outcome measures' },
    { paper_title: 'Thompson, Baker (2022)', year: '2022', study_design: 'RCT', sample_size: '412', data_collection_method: 'App usage + self-report', statistical_methods: 'Mixed-effects models', key_strength: 'Randomized design', key_limitation: 'Self-reported adherence' },
    { paper_title: 'Rodriguez, Martinez (2023)', year: '2023', study_design: 'Cross-sectional', sample_size: '2,847', data_collection_method: 'Survey + claims data', statistical_methods: 'Multivariate regression', key_strength: 'Large diverse sample', key_limitation: 'Cross-sectional design' },
    { paper_title: 'Johnson, Williams (2021)', year: '2021', study_design: 'Observational cohort', sample_size: '634', data_collection_method: 'Clinic recordings + surveys', statistical_methods: 'Path analysis', key_strength: 'Objective communication coding', key_limitation: 'Single health system' },
    { paper_title: 'Park, Kim, Lee (2022)', year: '2022', study_design: 'Retrospective cohort', sample_size: '1,203', data_collection_method: 'EHR + pharmacy data', statistical_methods: 'Cox regression', key_strength: 'Objective refill data', key_limitation: 'Insured population only' },
    { paper_title: 'Anderson, Davis (2023)', year: '2023', study_design: 'RCT', sample_size: '518', data_collection_method: 'Pharmacy refill + MEMS caps', statistical_methods: 'ITT analysis', key_strength: 'Objective measurement', key_limitation: '6-month follow-up only' },
    { paper_title: 'Nguyen, Patel (2022)', year: '2022', study_design: 'Quasi-experimental', sample_size: '186', data_collection_method: 'Interviews + pill counts', statistical_methods: 'DID analysis', key_strength: 'Culturally specific design', key_limitation: 'Small sample, single ethnicity' },
    { paper_title: 'Brown, Taylor (2021)', year: '2021', study_design: 'Longitudinal cohort', sample_size: '923', data_collection_method: 'HbA1c + refill data', statistical_methods: 'GEE models', key_strength: '18-month follow-up', key_limitation: 'Diabetes only' },
    { paper_title: 'Garcia, Lopez (2023)', year: '2023', study_design: 'Cross-sectional', sample_size: '445', data_collection_method: 'Validated scales', statistical_methods: 'SEM', key_strength: 'Validated instruments', key_limitation: 'Self-reported, single timepoint' },
    { paper_title: 'Wilson, Clark (2022)', year: '2022', study_design: 'Retrospective cohort', sample_size: '5,112', data_collection_method: 'Insurance claims', statistical_methods: 'Interrupted time series', key_strength: 'Very large sample', key_limitation: 'Administrative data only' },
    { paper_title: 'Lee, Zhang (2023)', year: '2023', study_design: 'Prospective cohort', sample_size: '342', data_collection_method: 'Telehealth logs + surveys', statistical_methods: 'Propensity matching', key_strength: 'Real-world telehealth data', key_limitation: 'Urban setting only' },
    { paper_title: 'Miller, Jones (2021)', year: '2021', study_design: 'Mixed methods', sample_size: '267', data_collection_method: 'Surveys + interviews', statistical_methods: 'Thematic + logistic regression', key_strength: 'Rich qualitative data', key_limitation: 'Convenience sample' },
    { paper_title: 'Davis, Moore (2022)', year: '2022', study_design: 'Quasi-experimental', sample_size: '198', data_collection_method: 'Family interviews + refill data', statistical_methods: 'Multilevel modeling', key_strength: 'Dyadic analysis', key_limitation: 'Selection bias' },
    { paper_title: 'Kumar, Singh (2023)', year: '2023', study_design: 'Pilot RCT', sample_size: '78', data_collection_method: 'Sensor data + EHR', statistical_methods: 'Descriptive + ROC analysis', key_strength: 'Novel technology', key_limitation: 'Very small sample, pilot only' },
  ],
  methodologyPatterns: [
    '64% of studies rely on self-reported adherence measures — a significant source of bias',
    'Only 2 of 14 studies (14%) follow patients beyond 12 months',
    'RCTs represent just 21% of the evidence base; most studies are observational',
    'Sample sizes vary from 78 to 5,112 — high heterogeneity limits direct comparison',
  ],
  evidence_scores: [
    { claim_text: 'Medication complexity significantly reduces adherence rates', evidence_score: 92, supporting_papers: 8, evidence_quality: 'Multiple large studies with objective measures', contradicted: false, flag: 'well-supported' },
    { claim_text: 'Socioeconomic status is the strongest independent predictor of non-adherence', evidence_score: 87, supporting_papers: 7, evidence_quality: 'Large samples, consistent across populations', contradicted: false, flag: 'well-supported' },
    { claim_text: 'Side effect burden accounts for 34% of voluntary discontinuation', evidence_score: 81, supporting_papers: 6, evidence_quality: 'Mixed-methods confirmation', contradicted: false, flag: 'well-supported' },
    { claim_text: 'Insurance coverage gaps create cyclical non-adherence patterns', evidence_score: 79, supporting_papers: 5, evidence_quality: 'Large claims data analysis', contradicted: false, flag: 'well-supported' },
    { claim_text: 'Simplified once-daily regimens improve adherence by 26%', evidence_score: 76, supporting_papers: 6, evidence_quality: 'Meta-analytic evidence', contradicted: false, flag: 'well-supported' },
    { claim_text: 'Digital reminders improve short-term adherence by 15-23%', evidence_score: 58, supporting_papers: 5, evidence_quality: 'RCT evidence but contradicted on duration', contradicted: true, flag: 'needs more evidence' },
    { claim_text: 'Family support networks improve adherence by 18% in elderly', evidence_score: 52, supporting_papers: 3, evidence_quality: 'Limited to elderly, quasi-experimental', contradicted: false, flag: 'needs more evidence' },
    { claim_text: 'Patient-provider communication quality correlates r=0.42', evidence_score: 48, supporting_papers: 4, evidence_quality: 'Single-system observational', contradicted: false, flag: 'needs more evidence' },
    { claim_text: 'Culturally tailored interventions outperform generic by 2.1x', evidence_score: 38, supporting_papers: 2, evidence_quality: 'Small quasi-experimental studies', contradicted: false, flag: 'widely cited but poorly evidenced' },
    { claim_text: 'AI-powered monitoring achieves 91% detection rate', evidence_score: 18, supporting_papers: 1, evidence_quality: 'Single pilot study, n=78', contradicted: false, flag: 'widely cited but poorly evidenced' },
  ],
  gap_analysis: `Given your research question 'What factors influence patient adherence to medication in chronic disease management?', the following gaps emerge from the literature:

### 1. Longitudinal Behavioral Trajectories
No study tracks adherence behavior changes beyond 18 months. The field lacks understanding of how adherence patterns evolve over the full chronic disease lifecycle — particularly transition points like retirement, relocation, or loss of a caregiver.

**Why it matters:** Your research question addresses "factors that influence adherence," but current evidence captures only snapshots. The factors at diagnosis may differ entirely from those at year five.

**Suggested design:** A 3–5 year prospective cohort study using pharmacy refill data and biannual interviews, stratified by disease duration at enrollment.

### 2. Culturally Specific Intervention Mechanisms
While Nguyen & Patel (2022) demonstrate that culturally tailored programs outperform generic ones, the mechanism is unexplored. What specific cultural elements drive the effect — language, trust, family involvement models, or health belief alignment?

**Why it matters:** Without mechanistic understanding, interventions cannot be adapted across cultures. Your research could clarify which cultural components are transferable.

**Suggested design:** A dismantling RCT comparing full culturally tailored intervention versus individual components across 3+ cultural groups (n≥150 per group).

### 3. Caregiver Burden and Adherence Spillover
Davis & Moore (2022) examine family support's positive effects but no study investigates the reverse: how caregiver burden, burnout, or competing demands affect the patient's adherence over time.

**Why it matters:** Caregiver-dependent adherence is fragile. Understanding this dynamic is essential for sustainable long-term management strategies.

**Suggested design:** Dyadic longitudinal study tracking both patient adherence and caregiver wellbeing measures over 24 months.

### 4. Intersection of Cost and Complexity
Rodriguez (2023) and Park (2022) disagree on whether cost or complexity is the dominant barrier, yet no study examines their interaction. Patients facing both high cost and high complexity likely experience compounded non-adherence — but this interaction term is absent from all models reviewed.

**Why it matters:** This directly addresses your research question by identifying whether barriers are additive or multiplicative.

**Suggested design:** Factorial survey experiment or natural experiment using insurance formulary changes as an exogenous shock to cost while measuring regimen complexity.

### 5. Real-World Digital Intervention Persistence
AI monitoring (Kumar, 2023) and digital reminders (Thompson, 2022) are studied in controlled settings. No study evaluates whether patients continue using these tools voluntarily after a study ends, or whether health systems can sustain them at scale.

**Why it matters:** Efficacy without adoption is irrelevant. The sustainability question is the true bottleneck for digital adherence interventions.

**Suggested design:** Post-trial follow-up study (12+ months after RCT completion) measuring continued app usage, adherence, and willingness-to-pay.`,

  literature_review_draft: `## 1. Introduction

Medication adherence in chronic disease management remains one of the most consequential yet intractable challenges in contemporary healthcare. The World Health Organization has identified non-adherence as the primary cause of suboptimal clinical outcomes in chronic conditions, with estimated adherence rates hovering around 50% in developed nations (Chen, Wang & Li, 2023). This review synthesizes findings from 14 studies published between 2021 and 2023 to examine the multifactorial determinants of medication adherence, areas of scholarly consensus and contestation, and persistent gaps that warrant further investigation. The central question guiding this analysis is: what factors influence patient adherence to medication in chronic disease management?

## 2. Thematic Analysis

### 2.1 Structural and Socioeconomic Determinants

A robust body of evidence identifies socioeconomic factors as foundational determinants of adherence behavior. Rodriguez and Martinez (2023) demonstrated that income level, insurance status, and medication cost collectively explain 41% of variance in adherence rates across a diverse urban sample (n=2,847). This structural framing is reinforced by Wilson and Clark (2022), whose interrupted time-series analysis of 5,112 insurance claims revealed that coverage gaps produce cyclical non-adherence patterns that persist even after coverage is restored. The evidence strongly suggests that adherence is not merely a behavioral choice but a structurally constrained outcome.

### 2.2 Medication Regimen Characteristics

The complexity of medication regimens has emerged as a critical modifiable factor. Park, Kim, and Lee (2022) found that each additional daily medication reduces adherence probability by 12% (HR=0.88, 95% CI: 0.83–0.93), while Chen, Wang, and Li (2023) report in their meta-analysis that simplified once-daily regimens improve adherence by 26% over multi-dose alternatives. Miller and Jones (2021) further demonstrated that side-effect burden accounts for 34% of voluntary treatment discontinuation, highlighting that the pharmacological properties of the regimen itself shape adherence behavior independently of patient motivation.

### 2.3 Psychosocial and Relational Factors

Beyond structural determinants, psychosocial dynamics play a significant mediating role. Garcia and Lopez (2023) established through structural equation modeling that self-efficacy mediates the relationship between health literacy and adherence behavior, suggesting that knowledge alone is insufficient without accompanying confidence in one's capacity to manage a regimen. Johnson and Williams (2021) found a moderate correlation (r=0.42) between objectively coded patient-provider communication quality and subsequent adherence. Davis and Moore (2022) extended this relational lens to family systems, demonstrating that structured family support networks improve adherence by 18% in elderly populations.

### 2.4 Digital and Behavioral Interventions

Digital health interventions represent the most actively debated domain. Thompson and Baker (2022) reported that app-based reminder systems improve adherence by 15–23% in their 12-month RCT, while Kumar and Singh (2023) demonstrated 91% detection accuracy for non-adherence events using AI-powered monitoring. However, the sustainability of these effects remains contested (see Section 4). Nguyen and Patel (2022) provide preliminary evidence that culturally tailored interventions outperform generic digital programs by a factor of 2.1, though this finding rests on a small quasi-experimental sample.

## 3. Methodological Considerations

The methodological landscape reveals important limitations. Sixty-four percent of reviewed studies rely on self-reported adherence measures, introducing systematic overestimation bias. Only two studies employ follow-up periods exceeding 12 months (Brown & Taylor, 2021; Wilson & Clark, 2022), limiting understanding of long-term adherence trajectories. RCTs constitute just 21% of the evidence base, with the majority of studies employing observational designs that constrain causal inference. Sample sizes vary from 78 (Kumar & Singh, 2023) to 5,112 (Wilson & Clark, 2022), and this heterogeneity limits cross-study comparison.

## 4. Contradictions and Debates

Several genuine disagreements emerge. Most notably, Thompson and Baker (2022) and Anderson and Davis (2023) reach opposing conclusions regarding the sustained efficacy of digital reminders — a discrepancy likely attributable to differences in outcome measurement (self-report versus pharmacy refill data). Rodriguez and Martinez (2023) and Park, Kim, and Lee (2022) disagree on whether cost or complexity is the dominant adherence barrier, though this likely reflects population differences rather than a true contradiction. The telehealth debate between Lee and Zhang (2023) and Brown and Taylor (2021) appears driven by differing adoption contexts rather than conflicting evidence.

## 5. Research Gaps and Future Directions

Despite the breadth of this literature, critical gaps remain. No study tracks adherence trajectories beyond 18 months, leaving the long-term evolution of adherence behavior poorly understood. The mechanisms underlying cultural tailoring's effectiveness are unexplored. The interaction between cost and complexity barriers — potentially multiplicative rather than additive — has not been modeled. Caregiver burden's effect on patient adherence is absent from the literature, as is evidence on the real-world persistence of digital interventions after controlled study conditions end.

## 6. Conclusion

The literature reveals that medication adherence in chronic disease is shaped by the convergence of structural barriers, regimen characteristics, psychosocial resources, and intervention design. While the field has achieved consensus on several key predictors, important contradictions persist around digital intervention sustainability and the relative weight of cost versus complexity barriers. Significant gaps — particularly around longitudinal trajectories, cultural mechanisms, and caregiver dynamics — present clear opportunities for future research. These gaps speak directly to the research question of what factors influence adherence: the answer is that many factors are well-documented in isolation, but their interactions, evolution over time, and cultural contingencies remain poorly understood.`,

  agent_history: [
    'Paper Ingestion: Extracted metadata from 14/14 papers',
    'Claim Extraction: Identified 23 claims across 14 papers',
    'Contradiction Detector: Found 3 contradictions/disagreements',
    'Methodology Comparator: Compared 14 studies, found 4 patterns',
    'Evidence Scorer: Scored 14 claims',
    'Gap Analyzer: Identified research gaps and opportunities',
    'Literature Review Writer: Drafted thematic literature review',
    'Report Generator: Assembled overview — 14 papers, 23 claims, 3 contradictions, ~5 gaps',
  ],
};

const ANALYSIS_STEPS = [
  { agent: 'Ingesting papers', duration: 1800, tab: null },
  { agent: 'Extracting claims', duration: 2200, tab: null },
  { agent: 'Detecting contradictions', duration: 2800, tab: 1 },
  { agent: 'Comparing methodologies', duration: 1600, tab: 2 },
  { agent: 'Scoring evidence', duration: 1400, tab: 3 },
  { agent: 'Analyzing research gaps', duration: 2600, tab: 4 },
  { agent: 'Writing literature review', duration: 3200, tab: 5 },
  { agent: 'Generating report', duration: 1000, tab: 0 },
];

const TAB_LABELS = ['Overview', 'Contradictions', 'Methodology', 'Evidence', 'Gaps', 'Draft'];

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: HOOKS & UTILITIES
   ═══════════════════════════════════════════════════════════════════════════ */
function useBreakpoint() {
  const [w, setW] = useState(typeof window !== 'undefined' ? window.innerWidth : 1400);
  useEffect(() => {
    const h = () => setW(window.innerWidth);
    window.addEventListener('resize', h);
    return () => window.removeEventListener('resize', h);
  }, []);
  if (w > 1200) return 'desktop';
  if (w > 900) return 'tablet';
  if (w > 600) return 'mobile';
  return 'small';
}

function downloadCSV(data, filename) {
  if (!data.length) return;
  const headers = Object.keys(data[0]);
  const rows = data.map(r => headers.map(h => `"${String(r[h] || '').replace(/"/g, '""')}"`).join(','));
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function downloadTxt(text, filename) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function scoreColor(s) {
  if (s >= 70) return T.accentGreen;
  if (s >= 40) return T.accentAmber;
  return T.accentRed;
}

function scoreGlow(s) {
  if (s >= 70) return `0 0 12px ${T.accentGreenDim}`;
  if (s >= 40) return `0 0 12px ${T.accentAmberDim}`;
  return `0 0 12px ${T.accentRedDim}`;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: SMALL COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */
function AnimatedNumber({ value, duration = 1000, active = true }) {
  const [display, setDisplay] = useState(0);
  const raf = useRef(null);
  useEffect(() => {
    if (!active) { setDisplay(0); return; }
    const target = typeof value === 'number' ? value : parseInt(value) || 0;
    let start = null;
    function step(ts) {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(Math.round(target * eased));
      if (p < 1) raf.current = requestAnimationFrame(step);
    }
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [value, duration, active]);
  return <>{display}</>;
}

function Pill({ text, onRemove, delay = 0 }) {
  const maxLen = 22;
  const label = text.length > maxLen ? text.slice(0, maxLen - 2) + '…' : text;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      background: T.bgElevated, border: `1px solid ${T.borderStrong}`,
      borderRadius: 100, padding: '4px 10px 4px 12px',
      fontFamily: T.fontMono, fontSize: 11, color: T.textSecondary,
      animation: `fadeInUp 300ms ease ${delay}ms both`,
      transition: 'all 150ms ease',
      cursor: 'default',
    }}>
      {label}
      {onRemove && (
        <button onClick={(e) => { e.stopPropagation(); onRemove(); }} aria-label={`Remove ${text}`}
          style={{
            background: 'none', border: 'none', color: T.textTertiary,
            cursor: 'pointer', padding: 0, fontSize: 13, lineHeight: 1,
            transition: 'color 150ms', fontFamily: T.fontMono,
          }}
          onMouseEnter={e => e.target.style.color = T.accentRed}
          onMouseLeave={e => e.target.style.color = T.textTertiary}
        >✕</button>
      )}
    </span>
  );
}

function SeverityBadge({ severity }) {
  const map = {
    'direct contradiction': { label: 'DIRECT', bg: T.accentRedDim, color: T.accentRed },
    'partial disagreement': { label: 'PARTIAL', bg: T.accentAmberDim, color: T.accentAmber },
    'different context': { label: 'CONTEXT', bg: T.accentBlueDim, color: T.accentBlue },
  };
  const s = map[severity] || { label: severity?.toUpperCase() || '?', bg: T.bgElevated, color: T.textTertiary };
  return (
    <span style={{
      fontFamily: T.fontMono, fontSize: 10, fontWeight: 600, letterSpacing: 1,
      background: s.bg, color: s.color, padding: '3px 10px', borderRadius: 100,
    }}>{s.label}</span>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: UPLOAD ZONE
   ═══════════════════════════════════════════════════════════════════════════ */
function UploadZone({ files, setFiles, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const hasFiles = files.length > 0;

  const borderColor = dragging ? T.accentGreen : hasFiles ? T.accentGreen : T.borderStrong;
  const borderStyle = dragging ? 'solid' : hasFiles ? 'solid' : 'dashed';

  return (
    <div
      onDragOver={e => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        e.preventDefault(); setDragging(false);
        if (disabled) return;
        const dropped = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
        if (dropped.length) setFiles(prev => [...prev, ...dropped]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: `2px ${borderStyle} ${borderColor}`,
        borderRadius: 8, padding: hasFiles ? 12 : 28,
        textAlign: 'center', cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 200ms ease',
        opacity: disabled ? 0.3 : 1,
        transform: dragging ? 'scale(1.01)' : 'scale(1)',
        boxShadow: dragging ? `0 0 16px ${T.accentGreenDim}` : 'none',
        background: dragging ? 'rgba(34,197,94,.04)' : 'transparent',
      }}
      aria-label="Upload research papers"
    >
      <input ref={inputRef} type="file" multiple accept=".pdf" hidden
        onChange={e => { setFiles(prev => [...prev, ...Array.from(e.target.files)]); e.target.value = ''; }} />
      {!hasFiles ? (
        <div>
          <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
            Upload Research Papers (PDF)
          </div>
          <div style={{ fontSize: 13, color: T.textSecondary }}>Drop files here or click to browse</div>
        </div>
      ) : (
        <div>
          <div style={{
            fontFamily: T.fontMono, fontSize: 12, color: T.accentGreen, marginBottom: 8,
            animation: 'fadeIn 300ms ease',
          }}>
            {files.length} paper{files.length !== 1 ? 's' : ''} uploaded
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center' }}>
            {files.map((f, i) => (
              <Pill key={`${f.name}-${i}`} text={f.name} delay={i * 60}
                onRemove={() => setFiles(prev => prev.filter((_, j) => j !== i))} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: ANALYZE BUTTON
   ═══════════════════════════════════════════════════════════════════════════ */
function AnalyzeButton({ canRun, appState, onClick }) {
  const isDisabled = !canRun || appState === 'analyzing';
  const isLoading = appState === 'analyzing';
  const isDone = appState === 'results';
  const [hover, setHover] = useState(false);

  let bg, color, text, cursor, shadow;
  if (isLoading) {
    bg = `linear-gradient(90deg, ${T.accentIndigo} 0%, ${T.accentIndigoHover} 50%, ${T.accentIndigo} 100%)`;
    color = '#fff'; text = 'Analyzing…'; cursor = 'wait'; shadow = 'none';
  } else if (isDone) {
    bg = T.accentGreen; color = '#fff'; text = '✓ Complete'; cursor = 'default'; shadow = 'none';
  } else if (isDisabled) {
    bg = T.bgElevated; color = T.textTertiary; text = 'Analyze Literature →'; cursor = 'not-allowed'; shadow = 'none';
  } else {
    bg = T.accentIndigo; color = '#fff'; text = 'Analyze Literature →';
    cursor = 'pointer'; shadow = hover ? `0 0 20px ${T.accentIndigoGlow}` : 'none';
  }

  return (
    <button
      onClick={() => canRun && appState !== 'analyzing' && onClick()}
      disabled={isDisabled}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: '100%', height: 48, border: 'none', borderRadius: 6,
        background: bg, color, fontFamily: T.fontMono, fontSize: 13,
        fontWeight: 600, letterSpacing: 0.5, cursor, boxShadow: shadow,
        transition: 'all 150ms ease',
        transform: hover && !isDisabled && !isLoading ? 'scale(1)' : 'scale(1)',
        backgroundSize: isLoading ? '200% 100%' : 'auto',
        animation: isLoading ? 'shimmer 1.5s infinite linear' : 'none',
      }}
      onMouseDown={e => { if (!isDisabled) e.target.style.transform = 'scale(0.98)'; }}
      onMouseUp={e => { e.target.style.transform = 'scale(1)'; }}
      aria-label="Start literature analysis"
    >
      {text}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: SIDEBAR
   ═══════════════════════════════════════════════════════════════════════════ */
function Sidebar({ files, setFiles, rq, setRq, domain, setDomain, onAnalyze, appState, step, tabReady, cost }) {
  const isAnalyzing = appState === 'analyzing';
  const inputStyle = {
    width: '100%', background: T.bgElevated, border: `1px solid ${T.borderStrong}`,
    borderRadius: 4, padding: '10px 12px', color: T.textPrimary,
    fontFamily: T.fontBody, fontSize: 14, outline: 'none',
    transition: 'border-color 150ms',
  };
  const labelStyle = {
    fontFamily: T.fontMono, fontSize: 10, textTransform: 'uppercase',
    letterSpacing: 1.5, color: T.textTertiary, marginBottom: 6, display: 'block',
  };

  return (
    <aside className="ll-sidebar" style={{
      background: T.bgSurface, borderRight: `1px solid ${T.borderSubtle}`,
      padding: 24, display: 'flex', flexDirection: 'column', gap: 20,
      overflowY: 'auto', position: 'relative',
    }}>
      <div>
        <div style={{ fontFamily: T.fontDisplay, fontSize: 28, fontWeight: 700, color: T.textPrimary }}>LitLens</div>
        <div style={{ fontFamily: T.fontBody, fontSize: 14, fontStyle: 'italic', color: T.textSecondary, marginTop: 2 }}>
          Know your literature, faster
        </div>
      </div>

      <div style={{ height: 1, background: T.borderSubtle }} />

      <div style={{ opacity: isAnalyzing ? 0.3 : 1, pointerEvents: isAnalyzing ? 'none' : 'auto', transition: 'opacity 300ms', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <UploadZone files={files} setFiles={setFiles} disabled={isAnalyzing} />

        <div>
          <label style={labelStyle}>Your Research Question</label>
          <input value={rq} onChange={e => setRq(e.target.value)} style={inputStyle}
            placeholder="e.g., What factors influence patient adherence to medication?"
            onFocus={e => e.target.style.borderColor = T.accentIndigo}
            onBlur={e => e.target.style.borderColor = T.borderStrong}
            aria-label="Research question" />
        </div>

        <div>
          <label style={labelStyle}>Research Domain</label>
          <input value={domain} onChange={e => setDomain(e.target.value)} style={inputStyle}
            placeholder="e.g., Public Health, Machine Learning"
            onFocus={e => e.target.style.borderColor = T.accentIndigo}
            onBlur={e => e.target.style.borderColor = T.borderStrong}
            aria-label="Research domain" />
        </div>

        <AnalyzeButton canRun={files.length >= 2 && rq.trim().length > 0} appState={appState} onClick={onAnalyze} />
        {files.length > 0 && files.length < 2 && (
          <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.accentAmber }}>Upload at least 2 papers.</div>
        )}
      </div>

      {isAnalyzing && (
        <div style={{ animation: 'fadeIn 300ms ease', display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textSecondary, textTransform: 'uppercase', letterSpacing: 1 }}>
            Agent {step + 1} of 8
          </div>
          <div style={{ fontFamily: T.fontMono, fontSize: 13, color: T.textPrimary }}>
            {ANALYSIS_STEPS[step]?.agent || 'Processing…'}
          </div>
          <div style={{ height: 3, background: T.bgElevated, borderRadius: 2, overflow: 'hidden' }}>
            <div style={{
              height: '100%', background: T.accentIndigo, borderRadius: 2,
              width: `${((step + 1) / 8) * 100}%`, transition: 'width 400ms ease',
              animation: 'pulseGlow 2s infinite',
            }} />
          </div>
        </div>
      )}

      {appState === 'results' && cost > 0 && (
        <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary, marginTop: 'auto' }}>
          Estimated cost: ${cost.toFixed(4)}
        </div>
      )}
    </aside>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: LANDING PAGE
   ═══════════════════════════════════════════════════════════════════════════ */
function FeatureCard({ icon, title, desc, delay }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        padding: 28, borderRadius: 8, cursor: 'default',
        border: hover ? `1px solid ${T.borderStrong}` : '1px solid transparent',
        background: hover ? T.bgElevated : 'transparent',
        transform: hover ? 'translateY(-2px)' : 'translateY(0)',
        transition: 'all 200ms ease',
        animation: `fadeInUp 500ms ease ${delay}ms both`,
      }}
    >
      <div style={{ fontSize: 36, marginBottom: 12, color: hover ? T.accentIndigo : T.textTertiary, transition: 'color 200ms' }}>
        {icon}
      </div>
      <div style={{ fontFamily: T.fontMono, fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1.5, color: T.textPrimary, marginBottom: 8 }}>
        {title}
      </div>
      <div style={{ fontSize: 14, color: T.textSecondary, lineHeight: 1.7 }}>{desc}</div>
    </div>
  );
}

function Landing() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100%', padding: '60px 40px', textAlign: 'center' }}>
      <div style={{ fontFamily: T.fontDisplay, fontSize: 48, fontWeight: 700, color: T.textPrimary, animation: 'fadeIn 600ms ease both' }}>
        LitLens
      </div>
      <div style={{ fontFamily: T.fontBody, fontSize: 18, fontStyle: 'italic', color: T.textSecondary, marginTop: 8, animation: 'fadeIn 600ms ease 200ms both' }}>
        Know your literature, faster
      </div>

      <div className="ll-features" style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12,
        maxWidth: 900, marginTop: 56, width: '100%',
      }}>
        <FeatureCard icon="◉" title="Synthesize" delay={400}
          desc="Upload 10–50 papers and get structured extraction of every claim, method, and finding." />
        <FeatureCard icon="⬡" title="Discover" delay={550}
          desc="Surface contradictions, score evidence strength, and identify what the field has missed." />
        <FeatureCard icon="△" title="Draft" delay={700}
          desc="Generate a thematic, citation-ready literature review draft in minutes." />
      </div>

      <div style={{
        marginTop: 64, fontFamily: T.fontMono, fontSize: 12, color: T.textTertiary,
        animation: 'fadeIn 600ms ease 900ms both',
        letterSpacing: 1,
      }}>
        ↑ upload your papers to begin
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: ANALYSIS PROGRESS (main area)
   ═══════════════════════════════════════════════════════════════════════════ */
function AnalysisProgress({ step }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100%', gap: 32 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {ANALYSIS_STEPS.map((s, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: i === step ? 16 : 10, height: i === step ? 16 : 10,
              borderRadius: '50%',
              background: i < step ? T.accentIndigo : i === step ? T.accentIndigo : 'transparent',
              border: i <= step ? 'none' : `1.5px solid ${T.textTertiary}`,
              transition: 'all 300ms ease',
              animation: i === step ? 'pulseDot 1.5s infinite ease-in-out' : 'none',
            }} />
            {i < ANALYSIS_STEPS.length - 1 && (
              <div style={{ display: 'none' }} />
            )}
          </div>
        ))}
      </div>
      <div style={{ fontFamily: T.fontMono, fontSize: 13, color: T.textSecondary, textAlign: 'center' }}>
        {ANALYSIS_STEPS[step]?.agent || 'Finishing up…'}
      </div>
      <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary }}>
        Agent {step + 1} of {ANALYSIS_STEPS.length}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB BAR
   ═══════════════════════════════════════════════════════════════════════════ */
function TabBar({ active, setActive, tabReady }) {
  return (
    <div style={{
      display: 'flex', gap: 0, borderBottom: `1px solid ${T.borderSubtle}`,
      animation: 'slideDown 300ms ease both', overflowX: 'auto',
    }}>
      {TAB_LABELS.map((label, i) => {
        const isActive = active === i;
        const ready = tabReady[i];
        return (
          <button key={i} onClick={() => ready && setActive(i)}
            style={{
              background: 'none', border: 'none',
              borderBottom: isActive ? `2px solid ${T.accentIndigo}` : '2px solid transparent',
              padding: '12px 20px', cursor: ready ? 'pointer' : 'not-allowed',
              fontFamily: T.fontMono, fontSize: 11, fontWeight: 500,
              textTransform: 'uppercase', letterSpacing: 2,
              color: isActive ? '#fff' : ready ? T.textTertiary : 'rgba(82,84,92,.4)',
              transition: 'all 150ms ease', whiteSpace: 'nowrap',
              display: 'flex', alignItems: 'center', gap: 8,
              opacity: ready ? 1 : 0.5,
            }}
            aria-label={`${label} tab`}
            onMouseEnter={e => { if (ready && !isActive) e.target.style.color = T.textSecondary; }}
            onMouseLeave={e => { if (!isActive) e.target.style.color = ready ? T.textTertiary : 'rgba(82,84,92,.4)'; }}
          >
            {label}
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: ready ? T.accentGreen : T.textTertiary,
              transition: 'background 300ms',
              flexShrink: 0,
            }} />
          </button>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB 1 — OVERVIEW
   ═══════════════════════════════════════════════════════════════════════════ */
function MetricCard({ label, value, color, delay, animated }) {
  return (
    <div style={{
      background: T.bgSurface, border: `1px solid ${T.borderSubtle}`,
      borderRadius: 8, padding: 28, textAlign: 'center',
      animation: `fadeInUp 400ms ease ${delay}ms both`,
    }}>
      <div style={{ fontFamily: T.fontDisplay, fontSize: 52, fontWeight: 700, color, lineHeight: 1 }}>
        <AnimatedNumber value={value} duration={800} active={animated} />
      </div>
      <div style={{ fontFamily: T.fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: 2, color: T.textTertiary, marginTop: 10 }}>
        {label}
      </div>
    </div>
  );
}

function OverviewTab({ data, animated }) {
  const r = data.report;
  return (
    <div style={{ padding: '28px 0', animation: 'fadeIn 400ms ease 200ms both' }}>
      <div className="ll-metrics" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 32 }}>
        <MetricCard label="Papers Analyzed" value={r.n_papers} color={T.accentIndigo} delay={0} animated={animated} />
        <MetricCard label="Claims Extracted" value={r.n_claims} color={T.accentGreen} delay={100} animated={animated} />
        <MetricCard label="Contradictions" value={r.n_contradictions} color={T.accentAmber} delay={200} animated={animated} />
        <MetricCard label="Gaps Identified" value={r.n_gaps} color={T.accentRed} delay={300} animated={animated} />
      </div>

      <div style={{
        borderLeft: `3px solid ${T.accentIndigo}`, padding: '16px 20px',
        background: T.bgSurface, borderRadius: '0 6px 6px 0',
      }}>
        <div style={{ fontFamily: T.fontBody, fontStyle: 'italic', fontSize: 15, color: T.textSecondary, marginBottom: 10 }}>
          "{data.researchQuestion || MOCK.researchQuestion}"
        </div>
        <div style={{ fontSize: 14, color: T.textPrimary, lineHeight: 1.8 }}>
          {r.executive_summary}
        </div>
      </div>

      <details style={{ marginTop: 24 }}>
        <summary style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary, cursor: 'pointer', textTransform: 'uppercase', letterSpacing: 1 }}>
          Agent Pipeline Trace
        </summary>
        <div style={{ marginTop: 10, padding: '12px 16px', background: T.bgSurface, borderRadius: 6 }}>
          {data.agent_history.map((h, i) => (
            <div key={i} style={{ fontFamily: T.fontMono, fontSize: 12, color: T.textSecondary, padding: '4px 0' }}>→ {h}</div>
          ))}
        </div>
      </details>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB 2 — CONTRADICTIONS
   ═══════════════════════════════════════════════════════════════════════════ */
function ContradictionCard({ ct, index }) {
  const [open, setOpen] = useState(false);
  const borderColors = { 'direct contradiction': T.accentRed, 'partial disagreement': T.accentAmber, 'different context': T.accentBlue };
  const borderColor = borderColors[ct.severity] || T.borderStrong;
  return (
    <div style={{
      borderLeft: `3px solid ${borderColor}`, background: T.bgSurface,
      borderRadius: '0 8px 8px 0', overflow: 'hidden', marginBottom: 12,
      animation: `fadeInUp 400ms ease ${index * 80}ms both`,
    }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'none', border: 'none', padding: '16px 20px',
        display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer', textAlign: 'left',
      }}>
        <SeverityBadge severity={ct.severity} />
        <span style={{ fontFamily: T.fontBody, fontSize: 15, color: T.textPrimary, flex: 1 }}>
          {ct.topic}
        </span>
        <span style={{ color: T.textTertiary, fontSize: 14, transition: 'transform 200ms', transform: open ? 'rotate(180deg)' : 'rotate(0)' }}>▾</span>
      </button>

      <div style={{
        maxHeight: open ? 600 : 0, opacity: open ? 1 : 0,
        transition: 'max-height 300ms ease, opacity 300ms ease', overflow: 'hidden',
      }}>
        <div className="ll-contra-cols" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, padding: '0 20px 20px' }}>
          <div>
            <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary, marginBottom: 6 }}>
              {ct.paper_a.title} ({ct.paper_a.year})
            </div>
            <div style={{ fontSize: 14, color: T.textPrimary, lineHeight: 1.7 }}>{ct.paper_a.position}</div>
          </div>
          <div>
            <div style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary, marginBottom: 6 }}>
              {ct.paper_b.title} ({ct.paper_b.year})
            </div>
            <div style={{ fontSize: 14, color: T.textPrimary, lineHeight: 1.7 }}>{ct.paper_b.position}</div>
          </div>
        </div>
        <div style={{ padding: '0 20px 16px', fontStyle: 'italic', fontSize: 13, color: T.textSecondary, lineHeight: 1.7 }}>
          {ct.possible_explanation}
        </div>
      </div>
    </div>
  );
}

function ContradictionsTab({ data }) {
  const papers = data.papers || [];
  const contradictions = data.contradictions || [];
  if (papers.length < 5) {
    return (
      <div style={{ padding: '40px 0', textAlign: 'center' }}>
        <div style={{ fontFamily: T.fontMono, fontSize: 12, color: T.accentAmber, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
          Minimum requirement not met
        </div>
        <div style={{ fontSize: 15, color: T.textSecondary }}>
          Contradiction analysis requires 5 or more papers. You uploaded {papers.length}.
        </div>
      </div>
    );
  }
  return (
    <div style={{ padding: '24px 0', animation: 'fadeIn 300ms ease both' }}>
      {contradictions.map((ct, i) => <ContradictionCard key={i} ct={ct} index={i} />)}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB 3 — METHODOLOGY
   ═══════════════════════════════════════════════════════════════════════════ */
function MethodologyTab({ data }) {
  const raw = data.methodology || [];
  const patterns = data.methodologyPatterns || [];
  const [sortCol, setSortCol] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);

  const keyMap = { paper_title: ['paper_title','title','paper','name'], year: ['year','date'],
    study_design: ['study_design','design','methodology','method','research_design'],
    sample_size: ['sample_size','sample','n','participants'],
    data_collection_method: ['data_collection_method','data_collection','collection','data_method'],
    key_strength: ['key_strength','strength','strengths'],
    key_limitation: ['key_limitation','limitation','limitations','weakness'] };

  function resolve(row, canonical) {
    const alts = keyMap[canonical] || [canonical];
    for (const k of alts) { if (row[k] !== undefined && row[k] !== null) return String(row[k]); }
    return '—';
  }

  const meth = raw.map(row => {
    const out = {};
    for (const k of Object.keys(keyMap)) out[k] = resolve(row, k);
    return out;
  });

  const headers = ['paper_title', 'year', 'study_design', 'sample_size', 'data_collection_method', 'key_strength', 'key_limitation'];
  const headerLabels = ['Paper', 'Year', 'Design', 'Sample', 'Data Collection', 'Strength', 'Limitation'];

  const sorted = [...meth].sort((a, b) => {
    if (!sortCol) return 0;
    return sortAsc ? (a[sortCol]||'').localeCompare(b[sortCol]||'') : (b[sortCol]||'').localeCompare(a[sortCol]||'');
  });
  const toggleSort = (col) => { if (sortCol === col) setSortAsc(p => !p); else { setSortCol(col); setSortAsc(true); } };

  const thStyle = {
    fontFamily: T.fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: 1,
    padding: '10px 12px', textAlign: 'left', color: T.textTertiary,
    position: 'sticky', top: 0, background: T.bgSurface, cursor: 'pointer',
    borderBottom: `1px solid ${T.borderSubtle}`, whiteSpace: 'nowrap', userSelect: 'none',
  };

  if (!raw.length) return (
    <div style={{ padding: '40px 0', textAlign: 'center', color: T.textTertiary, fontFamily: T.fontMono, fontSize: 13 }}>
      No methodology data available.
    </div>
  );

  return (
    <div style={{ padding: '24px 0', animation: 'fadeIn 300ms ease both' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <button onClick={() => downloadCSV(meth, 'methodology_comparison.csv')} style={{
          background: 'none', border: `1px solid ${T.borderStrong}`, borderRadius: 4,
          padding: '6px 14px', fontFamily: T.fontMono, fontSize: 11, color: T.textSecondary,
          cursor: 'pointer', transition: 'all 150ms',
        }}
          onMouseEnter={e => { e.target.style.borderColor = T.accentIndigo; e.target.style.color = T.textPrimary; }}
          onMouseLeave={e => { e.target.style.borderColor = T.borderStrong; e.target.style.color = T.textSecondary; }}
        >
          Download CSV
        </button>
      </div>

      <div style={{ overflowX: 'auto', borderRadius: 6, border: `1px solid ${T.borderSubtle}` }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 700 }}>
          <thead>
            <tr>{headers.map((h, i) => (
              <th key={h} style={thStyle} onClick={() => toggleSort(h)}>
                {headerLabels[i]} {sortCol === h ? (sortAsc ? '↑' : '↓') : ''}
              </th>
            ))}</tr>
          </thead>
          <tbody>
            {sorted.map((row, ri) => (
              <tr key={ri} style={{ background: ri % 2 === 0 ? 'transparent' : 'rgba(255,255,255,.015)' }}>
                {headers.map((h, ci) => (
                  <td key={h} style={{
                    padding: '10px 12px', fontSize: 13,
                    color: ci === 0 ? T.textPrimary : T.textSecondary,
                    fontWeight: ci === 0 ? 600 : 400,
                    borderBottom: `1px solid ${T.borderSubtle}`,
                  }}>
                    {row[h] || '—'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {patterns.length > 0 && (
        <div style={{
          borderLeft: `3px solid ${T.accentAmber}`, padding: '14px 18px',
          background: T.bgSurface, borderRadius: '0 6px 6px 0', marginTop: 20,
        }}>
          <div style={{ fontFamily: T.fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, color: T.accentAmber, marginBottom: 8 }}>
            ⚠ Pattern Detected
          </div>
          {patterns.map((p, i) => (
            <div key={i} style={{ fontSize: 13, color: T.textSecondary, padding: '3px 0', lineHeight: 1.7 }}>• {p}</div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB 4 — EVIDENCE
   ═══════════════════════════════════════════════════════════════════════════ */
function EvidenceTab({ data }) {
  const scores = [...(data.evidence_scores || [])].sort((a, b) => (b.evidence_score || 0) - (a.evidence_score || 0));
  return (
    <div style={{ padding: '24px 0', animation: 'fadeIn 300ms ease both' }}>
      {scores.map((s, i) => {
        const sc = s.evidence_score || 0;
        const flagged = s.flag && s.flag.includes('poorly');
        return (
          <div key={i} style={{
            display: 'flex', alignItems: 'flex-start', gap: 20, padding: '18px 0',
            borderBottom: i < scores.length - 1 ? `1px solid ${T.borderSubtle}` : 'none',
            animation: `fadeInUp 400ms ease ${i * 60}ms both`,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 15, color: T.textPrimary, lineHeight: 1.6, marginBottom: 8 }}>
                {s.claim_text}
              </div>
              <div style={{
                height: 4, background: T.bgElevated, borderRadius: 2, overflow: 'hidden',
                marginBottom: 8,
              }}>
                <div style={{
                  height: '100%', borderRadius: 2,
                  background: scoreColor(sc),
                  boxShadow: scoreGlow(sc),
                  width: `${sc}%`,
                  transformOrigin: 'left',
                  animation: `fillBar 800ms ease ${200 + i * 60}ms both`,
                }} />
              </div>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                {flagged && (
                  <span style={{
                    fontFamily: T.fontMono, fontSize: 10, fontWeight: 600, letterSpacing: 0.5,
                    background: T.accentAmberDim, color: T.accentAmber, padding: '3px 10px', borderRadius: 100,
                  }}>
                    WIDELY CITED, WEAK EVIDENCE
                  </span>
                )}
                <span style={{ fontFamily: T.fontMono, fontSize: 11, color: T.textTertiary }}>
                  supported by {s.supporting_papers} paper{s.supporting_papers !== 1 ? 's' : ''}
                </span>
              </div>
            </div>
            <div style={{
              fontFamily: T.fontDisplay, fontSize: 40, fontWeight: 700,
              color: scoreColor(sc), lineHeight: 1, minWidth: 60, textAlign: 'right',
            }}>
              {sc}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB 5 — GAPS
   ═══════════════════════════════════════════════════════════════════════════ */
function SimpleMarkdown({ text }) {
  if (!text) return null;
  const lines = text.split('\n');
  return lines.map((line, i) => {
    const trimmed = line.trim();
    if (!trimmed) return <div key={i} style={{ height: 8 }} />;
    if (trimmed.startsWith('### '))
      return <h4 key={i} style={{ fontSize: 15, fontWeight: 700, color: T.textPrimary, margin: '16px 0 6px' }}>{trimmed.slice(4)}</h4>;
    if (trimmed.startsWith('## '))
      return <h3 key={i} style={{ fontSize: 17, fontWeight: 700, color: T.textPrimary, margin: '20px 0 8px' }}>{trimmed.slice(3)}</h3>;
    const rendered = trimmed
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code style="background:rgba(255,255,255,.06);padding:1px 5px;border-radius:3px;font-size:12px">$1</code>');
    if (trimmed.startsWith('- ') || trimmed.startsWith('• '))
      return <div key={i} style={{ paddingLeft: 16, fontSize: 14, color: T.textSecondary, lineHeight: 1.8 }} dangerouslySetInnerHTML={{ __html: '&#8226; ' + rendered.slice(2) }} />;
    return <div key={i} style={{ fontSize: 14, color: T.textPrimary, lineHeight: 1.8 }} dangerouslySetInnerHTML={{ __html: rendered }} />;
  });
}


function GapsTab({ data }) {
  const rq = data.researchQuestion || '';
  const raw = data.gap_analysis || '';

  return (
    <div style={{ padding: '24px 0', animation: 'fadeIn 300ms ease both' }}>
      {rq && (
        <div style={{
          border: `1px solid ${T.borderStrong}`, borderRadius: 6, padding: '14px 18px', marginBottom: 28,
        }}>
          <div style={{ fontFamily: T.fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: 2, color: T.textTertiary, marginBottom: 6 }}>
            Your Research Question
          </div>
          <div style={{ fontFamily: T.fontBody, fontStyle: 'italic', fontSize: 15, color: T.textSecondary, lineHeight: 1.7 }}>
            {rq}
          </div>
        </div>
      )}

      <div style={{ background: T.bgSurface, borderRadius: 8, padding: 24, border: `1px solid ${T.borderSubtle}` }}>
        <SimpleMarkdown text={raw} />
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: TAB 6 — DRAFT
   ═══════════════════════════════════════════════════════════════════════════ */
function DraftTab({ data }) {
  const [text, setText] = useState(data.literature_review_draft || '');
  const [copied, setCopied] = useState(false);
  const ref = useRef(null);

  useEffect(() => { setText(data.literature_review_draft || ''); }, [data.literature_review_draft]);

  useEffect(() => {
    if (ref.current) ref.current.style.height = Math.max(500, ref.current.scrollHeight) + 'px';
  }, [text]);

  const copy = async () => {
    try { await navigator.clipboard.writeText(text); } catch { /* fallback omitted for brevity */ }
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div style={{ padding: '24px 0', animation: 'fadeIn 300ms ease both' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ fontFamily: T.fontMono, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, color: T.textTertiary }}>
          Literature Review Draft
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={copy} style={{
            background: 'none', border: `1px solid ${T.borderStrong}`, borderRadius: 4,
            padding: '5px 12px', fontFamily: T.fontMono, fontSize: 11,
            color: copied ? T.accentGreen : T.textSecondary, cursor: 'pointer', transition: 'all 150ms',
          }}>{copied ? '✓ Copied' : 'Copy'}</button>
          <button onClick={() => downloadTxt(text, 'litlens_literature_review.txt')} style={{
            background: 'none', border: `1px solid ${T.borderStrong}`, borderRadius: 4,
            padding: '5px 12px', fontFamily: T.fontMono, fontSize: 11,
            color: T.textSecondary, cursor: 'pointer', transition: 'all 150ms',
          }}>Download .txt</button>
        </div>
      </div>

      <div style={{
        background: T.accentAmberDim, border: `1px solid rgba(245,158,11,.2)`,
        borderRadius: 4, padding: '8px 14px', marginBottom: 16,
        fontFamily: T.fontMono, fontSize: 11, color: T.accentAmber,
      }}>
        ⚠ AI-generated draft. Verify all citations and claims before academic submission.
      </div>

      <textarea ref={ref} value={text} onChange={e => setText(e.target.value)}
        aria-label="Literature review draft"
        style={{
          width: '100%', minHeight: 500, resize: 'none',
          background: T.bgSurface, border: `1px solid ${T.borderSubtle}`,
          borderRadius: 6, padding: 24,
          fontFamily: T.fontBody, fontSize: 15, lineHeight: 1.9,
          color: T.textPrimary, outline: 'none',
          overflow: 'hidden',
        }}
        onFocus={e => e.target.style.borderColor = T.accentIndigo}
        onBlur={e => e.target.style.borderColor = T.borderSubtle}
      />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: CHAT INTERFACE
   ═══════════════════════════════════════════════════════════════════════════ */
function Chat({ data }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const scrollRef = useRef(null);

  const send = async () => {
    const q = input.trim();
    if (!q) return;
    setMessages(prev => [...prev, { role: 'user', text: q }]);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, research_question: data?.researchQuestion || '' }),
      });
      const result = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', text: result.answer, sources: result.sources }]);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Could not reach the backend. Is the API server running?', sources: [] }]);
    }
  };

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  return (
    <div style={{ borderTop: `1px solid ${T.borderSubtle}`, background: T.bgSurface, padding: 16, marginTop: 24, borderRadius: '0 0 8px 8px' }}>
      {messages.length > 0 && (
        <div ref={scrollRef} style={{ maxHeight: 180, overflowY: 'auto', marginBottom: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {messages.map((m, i) => (
            <div key={i} style={{ textAlign: m.role === 'user' ? 'right' : 'left' }}>
              <span style={{ fontFamily: T.fontMono, fontSize: 10, color: T.textTertiary }}>
                {m.role === 'user' ? 'you' : 'lens'} —
              </span>
              <div style={{
                fontSize: 13, lineHeight: 1.6, marginTop: 2,
                color: m.role === 'user' ? T.textPrimary : T.textSecondary,
                fontStyle: m.role === 'assistant' ? 'italic' : 'normal',
              }}>
                {m.text}
              </div>
              {m.sources && (
                <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  {m.sources.map((s, j) => (
                    <span key={j} style={{
                      fontFamily: T.fontMono, fontSize: 10, background: T.bgElevated,
                      border: `1px solid ${T.borderSubtle}`, borderRadius: 100,
                      padding: '2px 8px', color: T.textTertiary,
                    }}>{s}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', gap: 8 }}>
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Ask anything about your papers..."
          aria-label="Chat input"
          style={{
            flex: 1, background: T.bgElevated, border: `1px solid ${T.borderStrong}`,
            borderRadius: 2, padding: '10px 14px',
            fontFamily: T.fontMono, fontSize: 13, color: T.textPrimary, outline: 'none',
          }}
          onFocus={e => e.target.style.borderColor = T.accentIndigo}
          onBlur={e => e.target.style.borderColor = T.borderStrong}
        />
        <button onClick={send} style={{
          background: T.accentIndigo, border: 'none', borderRadius: 2,
          width: 40, color: '#fff', fontSize: 16, cursor: 'pointer',
          transition: 'all 150ms',
        }}
          onMouseDown={e => e.target.style.transform = 'scale(0.95)'}
          onMouseUp={e => e.target.style.transform = 'scale(1)'}
          aria-label="Send message"
        >→</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: RESULTS VIEW
   ═══════════════════════════════════════════════════════════════════════════ */
function ResultsView({ data, tabReady }) {
  const [activeTab, setActiveTab] = useState(0);
  const [metricsAnimated, setMetricsAnimated] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setMetricsAnimated(true), 600);
    return () => clearTimeout(t);
  }, []);

  const isReal = data && data.researchQuestion !== undefined && data.researchQuestion !== MOCK.researchQuestion;
  const viewData = isReal ? data : {
    ...data,
    researchQuestion: data.researchQuestion ?? MOCK.researchQuestion,
    papers: data.papers ?? MOCK.papers,
    claims: data.claims ?? MOCK.claims,
    contradictions: data.contradictions ?? MOCK.contradictions,
    methodology: data.methodology ?? MOCK.methodology,
    methodologyPatterns: data.methodologyPatterns ?? MOCK.methodologyPatterns,
    evidence_scores: data.evidence_scores ?? MOCK.evidence_scores,
    gap_analysis: data.gap_analysis ?? MOCK.gap_analysis,
    literature_review_draft: data.literature_review_draft ?? MOCK.literature_review_draft,
    report: data.report ?? MOCK.report,
    agent_history: data.agent_history ?? MOCK.agent_history,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <TabBar active={activeTab} setActive={setActiveTab} tabReady={tabReady} />
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 4px' }}>
        {activeTab === 0 && <OverviewTab data={viewData} animated={metricsAnimated} />}
        {activeTab === 1 && <ContradictionsTab data={viewData} />}
        {activeTab === 2 && <MethodologyTab data={viewData} />}
        {activeTab === 3 && <EvidenceTab data={viewData} />}
        {activeTab === 4 && <GapsTab data={viewData} />}
        {activeTab === 5 && <DraftTab data={viewData} />}
        <Chat data={viewData} />
      </div>
      <div style={{
        padding: '12px 0', textAlign: 'center',
        fontFamily: T.fontMono, fontSize: 10, color: T.textTertiary, letterSpacing: 0.5,
      }}>
        LitLens is a research aid. Always verify claims, citations, and conclusions independently before academic use.
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION: MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */
export default function LitLens() {
  const [appState, setAppState] = useState('landing');
  const [files, setFiles] = useState([]);
  const [rq, setRq] = useState('');
  const [domain, setDomain] = useState('');
  const [analysisStep, setAnalysisStep] = useState(0);
  const [tabReady, setTabReady] = useState([false, false, false, false, false, false]);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const stepTimers = useRef([]);

  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = STYLES;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const startAnalysis = useCallback(() => {
    setAppState('analyzing');
    setAnalysisStep(0);
    setError('');
    setTabReady([false, false, false, false, false, false]);

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('research_question', rq);
    formData.append('research_domain', domain || 'General');

    let stepIdx = 0;
    const stepInterval = setInterval(() => {
      if (stepIdx < ANALYSIS_STEPS.length - 1) {
        stepIdx++;
        setAnalysisStep(stepIdx);
      }
    }, 8000);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://localhost:8000/api/analyze');
    xhr.timeout = 10 * 60 * 1000;
    xhr.onload = () => {
      clearInterval(stepInterval);
      console.log('[LitLens] XHR status:', xhr.status, 'size:', xhr.responseText.length);
      if (xhr.status === 200) {
        try {
          const result = JSON.parse(xhr.responseText);
          console.log('[LitLens] Papers:', result.papers?.length, 'Claims:', result.claims?.length);
          setData(result);
          setTabReady([true, true, true, true, true, true]);
          setAppState('results');
        } catch (e) {
          console.error('[LitLens] JSON parse error:', e);
          setError('Failed to parse server response: ' + e.message);
          setAppState('landing');
        }
      } else {
        setError('Server error: ' + xhr.status);
        setAppState('landing');
      }
    };
    xhr.onerror = () => {
      clearInterval(stepInterval);
      console.error('[LitLens] XHR error');
      setError('Network error — is the backend running on port 8000?');
      setAppState('landing');
    };
    xhr.ontimeout = () => {
      clearInterval(stepInterval);
      setError('Request timed out (5 min). Try fewer papers.');
      setAppState('landing');
    };
    xhr.send(formData);
  }, [files, rq, domain]);

  useEffect(() => {
    return () => stepTimers.current.forEach(t => clearTimeout(t));
  }, []);

  return (
    <div className="ll-layout" style={{
      display: 'grid', gridTemplateColumns: '300px 1fr',
      height: '100vh', overflow: 'hidden',
    }}>
      <Sidebar
        files={files} setFiles={setFiles} rq={rq} setRq={setRq}
        domain={domain} setDomain={setDomain}
        onAnalyze={startAnalysis} appState={appState}
        step={analysisStep} tabReady={tabReady}
        cost={data?.total_cost || 0}
      />
      <main style={{ overflowY: 'auto', padding: '0 32px' }}>
        {error && (
          <div style={{ margin: '20px 0', padding: '14px 18px', background: T.accentRedDim,
            border: `1px solid ${T.accentRed}`, borderRadius: 6, fontFamily: T.fontMono, fontSize: 13, color: T.accentRed }}>
            {error}
          </div>
        )}
        {appState === 'landing' && <Landing />}
        {appState === 'analyzing' && <AnalysisProgress step={analysisStep} />}
        {appState === 'results' && data && <ResultsView data={data} tabReady={tabReady} />}
      </main>
    </div>
  );
}
