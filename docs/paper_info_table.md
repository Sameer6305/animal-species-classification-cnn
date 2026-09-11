# Research Paper Information (Section 3 Requirement)

| Field | Details |
|---|---|
| Research Paper Title | Animal species detection and classification framework based on modified multi-scale attention mechanism and feature pyramid network |
| Authors | Chiagoziem C. Ukwuoma, Zhiguang Qin, Sophyani B. Yussif, Monday N. Happy, Grace U. Nneji, Gilbert C. Urama, Chibueze D. Ukwuoma, Nimo B. Darkwa, Harriet Agobah |
| Journal/Conference | Scientific African (Elsevier) |
| Publication Year | 2022 |
| DOI/Source | 10.1016/j.sciaf.2022.e01151 |
| Application Domain | Animal species detection and classification |
| Dataset Name | Animal-80 (used for this activity); African Wildlife Dataset (used in paper, not used here) |
| Dataset Size | Paper-reported: 45,132 train / 13,010 test. Actual Kaggle release used: 22,566 train / 6,505 test (29,071 total, pre-processing) |
| Number of Classes | 80 |
| Original Model | Two-stage network: ResNet-50 backbone + modified multi-scale attention + Feature Pyramid Network (detection stage) → DenseNet (classification stage), trained with Focal Loss |
| Reported Performance | mAP 0.87 (African Wildlife, with attention) vs. 0.85 (without attention); Animal-80 mAP improvement of ~+0.1% with attention, AP gains of 5–20% per class |